"""
Self-play data generation for Stochastic MuZero.

Generates training trajectories by playing games with MCTS.
Supports parallel game generation using multiprocessing for faster training.
"""

from __future__ import annotations

import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from numpy import max as np_max
from numpy import ndarray, zeros
from tqdm import tqdm

from reinforce.mcts.network_search import (
    DecisionNode,
    get_policy_from_visits,
    run_network_mcts,
    select_action_from_root,
)
from reinforce.mcts.parallel_search import DecisionNode as ParDecisionNode
from reinforce.mcts.parallel_search import ParallelMCTS
from reinforce.neural.network import StochasticNetwork
from twentyfortyeight.envs.twentyfortyeight import TwentyFortyEight

from .config import StochasticMuZeroConfig
from .replay_buffer import Trajectory, TransitionData


class SelfPlayActor:
    """
    Self-play actor that generates training data.

    Plays games using MCTS with the current network and records
    trajectories for training. Supports both sequential and parallel MCTS.

    Attributes
    ----------
    config : StochasticMuZeroConfig
        Training configuration.
    network : StochasticNetwork
        Neural network for MCTS guidance.
    env : TwentyFortyEight
        Game environment.
    _parallel_mcts : ParallelMCTS | None
        Parallel MCTS instance (created lazily when using parallel mode).
    """

    def __init__(
        self,
        config: StochasticMuZeroConfig,
        network: StochasticNetwork,
    ):
        """
        Initialize the self-play actor.

        Parameters
        ----------
        config : StochasticMuZeroConfig
            Training configuration.
        network : StochasticNetwork
            Neural network for predictions.
        """
        self.config = config
        self.network = network

        # ##>: Create environment (encoded for network input, normalized rewards).
        self.env = TwentyFortyEight(encoded=True, normalize=True)

        self._training_step = 0
        self._parallel_mcts: ParallelMCTS | None = None

        # ##>: Pre-create parallel MCTS if using parallel mode.
        if config.search_mode == 'parallel':
            self._create_parallel_mcts()

    def _create_parallel_mcts(self) -> None:
        """Create or recreate the parallel MCTS instance."""
        self._parallel_mcts = ParallelMCTS(
            network=self.network,
            num_simulations=self.config.num_simulations,
            batch_size=self.config.mcts_batch_size,
            exploration_weight=self.config.exploration_weight,
            add_exploration_noise=True,
            dirichlet_alpha=self.config.root_dirichlet_alpha,
            noise_fraction=self.config.root_dirichlet_fraction,
        )

    def update_network(self, network: StochasticNetwork) -> None:
        """
        Update the network used for self-play.

        Parameters
        ----------
        network : StochasticNetwork
            New network weights.
        """
        self.network = network

        # ##>: Recreate parallel MCTS with new network.
        if self.config.search_mode == 'parallel':
            self._create_parallel_mcts()

    def set_training_step(self, step: int) -> None:
        """
        Update the training step (for temperature scheduling).

        Parameters
        ----------
        step : int
            Current training step.
        """
        self._training_step = step

    def _get_temperature(self) -> float:
        """Get action selection temperature for current training step."""
        return self.config.get_temperature(self._training_step)

    def play_game(self, seed: int | None = None) -> Trajectory:
        """
        Play a complete game and return the trajectory.

        Parameters
        ----------
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        Trajectory
            Complete game trajectory with MCTS statistics.
        """
        trajectory = Trajectory()

        # ##>: Reset environment.
        obs = self.env.reset(seed=seed)
        done = False
        total_reward = 0.0
        step = 0

        while not done and step < self.config.max_trajectory_length:
            # ##>: Get raw state for MCTS (need 4x4 array).
            raw_state = self.env._current_state

            # ##>: Run MCTS (parallel or sequential based on config).
            temperature = self._get_temperature()

            if self.config.search_mode == 'parallel' and self._parallel_mcts is not None:
                root = self._parallel_mcts.search(raw_state)
                policy = self._parallel_mcts.get_policy(root, temperature=1.0)
                search_policy = self._policy_dict_to_array(policy, self.config.action_size)
                search_value = self._compute_search_value_parallel(root)
                action = self._parallel_mcts.select_action(root, temperature=temperature)
            else:
                root = run_network_mcts(
                    game_state=raw_state,
                    network=self.network,
                    num_simulations=self.config.num_simulations,
                    exploration_weight=self.config.exploration_weight,
                    add_exploration_noise=True,
                    dirichlet_alpha=self.config.root_dirichlet_alpha,
                    noise_fraction=self.config.root_dirichlet_fraction,
                )
                policy = get_policy_from_visits(root, temperature=1.0)
                search_policy = self._policy_dict_to_array(policy, self.config.action_size)
                search_value = self._compute_search_value(root)
                action = select_action_from_root(root, temperature=temperature)

            # ##>: Take action in environment.
            next_obs, reward, done = self.env.step(action)

            # ##>: Record transition.
            transition = TransitionData(
                observation=obs,
                action=action,
                reward=reward,
                discount=0.0 if done else self.config.discount,
                search_policy=search_policy,
                search_value=search_value,
            )
            trajectory.add(transition)

            total_reward += reward
            obs = next_obs
            step += 1

        # ##>: Update trajectory metadata.
        trajectory.total_reward = total_reward
        trajectory.max_tile = int(np_max(self.env._current_state))

        # ##>: Compute initial priority (based on search value variance).
        trajectory.priority = self._compute_trajectory_priority(trajectory)

        return trajectory

    def _policy_dict_to_array(self, policy: dict[int, float], action_size: int) -> ndarray:
        """
        Convert policy dictionary to array.

        Parameters
        ----------
        policy : dict[int, float]
            Policy as dict of action -> probability.
        action_size : int
            Total number of actions.

        Returns
        -------
        ndarray
            Policy array.
        """
        policy_array = zeros(action_size)
        for action, prob in policy.items():
            if 0 <= action < action_size:
                policy_array[action] = prob

        # ##>: Normalize if needed.
        total = policy_array.sum()
        if total > 0:
            policy_array /= total

        return policy_array

    def _compute_search_value(self, root: DecisionNode) -> float:
        """
        Compute the search value from the root node (sequential MCTS).

        Uses the weighted average of child Q-values by visit count.

        Parameters
        ----------
        root : DecisionNode
            Root node after MCTS.

        Returns
        -------
        float
            Search value estimate.
        """
        if not root.children:
            return root.value

        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits == 0:
            return root.value

        weighted_value = sum(child.value_sum for child in root.children.values()) / total_visits
        return weighted_value

    def _compute_search_value_parallel(self, root: ParDecisionNode) -> float:
        """
        Compute the search value from the root node (parallel MCTS).

        Uses the weighted average of child Q-values by visit count.

        Parameters
        ----------
        root : ParDecisionNode
            Root node after parallel MCTS.

        Returns
        -------
        float
            Search value estimate.
        """
        if not root.children:
            return root.value

        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits == 0:
            return root.value

        weighted_value = sum(child.value_sum for child in root.children.values()) / total_visits
        return weighted_value

    def _compute_trajectory_priority(self, trajectory: Trajectory) -> float:
        """
        Compute initial priority for a trajectory.

        Uses variance in search values as a proxy for learning potential.

        Parameters
        ----------
        trajectory : Trajectory
            The trajectory.

        Returns
        -------
        float
            Priority value.
        """
        if len(trajectory) == 0:
            return 1.0

        values = [t.search_value for t in trajectory.transitions]
        mean_value = sum(values) / len(values)
        variance = sum((v - mean_value) ** 2 for v in values) / len(values)

        # ##>: Higher variance = more interesting trajectory.
        return max(1.0, variance)


def _save_network_weights(network: StochasticNetwork, weights_dir: Path) -> None:
    """
    Save network weights to a directory for worker processes.

    Parameters
    ----------
    network : StochasticNetwork
        Network to save.
    weights_dir : Path
        Directory to save weights.
    """
    weights_dir.mkdir(parents=True, exist_ok=True)
    network._representation.save(weights_dir / 'representation.keras')
    network._prediction.save(weights_dir / 'prediction.keras')
    network._afterstate_dynamics.save(weights_dir / 'afterstate_dynamics.keras')
    network._afterstate_prediction.save(weights_dir / 'afterstate_prediction.keras')
    network._dynamics.save(weights_dir / 'dynamics.keras')
    network._encoder.save(weights_dir / 'encoder.keras')


def _play_game_worker(args: tuple) -> Trajectory | None:
    """
    Worker function for parallel game generation.

    Loads the network from saved weights and plays a single game.
    This function runs in a separate process.

    Parameters
    ----------
    args : tuple
        Tuple of (config, weights_dir, training_step, codebook_size).

    Returns
    -------
    Trajectory | None
        The generated game trajectory, or None if generation failed.
    """
    config, weights_dir, training_step, codebook_size = args

    try:
        # ##!: Force CPU-only inference to avoid CUDA context inheritance issues.
        # ##>: When ProcessPoolExecutor forks, workers inherit an invalid GPU context.
        # ##>: Using CPU for self-play inference is efficient and avoids this conflict.
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        import tensorflow as tf

        tf.config.set_visible_devices([], 'GPU')

        # ##>: Import models to register custom Keras layers before loading.
        import reinforce.neural.models  # noqa: F401

        # ##>: Load network from weights in worker process.
        network = StochasticNetwork.from_path(
            representation_path=str(weights_dir / 'representation.keras'),
            prediction_path=str(weights_dir / 'prediction.keras'),
            afterstate_dynamics_path=str(weights_dir / 'afterstate_dynamics.keras'),
            afterstate_prediction_path=str(weights_dir / 'afterstate_prediction.keras'),
            dynamics_path=str(weights_dir / 'dynamics.keras'),
            encoder_path=str(weights_dir / 'encoder.keras'),
            codebook_size=codebook_size,
        )

        actor = SelfPlayActor(config, network)
        actor.set_training_step(training_step)
        return actor.play_game()
    except Exception as error:
        # ##>: Log error and return None instead of crashing the batch.
        import logging

        logging.error(f'Worker failed to generate game: {error}')
        return None


def generate_games(
    config: StochasticMuZeroConfig,
    network: StochasticNetwork,
    num_games: int,
    training_step: int = 0,
    show_progress: bool = False,
    num_workers: int = 1,
) -> list[Trajectory]:
    """
    Generate multiple games for training.

    Supports parallel game generation using multiprocessing when num_workers > 1.

    Parameters
    ----------
    config : StochasticMuZeroConfig
        Training configuration.
    network : StochasticNetwork
        Neural network.
    num_games : int
        Number of games to generate.
    training_step : int
        Current training step.
    show_progress : bool
        Whether to show a progress bar.
    num_workers : int
        Number of parallel workers. If 1, runs sequentially (no overhead).
        Recommended: 2-4 workers for typical training.

    Returns
    -------
    list[Trajectory]
        List of generated trajectories.

    Notes
    -----
    Performance trade-offs for parallel mode (num_workers > 1):
    - Speedup: ~2-3x with 4 workers on typical hardware
    - Memory: Each worker loads full network copy (~100MB per worker)
    - Overhead: ~1-2s startup for model serialization per batch
    - Best for: Long games or large num_games batches
    """
    # ##>: Sequential execution for single worker (avoids process overhead).
    if num_workers <= 1:
        return _generate_games_sequential(config, network, num_games, training_step, show_progress)

    return _generate_games_parallel(config, network, num_games, training_step, show_progress, num_workers)


def _generate_games_sequential(
    config: StochasticMuZeroConfig,
    network: StochasticNetwork,
    num_games: int,
    training_step: int,
    show_progress: bool,
) -> list[Trajectory]:
    """
    Generate games sequentially (original implementation).

    Parameters
    ----------
    config : StochasticMuZeroConfig
        Training configuration.
    network : StochasticNetwork
        Neural network.
    num_games : int
        Number of games to generate.
    training_step : int
        Current training step.
    show_progress : bool
        Whether to show a progress bar.

    Returns
    -------
    list[Trajectory]
        List of generated trajectories.
    """
    actor = SelfPlayActor(config, network)
    actor.set_training_step(training_step)

    trajectories = []
    progress_bar = tqdm(range(num_games), desc='Self-play', unit='game', leave=False) if show_progress else None
    game_iter = progress_bar if progress_bar is not None else range(num_games)

    for _ in game_iter:
        traj = actor.play_game()
        trajectories.append(traj)

        # ##>: Update progress bar with game stats.
        if progress_bar is not None:
            progress_bar.set_postfix(
                reward=f'{traj.total_reward:.0f}',
                tile=traj.max_tile,
                moves=len(traj),
            )

    return trajectories


def _generate_games_parallel(
    config: StochasticMuZeroConfig,
    network: StochasticNetwork,
    num_games: int,
    training_step: int,
    show_progress: bool,
    num_workers: int,
) -> list[Trajectory]:
    """
    Generate games in parallel using multiprocessing.

    Each worker loads its own copy of the network from saved weights.
    This uses ~100MB per worker, so memory usage scales with num_workers.

    Parameters
    ----------
    config : StochasticMuZeroConfig
        Training configuration.
    network : StochasticNetwork
        Neural network.
    num_games : int
        Number of games to generate.
    training_step : int
        Current training step.
    show_progress : bool
        Whether to show a progress bar.
    num_workers : int
        Number of parallel workers.

    Returns
    -------
    list[Trajectory]
        List of successfully generated trajectories (failed games are skipped).
    """
    # ##>: Use temp directory for network weights (shared across workers).
    with tempfile.TemporaryDirectory() as temp_dir:
        weights_dir = Path(temp_dir) / 'weights'
        _save_network_weights(network, weights_dir)

        # ##>: Prepare arguments for each game.
        worker_args = [(config, weights_dir, training_step, network.codebook_size) for _ in range(num_games)]

        trajectories = []
        pbar = None

        # ##>: Use ProcessPoolExecutor for true parallelism (bypasses GIL).
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_play_game_worker, args) for args in worker_args]

            try:
                if show_progress:
                    pbar = tqdm(total=num_games, desc='Self-play (parallel)', unit='game', leave=False)

                for future in as_completed(futures):
                    traj = future.result()

                    # ##>: Skip failed games (worker returns None on error).
                    if traj is None:
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    trajectories.append(traj)

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(
                            reward=f'{traj.total_reward:.0f}',
                            tile=traj.max_tile,
                            moves=len(traj),
                        )
            finally:
                if pbar is not None:
                    pbar.close()

            # ##>: Ensure executor fully shuts down before temp directory cleanup.
            executor.shutdown(wait=True)

    return trajectories
