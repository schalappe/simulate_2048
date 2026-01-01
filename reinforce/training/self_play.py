"""
Self-play data generation for Stochastic MuZero.

Generates training trajectories by playing games with MCTS.
"""

from __future__ import annotations

from numpy import max as np_max
from numpy import ndarray, zeros

from reinforce.mcts.network_search import (
    DecisionNode,
    get_policy_from_visits,
    run_network_mcts,
    select_action_from_root,
)
from reinforce.neural.network import StochasticNetwork
from twentyfortyeight.envs.twentyfortyeight import TwentyFortyEight

from .config import StochasticMuZeroConfig
from .replay_buffer import Trajectory, TransitionData


class SelfPlayActor:
    """
    Self-play actor that generates training data.

    Plays games using MCTS with the current network and records
    trajectories for training.

    Attributes
    ----------
    config : StochasticMuZeroConfig
        Training configuration.
    network : StochasticNetwork
        Neural network for MCTS guidance.
    env : TwentyFortyEight
        Game environment.
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

    def update_network(self, network: StochasticNetwork) -> None:
        """
        Update the network used for self-play.

        Parameters
        ----------
        network : StochasticNetwork
            New network weights.
        """
        self.network = network

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

            # ##>: Run MCTS.
            root = run_network_mcts(
                game_state=raw_state,
                network=self.network,
                num_simulations=self.config.num_simulations,
                exploration_weight=self.config.exploration_weight,
                add_exploration_noise=True,
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                noise_fraction=self.config.root_dirichlet_fraction,
            )

            # ##>: Get search statistics.
            temperature = self._get_temperature()
            policy = get_policy_from_visits(root, temperature=1.0)  # Always store visit proportions
            search_policy = self._policy_dict_to_array(policy, self.config.action_size)
            search_value = self._compute_search_value(root)

            # ##>: Select action.
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
        Compute the search value from the root node.

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


def generate_games(
    config: StochasticMuZeroConfig,
    network: StochasticNetwork,
    num_games: int,
    training_step: int = 0,
) -> list[Trajectory]:
    """
    Generate multiple games for training.

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

    Returns
    -------
    list[Trajectory]
        List of generated trajectories.
    """
    actor = SelfPlayActor(config, network)
    actor.set_training_step(training_step)

    trajectories = []
    for _ in range(num_games):
        traj = actor.play_game()
        trajectories.append(traj)

    return trajectories
