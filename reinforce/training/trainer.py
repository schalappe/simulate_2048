"""
Main training orchestrator for Stochastic MuZero.

Coordinates self-play data generation and network training.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from time import time

from numpy import mean

from .config import StochasticMuZeroConfig
from .learner import StochasticMuZeroLearner
from .replay_buffer import ReplayBuffer
from .self_play import generate_games

# ##>: Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StochasticMuZeroTrainer:
    """
    Main training orchestrator.

    Coordinates:
    1. Self-play actors generating training data
    2. Learner updating network weights
    3. Checkpointing and logging

    Attributes
    ----------
    config : StochasticMuZeroConfig
        Training configuration.
    learner : StochasticMuZeroLearner
        Network trainer.
    replay_buffer : ReplayBuffer
        Experience storage.
    """

    def __init__(
        self,
        config: StochasticMuZeroConfig | None = None,
        checkpoint_dir: str | Path | None = None,
    ):
        """
        Initialize the trainer.

        Parameters
        ----------
        config : StochasticMuZeroConfig | None
            Configuration. Uses default if None.
        checkpoint_dir : str | Path | None
            Directory for checkpoints. Creates if doesn't exist.
        """
        from .config import default_2048_config

        if config is None:
            config = default_2048_config()

        self.config = config

        # ##>: Initialize components.
        self.learner = StochasticMuZeroLearner(config)
        self.replay_buffer = ReplayBuffer(
            max_size=config.replay_buffer_size,
            alpha=config.priority_alpha,
            beta=config.priority_beta,
        )

        # ##>: Set up checkpoint directory.
        if checkpoint_dir is None:
            checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ##>: Training statistics.
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._max_tiles: list[int] = []
        self._train_start_time: float | None = None

    def _fill_replay_buffer(self, min_games: int = 100) -> None:
        """
        Fill replay buffer with initial games.

        Parameters
        ----------
        min_games : int
            Minimum number of games to generate.
        """
        logger.info(f'Filling replay buffer with {min_games} games...')
        start_time = time()

        trajectories = generate_games(
            config=self.config,
            network=self.learner.get_network(),
            num_games=min_games,
            training_step=0,
        )

        for traj in trajectories:
            self.replay_buffer.add(traj)
            self._episode_rewards.append(traj.total_reward)
            self._episode_lengths.append(len(traj))
            self._max_tiles.append(traj.max_tile)

        elapsed = time() - start_time
        logger.info(f'Generated {min_games} games in {elapsed:.1f}s')
        logger.info(f'Average reward: {mean(self._episode_rewards[-min_games:]):.1f}')
        logger.info(f'Average length: {mean(self._episode_lengths[-min_games:]):.1f}')

    def _generate_self_play_games(self, num_games: int) -> None:
        """
        Generate self-play games and add to replay buffer.

        Parameters
        ----------
        num_games : int
            Number of games to generate.
        """
        trajectories = generate_games(
            config=self.config,
            network=self.learner.get_network(),
            num_games=num_games,
            training_step=self.learner.training_step,
        )

        for traj in trajectories:
            self.replay_buffer.add(traj)
            self._episode_rewards.append(traj.total_reward)
            self._episode_lengths.append(len(traj))
            self._max_tiles.append(traj.max_tile)

    def train(
        self,
        num_steps: int | None = None,
        log_interval: int = 100,
        checkpoint_interval: int | None = None,
        games_per_step: int = 1,
        callback: Callable[[int, dict], None] | None = None,
    ) -> dict[str, list[float]]:
        """
        Run the main training loop.

        Parameters
        ----------
        num_steps : int | None
            Number of training steps. Uses config default if None.
        log_interval : int
            Steps between logging.
        checkpoint_interval : int | None
            Steps between checkpoints. Uses config default if None.
        games_per_step : int
            Number of self-play games per training step.
        callback : Callable | None
            Optional callback function(step, metrics).

        Returns
        -------
        dict[str, list[float]]
            Training history.
        """
        if num_steps is None:
            num_steps = self.config.training_steps
        if checkpoint_interval is None:
            checkpoint_interval = self.config.export_network_every

        # ##>: Fill replay buffer initially.
        if len(self.replay_buffer) == 0:
            self._fill_replay_buffer(min_games=max(100, self.config.batch_size))

        logger.info(f'Starting training for {num_steps} steps...')
        self._train_start_time = time()

        history = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'reward_loss': [],
            'chance_loss': [],
            'episode_reward': [],
            'max_tile': [],
        }

        for step in range(num_steps):
            # ##>: Generate self-play games.
            if step % 10 == 0:  # Generate games periodically
                self._generate_self_play_games(games_per_step)

            # ##>: Training step.
            losses = self.learner.train_step(self.replay_buffer)

            # ##>: Record history.
            history['total_loss'].append(losses['total'])
            history['policy_loss'].append(losses['policy'])
            history['value_loss'].append(losses['value'])
            history['reward_loss'].append(losses['reward'])
            history['chance_loss'].append(losses['chance'])

            if len(self._episode_rewards) > 0:
                history['episode_reward'].append(self._episode_rewards[-1])
                history['max_tile'].append(self._max_tiles[-1])

            # ##>: Logging.
            if step > 0 and step % log_interval == 0:
                self._log_progress(step, losses)

            # ##>: Checkpointing.
            if step > 0 and step % checkpoint_interval == 0:
                self._save_checkpoint(step)

            # ##>: Callback.
            if callback is not None:
                metrics = {
                    'step': step,
                    'losses': losses,
                    'buffer_size': len(self.replay_buffer),
                    'avg_reward': mean(self._episode_rewards[-100:]) if self._episode_rewards else 0,
                }
                callback(step, metrics)

        # ##>: Final checkpoint.
        self._save_checkpoint(num_steps)
        logger.info('Training complete!')

        return history

    def _log_progress(self, step: int, losses: dict[str, float]) -> None:
        """Log training progress."""
        elapsed = time() - self._train_start_time if self._train_start_time else 0
        steps_per_sec = step / elapsed if elapsed > 0 else 0

        recent_rewards = self._episode_rewards[-100:] if self._episode_rewards else [0]
        recent_tiles = self._max_tiles[-100:] if self._max_tiles else [0]

        logger.info(
            f'Step {step:,} | '
            f'Loss: {losses["total"]:.4f} | '
            f'Policy: {losses["policy"]:.4f} | '
            f'Value: {losses["value"]:.4f} | '
            f'Reward: {mean(recent_rewards):.1f} | '
            f'MaxTile: {max(recent_tiles)} | '
            f'Buffer: {len(self.replay_buffer):,} | '
            f'{steps_per_sec:.1f} steps/s'
        )

    def _save_checkpoint(self, step: int) -> None:
        """Save a training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f'step_{step}'
        self.learner.save_checkpoint(checkpoint_path)
        logger.info(f'Saved checkpoint to {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Load a training checkpoint.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to checkpoint directory.
        """
        self.learner.load_checkpoint(checkpoint_path)
        logger.info(f'Loaded checkpoint from {checkpoint_path}')

    def evaluate(self, num_games: int = 100) -> dict[str, float]:
        """
        Evaluate the current network.

        Parameters
        ----------
        num_games : int
            Number of games to evaluate.

        Returns
        -------
        dict[str, float]
            Evaluation metrics.
        """
        from reinforce.mcts.stochastic_agent import StochasticMuZeroAgent
        from twentyfortyeight.envs.twentyfortyeight import TwentyFortyEight

        agent = StochasticMuZeroAgent(
            network=self.learner.get_network(),
            num_simulations=self.config.num_simulations,
            exploration_weight=self.config.exploration_weight,
            temperature=0.0,  # Greedy for evaluation
            add_noise=False,
        )

        env = TwentyFortyEight(encoded=False, normalize=False)
        rewards = []
        max_tiles = []
        lengths = []

        for _ in range(num_games):
            env.reset()
            total_reward = 0.0
            steps = 0

            while not env.is_finished:
                state = env._current_state
                action = agent.choose_action(state)
                _, reward, _ = env.step(action)
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            max_tiles.append(int(env._current_state.max()))
            lengths.append(steps)

        return {
            'mean_reward': float(mean(rewards)),
            'max_reward': float(max(rewards)),
            'mean_max_tile': float(mean(max_tiles)),
            'max_tile': int(max(max_tiles)),
            'mean_length': float(mean(lengths)),
        }


def train_stochastic_muzero(
    config: StochasticMuZeroConfig | None = None,
    num_steps: int = 100_000,
    checkpoint_dir: str = 'checkpoints',
    log_interval: int = 100,
    eval_interval: int = 10_000,
    eval_games: int = 10,
) -> StochasticMuZeroTrainer:
    """
    Convenience function to train Stochastic MuZero.

    Parameters
    ----------
    config : StochasticMuZeroConfig | None
        Configuration. Uses default if None.
    num_steps : int
        Number of training steps.
    checkpoint_dir : str
        Directory for checkpoints.
    log_interval : int
        Steps between logging.
    eval_interval : int
        Steps between evaluation.
    eval_games : int
        Number of games for evaluation.

    Returns
    -------
    StochasticMuZeroTrainer
        The trainer after training.
    """
    trainer = StochasticMuZeroTrainer(config=config, checkpoint_dir=checkpoint_dir)

    def eval_callback(step: int, metrics: dict) -> None:  # noqa: ARG001
        if step > 0 and step % eval_interval == 0:
            eval_metrics = trainer.evaluate(num_games=eval_games)
            logger.info(
                f'Evaluation at step {step}: '
                f'Reward: {eval_metrics["mean_reward"]:.1f} | '
                f'MaxTile: {eval_metrics["max_tile"]}'
            )

    trainer.train(
        num_steps=num_steps,
        log_interval=log_interval,
        callback=eval_callback,
    )

    return trainer
