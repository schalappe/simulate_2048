"""
Main training orchestrator for Stochastic MuZero.

This module provides the Trainer class that coordinates:
- Self-play game generation
- Replay buffer management
- Training loop execution
- Logging and checkpointing
- Evaluation
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
from tqdm import tqdm

from reinforce.training.config import TrainConfig, default_config
from reinforce.training.learner import (
    CheckpointManager,
    TrainState,
    create_train_state,
    train_step,
)
from reinforce.training.replay_buffer import ReplayBuffer
from reinforce.training.self_play import evaluate_games, generate_games

# ##>: Type aliases.
PRNGKey = jax.Array


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    step: int = 0
    total_loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    reward_loss: float = 0.0
    chance_loss: float = 0.0
    games_played: int = 0
    avg_reward: float = 0.0
    avg_max_tile: float = 0.0
    steps_per_second: float = 0.0


@dataclass
class Trainer:
    """
    Main training orchestrator for Stochastic MuZero.

    Coordinates self-play, training, logging, and checkpointing
    into a unified training loop.

    Attributes
    ----------
    config : TrainConfig
        Training configuration.
    checkpoint_dir : str | Path
        Directory for saving checkpoints.
    log_dir : str | Path | None
        Directory for logs. If None, logging is disabled.
    """

    config: TrainConfig
    checkpoint_dir: str | Path = 'checkpoints'
    log_dir: str | Path | None = None
    _state: TrainState | None = field(default=None, repr=False)
    _buffer: ReplayBuffer | None = field(default=None, repr=False)
    _checkpoint_manager: CheckpointManager | None = field(default=None, repr=False)
    _metrics_history: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Initialize trainer components."""
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.log_dir is not None:
            self.log_dir = Path(self.log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # ##>: Initialize checkpoint manager.
        self._checkpoint_manager = CheckpointManager(self.checkpoint_dir)

    def initialize(self, seed: int | None = None) -> None:
        """
        Initialize training state and replay buffer.

        Parameters
        ----------
        seed : int | None
            Random seed. If None, uses config seed.
        """
        if seed is None:
            seed = self.config.seed

        key = jax.random.PRNGKey(seed)

        # ##>: Try to restore from checkpoint.
        self._state = self._checkpoint_manager.restore_train_state(self.config)

        if self._state is None:
            # ##>: Create fresh training state.
            self._state = create_train_state(self.config, key)
            print('Initialized new training state at step 0')
        else:
            print(f'Restored training state from step {self._state.step}')

        # ##>: Initialize replay buffer.
        self._buffer = ReplayBuffer(self.config)

    def fill_buffer(self, num_games: int | None = None, show_progress: bool = True) -> None:
        """
        Fill the replay buffer with initial games.

        Parameters
        ----------
        num_games : int | None
            Number of games to generate. If None, fills to min_buffer_size.
        show_progress : bool
            Whether to show progress bar.
        """
        if self._state is None or self._buffer is None:
            raise RuntimeError('Trainer not initialized. Call initialize() first.')

        if num_games is None:
            num_games = self.config.min_buffer_size

        # ##>: Generate games.
        key, subkey = jax.random.split(self._state.key)
        self._state = self._state._replace(key=key)

        trajectories = generate_games(
            params=self._state.network.params,
            apply_fns=self._state.network.apply_fns,
            key=subkey,
            config=self.config,
            num_games=num_games,
            training_step=self._state.step,
            show_progress=show_progress,
        )

        # ##>: Add to buffer.
        for traj in trajectories:
            self._buffer.add(traj)

        stats = self._buffer.get_statistics()
        print(f'Buffer filled: {stats["size"]} games, avg reward: {stats["avg_reward"]:.1f}')

    def train(
        self,
        num_steps: int | None = None,
        show_progress: bool = True,
    ) -> dict:
        """
        Run the main training loop.

        Parameters
        ----------
        num_steps : int | None
            Number of training steps. If None, runs to config.training_steps.
        show_progress : bool
            Whether to show progress bar.

        Returns
        -------
        dict
            Final training statistics.
        """
        if self._state is None or self._buffer is None:
            raise RuntimeError('Trainer not initialized. Call initialize() first.')

        if num_steps is None:
            num_steps = self.config.training_steps - self._state.step

        start_step = self._state.step
        end_step = start_step + num_steps

        # ##>: Ensure buffer has minimum samples.
        if not self._buffer.is_ready():
            print('Buffer not ready, generating initial games...')
            self.fill_buffer()

        # ##>: Training loop.
        pbar = tqdm(total=num_steps, desc='Training', unit='step', initial=0) if show_progress else None

        start_time = time.time()
        last_log_time = start_time

        while self._state.step < end_step:
            step = self._state.step

            # ##>: Generate new games periodically.
            if step > 0 and step % 100 == 0:
                self._generate_games(1)

            # ##>: Sample batch and train.
            batch, weights = self._buffer.sample_batch(self.config.batch_size)
            self._state, loss_output = train_step(self._state, batch, self.config)

            # ##>: Logging.
            if step > 0 and step % self.config.log_interval == 0:
                current_time = time.time()
                steps_per_sec = self.config.log_interval / (current_time - last_log_time)
                last_log_time = current_time

                metrics = TrainingMetrics(
                    step=step,
                    total_loss=float(loss_output.total_loss),
                    policy_loss=float(loss_output.policy_loss),
                    value_loss=float(loss_output.value_loss),
                    reward_loss=float(loss_output.reward_loss),
                    chance_loss=float(loss_output.chance_loss),
                    steps_per_second=steps_per_sec,
                )
                self._metrics_history.append(metrics)

                if pbar is not None:
                    pbar.set_postfix(
                        loss=f'{metrics.total_loss:.4f}',
                        p=f'{metrics.policy_loss:.4f}',
                        v=f'{metrics.value_loss:.4f}',
                        sps=f'{steps_per_sec:.1f}',
                    )

            # ##>: Checkpointing.
            if step > 0 and step % self.config.checkpoint_interval == 0:
                self._checkpoint_manager.save(self._state, step)

            # ##>: Evaluation.
            if step > 0 and step % self.config.eval_interval == 0:
                eval_results = self.evaluate()
                if pbar is not None:
                    pbar.write(
                        f'Step {step}: mean_reward={eval_results["mean_reward"]:.1f}, '
                        f'max_tile={eval_results["max_tile"]}'
                    )

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        # ##>: Final checkpoint.
        self._checkpoint_manager.save(self._state, self._state.step)

        total_time = time.time() - start_time
        return {
            'total_steps': num_steps,
            'total_time_seconds': total_time,
            'steps_per_second': num_steps / total_time,
            'final_step': self._state.step,
        }

    def _generate_games(self, num_games: int) -> None:
        """Generate and add games to buffer."""
        key, subkey = jax.random.split(self._state.key)
        self._state = self._state._replace(key=key)

        trajectories = generate_games(
            params=self._state.network.params,
            apply_fns=self._state.network.apply_fns,
            key=subkey,
            config=self.config,
            num_games=num_games,
            training_step=self._state.step,
            show_progress=False,
        )

        for traj in trajectories:
            self._buffer.add(traj)

    def evaluate(self, num_games: int | None = None) -> dict:
        """
        Evaluate current agent performance.

        Parameters
        ----------
        num_games : int | None
            Number of evaluation games. If None, uses config.eval_games.

        Returns
        -------
        dict
            Evaluation statistics.
        """
        if self._state is None:
            raise RuntimeError('Trainer not initialized. Call initialize() first.')

        if num_games is None:
            num_games = self.config.eval_games

        key, subkey = jax.random.split(self._state.key)
        self._state = self._state._replace(key=key)

        return evaluate_games(
            params=self._state.network.params,
            apply_fns=self._state.network.apply_fns,
            key=subkey,
            config=self.config,
            num_games=num_games,
        )

    def get_metrics_history(self) -> list[TrainingMetrics]:
        """Return the metrics history."""
        return self._metrics_history

    def get_buffer_stats(self) -> dict:
        """Return replay buffer statistics."""
        if self._buffer is None:
            return {}
        return self._buffer.get_statistics()

    @property
    def current_step(self) -> int:
        """Return current training step."""
        if self._state is None:
            return 0
        return self._state.step

    @property
    def network(self):
        """Return current network."""
        if self._state is None:
            return None
        return self._state.network


def train_muzero(
    config: TrainConfig | None = None,
    checkpoint_dir: str = 'checkpoints',
    num_steps: int | None = None,
    seed: int = 42,
) -> Trainer:
    """
    Convenience function to run full Stochastic MuZero training.

    Parameters
    ----------
    config : TrainConfig | None
        Training configuration. If None, uses default.
    checkpoint_dir : str
        Directory for checkpoints.
    num_steps : int | None
        Number of training steps. If None, uses config.training_steps.
    seed : int
        Random seed.

    Returns
    -------
    Trainer
        The trained trainer instance.
    """
    if config is None:
        config = default_config()

    trainer = Trainer(config=config, checkpoint_dir=checkpoint_dir)
    trainer.initialize(seed=seed)
    trainer.train(num_steps=num_steps)

    return trainer
