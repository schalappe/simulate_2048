"""
Prioritized replay buffer for Stochastic MuZero training.

This module provides a JAX-compatible replay buffer that stores game trajectories and samples
training batches with prioritization.

Design considerations:
1. Efficient storage and sampling with NumPy (buffer lives on CPU)
2. Prioritized sampling based on TD-error or trajectory value
3. Support for n-step returns and TD(λ) target computation
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import default_rng

from reinforce.training.config import TrainConfig
from reinforce.training.losses import TrainingTargets

# ##>: Type aliases.
Array = jax.Array


class Trajectory(NamedTuple):
    """
    A complete game trajectory.

    Stores all information needed for training from a single game.

    Attributes
    ----------
    observations : np.ndarray
        Observations at each step, shape (T, observation_dim).
    actions : np.ndarray
        Actions taken at each step, shape (T,).
    rewards : np.ndarray
        Rewards received at each step, shape (T,).
    policies : np.ndarray
        MCTS policy targets at each step, shape (T, action_size).
    values : np.ndarray
        MCTS value targets at each step, shape (T,).
    done : bool
        Whether the game ended.
    total_reward : float
        Total cumulative reward.
    max_tile : int
        Maximum tile achieved.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    done: bool = True
    total_reward: float = 0.0
    max_tile: int = 0

    def __len__(self) -> int:
        """Return trajectory length."""
        return len(self.observations)


@dataclass
class ReplayBuffer:
    """
    Prioritized replay buffer for storing and sampling trajectories.

    Uses a circular buffer to store trajectories and supports
    prioritized sampling based on trajectory priority scores.

    Attributes
    ----------
    config : TrainConfig
        Training configuration.
    max_size : int
        Maximum number of trajectories to store.
    """

    config: TrainConfig
    max_size: int = field(default=125_000)
    _trajectories: list = field(default_factory=list)
    _priorities: list = field(default_factory=list)
    _position: int = field(default=0)
    _rng: np.random.Generator = field(default_factory=lambda: default_rng(42))

    def __post_init__(self):
        """Initialize the buffer."""
        self.max_size = self.config.replay_buffer_size

    def __len__(self) -> int:
        """Return number of trajectories in buffer."""
        return len(self._trajectories)

    def add(self, trajectory: Trajectory, priority: float | None = None) -> None:
        """
        Add a trajectory to the buffer.

        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to add.
        priority : float | None
            Priority for sampling. If None, uses trajectory's value variance.
        """
        if priority is None:
            priority = self._compute_priority(trajectory)

        if len(self._trajectories) < self.max_size:
            self._trajectories.append(trajectory)
            self._priorities.append(priority)
        else:
            # ##>: Replace oldest trajectory (circular buffer).
            self._trajectories[self._position] = trajectory
            self._priorities[self._position] = priority

        self._position = (self._position + 1) % self.max_size

    def _compute_priority(self, trajectory: Trajectory) -> float:
        """
        Compute priority score for a trajectory.

        Uses value variance as a proxy for learning potential.
        High variance trajectories contain more diverse value estimates.

        Parameters
        ----------
        trajectory : Trajectory
            The trajectory.

        Returns
        -------
        float
            Priority score.
        """
        if len(trajectory) == 0:
            return 1.0

        values = trajectory.values
        mean_value = np.mean(values)
        variance = np.mean((values - mean_value) ** 2)

        # ##>: Add small epsilon to ensure non-zero priority.
        return max(1.0, variance + 0.1)

    def sample_batch(self, batch_size: int, seed: int | None = None) -> tuple[TrainingTargets, np.ndarray]:
        """
        Sample a batch of training examples.

        Each example is a contiguous segment from a trajectory,
        with length K+1 for K unroll steps.

        Parameters
        ----------
        batch_size : int
            Number of examples to sample.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        tuple[TrainingTargets, np.ndarray]
            - Batch of training targets
            - Importance sampling weights
        """
        if seed is not None:
            self._rng = default_rng(seed)

        # ##>: Sample trajectories with priority-based probabilities.
        priorities = np.array(self._priorities[: len(self._trajectories)])
        probs = priorities**self.config.priority_alpha
        probs = probs / probs.sum()

        traj_indices = self._rng.choice(len(self._trajectories), size=batch_size, p=probs, replace=True)

        # ##>: For each trajectory, sample a starting position.
        unroll_length = self.config.num_unroll_steps + 1  # K steps + initial

        observations_list = []
        actions_list = []
        policies_list = []
        values_list = []
        rewards_list = []
        weights_list = []

        for idx in traj_indices:
            traj = self._trajectories[idx]
            traj_len = len(traj)

            # ##>: Sample starting position (ensure room for unroll).
            max_start = max(0, traj_len - unroll_length)
            start = 0 if max_start == 0 else self._rng.integers(0, max_start + 1)

            # ##>: Extract segment, padding if necessary.
            end = start + unroll_length

            obs_segment = traj.observations[start:end]
            act_segment = traj.actions[start : end - 1]  # K actions
            pol_segment = traj.policies[start:end]
            val_segment = traj.values[start:end]
            rew_segment = traj.rewards[start : end - 1]

            # ##>: Pad if segment is shorter than unroll_length.
            if len(obs_segment) < unroll_length:
                pad_len = unroll_length - len(obs_segment)
                obs_segment = np.pad(obs_segment, ((0, pad_len), (0, 0)), mode='edge')
                pol_segment = np.pad(pol_segment, ((0, pad_len), (0, 0)), mode='edge')
                val_segment = np.pad(val_segment, (0, pad_len), mode='edge')

            if len(act_segment) < unroll_length - 1:
                pad_len = (unroll_length - 1) - len(act_segment)
                act_segment = np.pad(act_segment, (0, pad_len), mode='edge')
                rew_segment = np.pad(rew_segment, (0, pad_len), mode='constant')

            observations_list.append(obs_segment)
            actions_list.append(act_segment)
            policies_list.append(pol_segment)
            values_list.append(val_segment)
            rewards_list.append(rew_segment)

            # ##>: Importance sampling weight.
            weight = (len(self._trajectories) * probs[idx]) ** (-self.config.priority_beta)
            weights_list.append(weight)

        # ##>: Stack into batch.
        observations = np.stack(observations_list, axis=0)
        actions = np.stack(actions_list, axis=0)
        policies = np.stack(policies_list, axis=0)
        values = np.stack(values_list, axis=0)
        rewards = np.stack(rewards_list, axis=0)
        weights = np.array(weights_list)

        # ##>: Normalize weights.
        weights = weights / weights.max()

        # ##>: Convert to JAX arrays.
        batch = TrainingTargets(
            observations=jnp.array(observations),
            actions=jnp.array(actions),
            target_policies=jnp.array(policies),
            target_values=jnp.array(values),
            target_rewards=jnp.array(rewards),
        )

        return batch, jnp.array(weights)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for sampled trajectories.

        Parameters
        ----------
        indices : np.ndarray
            Trajectory indices.
        priorities : np.ndarray
            New priority values.
        """
        for idx, priority in zip(indices, priorities, strict=True):
            if idx < len(self._priorities):
                self._priorities[idx] = priority

    def get_statistics(self) -> dict:
        """
        Get buffer statistics.

        Returns
        -------
        dict
            Statistics including size, average reward, etc.
        """
        if len(self._trajectories) == 0:
            return {
                'size': 0,
                'avg_reward': 0.0,
                'avg_max_tile': 0,
                'avg_length': 0.0,
            }

        rewards = [t.total_reward for t in self._trajectories]
        max_tiles = [t.max_tile for t in self._trajectories]
        lengths = [len(t) for t in self._trajectories]

        return {
            'size': len(self._trajectories),
            'avg_reward': float(np.mean(rewards)),
            'avg_max_tile': float(np.mean(max_tiles)),
            'avg_length': float(np.mean(lengths)),
            'max_reward': float(np.max(rewards)),
            'max_tile': int(np.max(max_tiles)),
        }

    def is_ready(self) -> bool:
        """Check if buffer has enough samples to start training."""
        return len(self._trajectories) >= self.config.min_buffer_size

    def clear(self) -> None:
        """Clear all trajectories from buffer."""
        self._trajectories.clear()
        self._priorities.clear()
        self._position = 0
