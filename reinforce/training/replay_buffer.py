"""
Prioritized Replay Buffer for Stochastic MuZero.

Implements a replay buffer with prioritized sampling as described in the paper.
Priority is based on the TD error: |v_search - z_target|.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from numpy import array, float32, ndarray, zeros
from numpy import sum as np_sum
from numpy.random import PCG64DXSM, default_rng

GENERATOR = default_rng(PCG64DXSM(seed=42))


@dataclass
class TransitionData:
    """
    Data for a single game transition.

    Attributes
    ----------
    observation : ndarray
        The game observation at this step.
    action : int
        The action taken.
    reward : float
        The reward received after taking the action.
    discount : float
        The discount factor (0 if terminal, else γ).
    search_policy : ndarray
        The MCTS policy (visit distribution) at this step.
    search_value : float
        The root value from MCTS search.
    """

    observation: ndarray
    action: int
    reward: float
    discount: float
    search_policy: ndarray
    search_value: float


@dataclass
class Trajectory:
    """
    A complete game trajectory.

    Attributes
    ----------
    transitions : list[TransitionData]
        List of transitions in the trajectory.
    total_reward : float
        Total episode reward (for logging).
    max_tile : int
        Maximum tile achieved (for logging).
    priority : float
        Sampling priority for this trajectory.
    """

    transitions: list[TransitionData] = field(default_factory=list)
    total_reward: float = 0.0
    max_tile: int = 0
    priority: float = 1.0

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, transition: TransitionData) -> None:
        """Add a transition to the trajectory."""
        self.transitions.append(transition)

    def get_slice(self, start: int, length: int) -> list[TransitionData]:
        """
        Get a slice of transitions starting at index.

        Parameters
        ----------
        start : int
            Starting index.
        length : int
            Number of transitions to get.

        Returns
        -------
        list[TransitionData]
            The requested transitions.
        """
        end = min(start + length, len(self.transitions))
        return self.transitions[start:end]


class ReplayBuffer:
    """
    Prioritized replay buffer for Stochastic MuZero training.

    Implements prioritized sampling based on TD error, following the paper:
    P(i) = p_i^α / Σ_k p_k^α
    where p_i = |v_search - z_target|

    Attributes
    ----------
    max_size : int
        Maximum number of trajectories to store.
    alpha : float
        Priority exponent (0 = uniform, 1 = fully prioritized).
    beta : float
        Importance sampling exponent.
    """

    def __init__(
        self,
        max_size: int = 125_000,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        """
        Initialize the replay buffer.

        Parameters
        ----------
        max_size : int
            Maximum number of trajectories to store.
        alpha : float
            Priority exponent.
        beta : float
            Importance sampling exponent.
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta

        self._trajectories: deque[Trajectory] = deque(maxlen=max_size)
        self._priorities: deque[float] = deque(maxlen=max_size)

        # ##>: Small constant for numerical stability.
        self._epsilon = 1e-6

    def __len__(self) -> int:
        return len(self._trajectories)

    @property
    def is_ready(self) -> bool:
        """Check if buffer has enough data for training."""
        return len(self._trajectories) > 0

    def add(self, trajectory: Trajectory, priority: float | None = None) -> None:
        """
        Add a trajectory to the buffer.

        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to add.
        priority : float | None
            Initial priority. If None, uses max priority.
        """
        if priority is None:
            # ##>: Use max priority for new trajectories.
            priority = max(self._priorities) if len(self._priorities) > 0 else 1.0

        trajectory.priority = priority
        self._trajectories.append(trajectory)
        self._priorities.append(priority)

    def sample_trajectory(self) -> tuple[Trajectory, int, float]:
        """
        Sample a trajectory using prioritized sampling.

        Returns
        -------
        tuple[Trajectory, int, float]
            (trajectory, index, importance_weight).
        """
        if len(self._trajectories) == 0:
            raise ValueError('Cannot sample from empty buffer')

        # ##>: Compute sampling probabilities.
        priorities = array(list(self._priorities), dtype=float32)
        probs = priorities**self.alpha
        probs = probs / np_sum(probs)

        # ##>: Sample trajectory.
        idx = GENERATOR.choice(len(self._trajectories), p=probs)

        # ##>: Compute importance sampling weight.
        n = len(self._trajectories)
        weight = (n * probs[idx]) ** (-self.beta)
        max_weight = (n * min(probs)) ** (-self.beta)
        weight = weight / max_weight  # Normalize to [0, 1]

        return self._trajectories[idx], idx, weight

    def sample_position(self, trajectory: Trajectory, unroll_steps: int, td_steps: int) -> int:
        """
        Sample a starting position within a trajectory.

        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to sample from.
        unroll_steps : int
            Number of steps to unroll.
        td_steps : int
            Number of steps for TD returns.

        Returns
        -------
        int
            Starting position index.
        """
        # ##>: Ensure we have enough steps for unrolling and TD.
        max_start = max(0, len(trajectory) - max(unroll_steps, td_steps))
        if max_start == 0:
            return 0
        return GENERATOR.integers(0, max_start + 1)

    def sample_batch(
        self,
        batch_size: int,
        unroll_steps: int,
        td_steps: int,
    ) -> tuple[list[tuple[Trajectory, int]], ndarray]:
        """
        Sample a batch of trajectories with positions.

        Parameters
        ----------
        batch_size : int
            Number of samples.
        unroll_steps : int
            Number of steps to unroll.
        td_steps : int
            Number of steps for TD returns.

        Returns
        -------
        tuple[list[tuple[Trajectory, int]], ndarray]
            List of (trajectory, start_position) pairs and importance weights.
        """
        samples = []
        weights = zeros(batch_size, dtype=float32)

        for i in range(batch_size):
            traj, idx, weight = self.sample_trajectory()
            pos = self.sample_position(traj, unroll_steps, td_steps)
            samples.append((traj, pos))
            weights[i] = weight

        return samples, weights

    def update_priorities(self, indices: list[int], td_errors: ndarray) -> None:
        """
        Update priorities based on TD errors.

        Parameters
        ----------
        indices : list[int]
            Trajectory indices to update.
        td_errors : ndarray
            TD errors for each trajectory.
        """
        for idx, td_error in zip(indices, td_errors, strict=False):
            if 0 <= idx < len(self._priorities):
                # ##>: Priority = |TD error| + ε
                new_priority = abs(td_error) + self._epsilon
                # ##>: deque doesn't support item assignment, need to reconstruct.
                # ##@: This is inefficient; consider using list or numpy array.
                priorities_list = list(self._priorities)
                priorities_list[idx] = new_priority
                self._priorities = deque(priorities_list, maxlen=self.max_size)
                self._trajectories[idx].priority = new_priority

    def get_statistics(self) -> dict[str, float]:
        """
        Get buffer statistics for logging.

        Returns
        -------
        dict[str, float]
            Statistics dictionary.
        """
        if len(self._trajectories) == 0:
            return {'size': 0, 'avg_reward': 0.0, 'avg_max_tile': 0.0}

        rewards = [t.total_reward for t in self._trajectories]
        tiles = [t.max_tile for t in self._trajectories]

        return {
            'size': len(self._trajectories),
            'avg_reward': sum(rewards) / len(rewards),
            'avg_max_tile': sum(tiles) / len(tiles),
            'max_reward': max(rewards),
            'max_tile': max(tiles),
        }
