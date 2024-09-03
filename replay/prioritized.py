# -*- coding: utf-8 -*-
"""
Prioritized replay buffer for reinforcement learning.
"""
from typing import List, Tuple

from numpy import array, float32
from numpy import max as np_max
from numpy import ndarray, zeros
from numpy.random import PCG64DXSM, default_rng

from replay.buffer import ReplayBuffer

GENERATOR = default_rng(PCG64DXSM(seed=None))


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    A prioritized experience replay buffer.

    This class implements prioritized experience replay, where experiences are sampled based on their importance.

    Attributes
    ----------
    capacity : int
        Maximum number of experiences that can be stored in the buffer.
    buffer : List
        List to store experiences and their priorities.
    priorities : np.ndarray
        Array to store the priorities of experiences.
    alpha : float
        Exponent to determine how much prioritization is used.
    epsilon : float
        Small constant to avoid zero probabilities.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-6):
        """
        Initialize the PrioritizedReplayBuffer.

        Parameters
        ----------
        capacity : int
            Maximum number of experiences that can be stored.
        alpha : float, optional
            Exponent to determine how much prioritization is used (default is 0.6).
        epsilon : float, optional
            Small constant to avoid zero probabilities (default is 1e-6).
        """
        super().__init__(capacity)
        self.buffer = []
        self.priorities = zeros(capacity, dtype=float32)
        self.alpha = alpha
        self.epsilon = epsilon

    def add(self, state: ndarray, action: int, reward: float, next_state: ndarray, done: bool) -> None:
        """
        Add a new experience to the buffer with maximum priority.

        Parameters
        ----------
        state : np.ndarray
            The current state.
        action : int
            The action taken.
        reward : float
            The reward received.
        next_state : np.ndarray
            The resulting next state.
        done : bool
            Whether the episode has ended.
        """
        max_priority = np_max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            idx = len(self.buffer) % self.capacity
            self.buffer[idx] = (state, action, reward, next_state, done)
        self.priorities[len(self.buffer) - 1] = max_priority

    def sample(self, batch_size: int) -> Tuple[ndarray, ...]:
        """
        Sample a batch of experiences from the buffer based on priorities.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        Tuple[ndarray, ...]
            A tuple containing batches of states, actions, rewards, next_states, dones,
            and importance sampling weights.

        Examples
        --------
        >>> buffer = PrioritizedReplayBuffer(1000)
        >>> buffer.add(np.array([1, 2, 3]), 0, 1.0, np.array([2, 3, 4]), False)
        >>> result = buffer.sample(1)
        >>> states, actions, rewards, next_states, dones, weights = result
        """
        probs = self.priorities[: len(self.buffer)] ** self.alpha
        probs /= probs.sum()

        indices = GENERATOR.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** -1
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (array(states), array(actions), array(rewards), array(next_states), array(dones), weights)

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities for sampled experiences.

        Parameters
        ----------
        indices : List[int]
            Indices of the experiences to update.
        priorities : List[float]
            New priority values for the experiences.

        Notes
        -----
        This method should be called after computing TD errors for a batch of samples.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
