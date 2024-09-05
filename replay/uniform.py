# -*- coding: utf-8 -*-
"""
Uniform replay buffer for reinforcement learning.
"""
from collections import deque
from typing import Tuple

from numpy import array, ndarray
from numpy.random import PCG64DXSM, default_rng

from replay.buffer import ReplayBuffer

GENERATOR = default_rng(PCG64DXSM(seed=None))


class UniformReplayBuffer(ReplayBuffer):
    """
    A simple experience replay buffer.

    This class implements a circular buffer to store and sample experiences for reinforcement learning algorithms.

    Attributes
    ----------
    capacity : int
        Maximum number of experiences that can be stored in the buffer.
    buffer : deque
        Circular buffer to store experiences.
    """

    def __init__(self, capacity: int):
        """
        Initialize the ReplayBuffer.

        Parameters
        ----------
        capacity : int
            Maximum number of experiences that can be stored.
        """
        super().__init__(capacity)
        self.buffer = deque(maxlen=capacity)

    def add(self, state: ndarray, action: int, reward: float, next_state: ndarray, done: bool) -> None:
        """
        Add a new experience to the buffer.

        Parameters
        ----------
        state : ndarray
            The current state.
        action : int
            The action taken.
        reward : float
            The reward received.
        next_state : ndarray
            The resulting next state.
        done : bool
            Whether the episode has ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[ndarray, ...]:
        """
        Sample a batch of experiences from the buffer.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        Tuple[ndarray, ...]
            A tuple containing batches of states, actions, rewards, next_states, and dones.
        """
        indices = GENERATOR.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return tuple(map(array, zip(*batch)))

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
