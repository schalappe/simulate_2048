# -*- coding: utf-8 -*-
"""
Uniform Replay Buffer implementation for efficient experience storage and sampling in Reinforcement Learning.
Provides a circular buffer with uniform sampling to break correlations between consecutive experiences.
"""
from collections import deque
from typing import Tuple, Any

from numpy import array
from numpy.random import default_rng


class UniformReplayBuffer:
    """
    A uniform experience replay buffer for reinforcement learning.

    This class implements a circular buffer to store and efficiently sample experiences for reinforcement learning
    algorithms. It provides methods to add experiences, sample random batches, and manage buffer capacity.

    The buffer uses a uniform sampling strategy, meaning all stored experiences have an equal probability
    of being selected during sampling. This approach helps to break correlations between consecutive experiences
    and stabilize learning.
    """

    GENERATOR = default_rng()

    def __init__(self, capacity: int):
        """
        Initialize the ReplayBuffer.

        Parameters
        ----------
        capacity : int
            Maximum number of experiences that can be stored.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Tuple[Any, ...]) -> None:
        """
        Add a new experience to the buffer.

        Parameters
        ----------
        experience : Tuple[Any]
            Experience to add to the buffer.
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[Any, ...]:
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
        indices = self.GENERATOR.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return tuple(map(array, zip(*batch)))

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
