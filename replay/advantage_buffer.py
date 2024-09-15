# -*- coding: utf-8 -*-
"""A buffer for storing and processing experience tuples for advantage estimation."""
from typing import Tuple, Any
from numpy import ndarray, array, zeros
from numpy.random import default_rng

from collections import deque


class AdvantageBuffer:
    """
    A buffer for storing and processing experience tuples for advantage estimation.

    This class implements a fixed-size buffer to store experience tuples and provides methods to
    compute advantages and sample batches of experiences.
    """

    GENERATOR = default_rng()

    def __init__(self, capacity: int, gamma: float = 0.99, lambda_: float = 0.95):
        self.capacity = capacity
        self.gamma = gamma
        self.lambda_ = lambda_
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Tuple[Any, ...]):
        """
        Add an experience tuple to the buffer.

        Parameters
        ----------
        experience : tuple of Any
            The experience tuple to add to the buffer.
        """
        self.buffer.append(experience)

    def compute_advantages(self) -> ndarray:
        """
        Compute advantage estimates for all experiences in the buffer.

        Returns
        -------
        ndarray
            Array of computed advantages for each experience in the buffer.
        """
        _, _, rewards, _, dones, values = zip(*self.buffer)
        advantages = zeros(len(rewards))
        last_advantage = 0
        last_value = values[-1]

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_advantage = delta
            else:
                delta = rewards[t] + self.gamma * last_value - values[t]
                last_advantage = delta + self.gamma * self.lambda_ * last_advantage

            advantages[t] = last_advantage
            last_value = values[t]

        return advantages

    def sample(self, batch_size: int) -> Tuple[Any, ...]:
        """
        Sample a batch of experiences from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of experiences to sample.

        Returns
        -------
        tuple of Any
            A tuple containing batches of states, actions, rewards, next_states, dones, values, and advantages.
        """
        indices = self.GENERATOR.choice(len(self.buffer), batch_size, replace=False)
        batch = [array([experience[i] for experience in self.buffer]) for i in range(5)]
        advantages = self.compute_advantages()

        return tuple(item[indices] for item in batch) + (advantages[indices],)

    def __len__(self) -> int:
        """
        Get the current number of experiences in the buffer.

        Returns
        -------
        int
            The number of experiences currently stored in the buffer.
        """
        return len(self.buffer)
