# -*- coding: utf-8 -*-
"""
Prioritized experience replay buffer for efficient reinforcement learning.
Implements importance sampling and priority-based experience selection.
"""
from typing import List, Tuple, Any

from numpy import array, power, max as np_max
from numpy.random import default_rng
from .sum_tree import SumTree


class PrioritizedReplayBuffer:
    """
    A prioritized experience replay buffer for reinforcement learning.

    This class implements a prioritized experience replay mechanism, where experiences are sampled based
    on their importance. This can lead to more efficient learning by focusing on the most relevant experiences.

    Notes
    -----
    The implementation is based on the paper "Prioritized Experience Replay" by Schaul et al.
    (2015) https://arxiv.org/abs/1511.05952
    """

    GENERATOR = default_rng()

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sample: float = 1e-4,
        epsilon: float = 1e-6,
    ):
        """
        Initialize the PrioritizedReplayBuffer.

        Parameters
        ----------
        capacity : int
            Maximum number of experiences that can be stored.
        alpha : float, optional
            Exponent to determine how much prioritization is used (default is 0.6).
        beta : float, optional
            Importance-sampling bias correction factor (default is 0.4).
        beta_increment_per_sample : float, optional
            Increment value for beta per sample (default is 1e-4).
        epsilon : float, optional
            Small constant to avoid zero probabilities (default is 1e-6).
        """
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.beta_increment_per_sample = beta_increment_per_sample

    def add(self, experience: Tuple[Any, ...]) -> None:
        """
        Add a new experience to the buffer with maximum priority.

        Parameters
        ----------
        experience : Tuple[Any]
            Experience to add to the buffer.
        """
        max_priority = np_max(self.tree.tree[-self.tree.capacity :]) if len(self.tree) > 0 else 1.0
        self.tree.add(max_priority, experience)

    def sample(self, batch_size: int) -> Tuple[Any, ...]:
        """
        Sample a batch of experiences from the buffer based on priorities.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        Tuple[Any, ...]
            A tuple containing batches of states, actions, rewards, next_states, dones, and importance sampling weights.
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment_per_sample)

        for i in range(batch_size):
            s = self.GENERATOR.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        sampling_probabilities = array(priorities) / self.tree.total
        is_weights = power(len(self.tree) * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return array(states), array(actions), array(rewards), array(next_states), array(dones), is_weights, indices

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
            priority = power(priority + self.epsilon, self.alpha)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.tree.data)
