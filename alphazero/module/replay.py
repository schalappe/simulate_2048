# -*- coding: utf-8 -*-
"""
Class used to hold experience.
"""
from collections import deque
from typing import Sequence, Tuple

import numpy as np
from numpy import ndarray
from numpy.random import choice

from alphazero.addons.config import BufferConfig
from alphazero.addons.types import Trajectory

from .helpers import (
    _discount_rewards,
    _discount_values,
    _n_return,
    _policies,
    _priority,
)


def compute_n_return(all_rewards: ndarray, all_values: ndarray, td_steps: int, td_lambda: float) -> ndarray:
    """
    Compute the n-step return for all trajectories.

    Parameters
    ----------
    all_rewards: ndarray
        Reward of each episode in trajectory
    all_values: ndarray
        Value of each episode in trajectory
    td_steps:
        The number n of the n-step returns
    td_lambda:
        The lambda in TD(lambda)

    Returns
    -------
    ndarray
        All return of trajectories
    """
    # ##: Compute discount elements.
    discount_rewards = _discount_rewards(all_rewards, td_steps, td_lambda)
    discount_values = _discount_values(all_values, td_steps, td_lambda)

    # ##: Compute N-step returns
    n_returns = _n_return(discount_values, discount_rewards, td_steps)
    return n_returns


def compute_priorities(search_values: ndarray, n_returns: ndarray) -> ndarray:
    """
    Compute the priority for each episode of trajectory.

    Parameters
    ----------
    search_values: ndarray
        Value of each episode in trajectory
    n_returns: ndarray
        Return of each episode in trajectory

    Returns
    -------
    ndarray
        Prioritized replay buffer
    """
    # ##: Compute priority for buffer.
    priorities = _priority(search_values, n_returns)

    # ##: Normalize priority.
    norm = np.sum(priorities)
    return priorities / norm


def compute_target(trajectory: Trajectory, td_steps: int, td_lambda: float) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Compute target for training and sampling.

    Parameters
    ----------
    trajectory: Trajectory
        Sequence of State
    td_steps:
        The number n of the n-step returns
    td_lambda:
        The lambda in TD(lambda)

    Returns
    -------
    tuple
        N-step returns, policy target and priority
    """
    all_rewards = np.array([state.reward for state in trajectory])
    all_values = np.array([state.search_stats.search_value for state in trajectory])
    all_visits = np.array(
        [
            [state.search_stats.search_policy[a] if a in state.search_stats.search_policy else 0 for a in range(4)]
            for state in trajectory
        ]
    )

    # ##: Compute N-step returns.
    n_returns = compute_n_return(all_rewards, all_values, td_steps, td_lambda)

    # ## Compute priority.
    priorities = compute_priorities(all_values, n_returns)

    # ## Compute policies.
    policies = _policies(all_visits)

    return n_returns, policies, priorities


class ReplayBuffer:
    """
    A replay buffer to hold the experience generated by the self-play.
    """

    def __init__(self, config: BufferConfig):
        self._config = config
        self._data = deque(maxlen=config.num_trajectories)

    def __len__(self) -> int:
        return len(self._data)

    def save(self, sequence: Trajectory):
        """
        Save a sequence of state.

        Parameters
        ----------
        sequence: Trajectory
            A sequence of state
        """
        # ##: Compute n-step return.
        n_returns, policies, priorities = compute_target(sequence, self._config.td_steps, self._config.td_lambda)

        self._data.append([sequence, n_returns, policies, priorities])

    def sample_trajectory(self) -> Tuple[Trajectory, ndarray, ndarray, ndarray]:
        """
        Samples a trajectory uniformly.

        Returns
        -------
        Trajectory
            A sequence of state
        """
        indice = choice(len(self._data), 1, replace=False)[0]
        return self._data[indice]

    def sample_element(self) -> Sequence:
        """
        Samples a single element from the buffer.

        Returns
        -------
        Trajectory
            A selected trajectory
        """
        # ##: Sample a trajectory.
        trajectory, n_returns, policies, priorities = self.sample_trajectory()

        # ##: Choice index.
        indice = len(trajectory)
        while indice + self._config.td_steps >= len(trajectory):
            indice = choice(len(trajectory), 1, replace=False, p=priorities)[0]

        # ##: Compute weights
        max_weight = (len(trajectory) * min(priorities)) ** -1
        weight = (len(trajectory) * priorities[indice]) ** -1

        return [trajectory[indice].observation, n_returns[indice], policies[indice], weight / max_weight]

    def sample(self) -> Sequence:
        """
        Samples a training batch.

        Returns
        -------
        Sequence
            A sequence of trajectory
        """
        return [self.sample_element() for _ in range(self._config.batch_size)]
