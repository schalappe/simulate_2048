# -*- coding: utf-8 -*-
"""
Set of function useful for module.
"""
import numpy as np
from numpy import ndarray
from numba import prange, njit


@njit
def _sum_reward(rewards: ndarray, td_lambda: float) -> float:
    """
    Sum reward for N-step return.

    Parameters
    ----------
    rewards: ndarray
        List of rewards
    td_lambda: float
        Discount for n-step return

    Returns
    -------
    float
        n-step return
    """
    value = 0.0

    _len = rewards.shape[0]
    for i in prange(_len):
        value += rewards[i] * td_lambda ** i

    return value


@njit
def _discount_rewards(all_rewards: ndarray, td_steps: int, td_lambda: float) -> ndarray:
    """
    Compute the discount rewards.

    Parameters
    ----------
    all_rewards: ndarray
        Reward of each episode in trajectory
    td_steps:
        The number n of the n-step returns
    td_lambda:
        The lambda in TD(lambda)

    Returns
    -------
    ndarray
        All return of trajectories
    """
    _len = all_rewards.shape[0]
    discount_rewards = np.zeros(_len)

    for i in prange(_len):
        discount_rewards[i] = _sum_reward(all_rewards[i:i+td_steps], td_lambda)

    return discount_rewards


@njit
def _discount_values(all_values: ndarray, td_steps: int, td_lambda: float) -> ndarray:
    """
    Compute the discount values.

    Parameters
    ----------
    all_values: ndarray
        Value of each episode in trajectory
    td_steps:
        The number n of the n-step returns
    td_lambda:
        The lambda in TD(lambda)

    Returns
    -------
    ndarray
        Discount values
    """
    _len = all_values.shape[0]
    discount_values = np.zeros(_len)

    for i in prange(_len):
        discount_values[i] = all_values[i] * td_lambda ** td_steps

    return discount_values


@njit
def _n_return(discount_values: ndarray, discount_rewards: ndarray, td_steps: int) -> ndarray:
    """
    Compute the n-step reward for each episode of trajectory.

    Parameters
    ----------
    discount_values: ndarray
        Discount values for each episode in trajectory
    discount_rewards: ndarray
        Discount rewards for each episode in trajectory
    td_steps:
        The number n of the n-step returns

    Returns
    -------
    ndarray
        N-step return for each episode in trajectory
    """
    _len = discount_values.shape[0]
    all_returns = np.zeros(_len)

    for i in prange(_len):
        indice = _len - 1 if i + td_steps > _len else i + td_steps
        all_returns[i] = discount_values[indice] + discount_rewards[i]

    return all_returns


@njit
def _priority(search_values: ndarray, all_returns: ndarray) -> ndarray:
    """
    Compute the priority of transition for each episode of trajectory.

    Parameters
    ----------
    search_values: ndarray
        All search values of trajectories
    all_returns: ndarray
        All return of trajectories

    Returns
    -------
    ndarray
        All priority of trajectories
    """
    _len = all_returns.shape[0]
    priorities = np.zeros(_len)

    for i in prange(_len):
        priorities[i] = abs(search_values[i] - all_returns[i])

    return priorities


@njit
def _policies(all_visits: ndarray) -> ndarray:
    """
    Compute policy for each episode in trajectory

    Parameters
    ----------
    all_visits: ndarray
        Visitor for each episode in trajectory

    Returns
    -------
    ndarray
        Policy for each episode in trajectory
    """
    _len = all_visits.shape[0]
    policies = np.zeros(all_visits.shape)

    for i in prange(_len):
        total = np.sum(all_visits[i])
        policies[i] = all_visits[i] / total

    return policies
