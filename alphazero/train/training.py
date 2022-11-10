# -*- coding: utf-8 -*-
"""
Set of class for training.
"""
from typing import Sequence
import time

from alphazero.addons.config import StochasticAlphaZeroConfig
from alphazero.addons.optimizer import GCAdam
from alphazero.addons.types import State, Trajectory
from alphazero.models.network import Network
from alphazero.module.replay import ReplayBuffer


def compute_value_target(td_steps: int, td_lambda: float, trajectory: Sequence[State]) -> float:
    """
    Compute the TD value target. The value target is the discounted root value of last steps,
    plus the discounted sum of all rewards until then.

    Parameters
    ----------
    td_steps: int
        The number n of the n-step returns
    td_lambda: float
        The lambda in TD(lambda)
    trajectory: Sequence
        A sequence of states

    Returns
    -------
    The n-step value
    """
    value = trajectory[-1].search_stats.search_value * td_lambda**td_steps

    for i, state in enumerate(trajectory):
        value += state.reward * td_lambda**i

    return value


def compute_td_target(td_steps: int, td_lambda: float, trajectories: Sequence[Trajectory]) -> Sequence:
    """
    Computes the TD lambda targets given a trajectory.

    Parameters
    ----------
    td_steps:
        The number n of the n-step returns
    td_lambda:
        The lambda in TD(lambda)
    trajectories:
        A sequence of states

    Returns
    -------
    The n-step return
    """
    targets = []

    # ##: Loop over trajectory.
    for episode in trajectories:
        if len(episode) < td_steps:
            pass

        # ##: Compute value target.
        td_value = compute_value_target(td_steps, td_lambda, episode)

        # ##: Compute policy target
        sum_visits = sum(child for child in episode[0].search_stats.search_policy.values())
        td_policy = [
            episode[0].search_stats.search_policy[a] / sum_visits if a in episode[0].search_stats.search_policy else 0
            for a in range(4)
        ]

        # ##: pack all
        targets.append([episode[0].observation, td_value, td_policy])

    return targets


def train_network(config: StochasticAlphaZeroConfig, network: Network, replay_buffer: ReplayBuffer):
    """
    Applies a training step.

    Parameters
    ----------
    config: StochasticAlphaZeroConfig
        Configuration for self play
    network: Network
        Model to train
    replay_buffer: ReplayBuffer
        Buffer for experience
    """
    # ##: Optimizer function.
    optimizer = GCAdam(
        learning_rate=config.training.learning_rate,
        decay=config.training.learning_rate / config.training.epochs
    )

    # ##: Nth training.
    epoch_start = time.time()
    for _ in range(config.training.epochs):
        # ##: Compute targets.
        sample = replay_buffer.sample()
        batch = compute_td_target(config.replay.td_steps, config.replay.td_lambda, sample)

        # ##: Train network.
        network.train_step(batch, optimizer)

    epoch_end = time.time()
    elapsed = (epoch_end - epoch_start) / 60.0
    print("Training took {:.4} minutes".format(elapsed))
