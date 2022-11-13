# -*- coding: utf-8 -*-
"""
Set of class for training.
"""
import time
from typing import Sequence

import tensorflow as tf

from alphazero.addons.config import StochasticAlphaZeroConfig
from alphazero.addons.types import State, Trajectory
from alphazero.models.network import NetworkCacher
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


def train_network(config: StochasticAlphaZeroConfig, cacher: NetworkCacher, replay_buffer: ReplayBuffer):
    """
    Applies a training step.

    Parameters
    ----------
    config: StochasticAlphaZeroConfig
        Configuration for self play
    cacher: NetworkCacher
        List of network weights
    replay_buffer: ReplayBuffer
        Buffer for experience
    """
    # ##: Optimizer function.
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)

    # ##: Create new network with random initialization.
    network = config.factory.network_factory()

    # ##: Nth training.
    epoch_start = time.time()
    for _ in range(config.training.epochs):
        # ##: Compute targets.
        sample = replay_buffer.sample()
        batch = compute_td_target(config.replay.td_steps, config.replay.td_lambda, sample)

        # ##: Train network.
        network.train_step(batch, optimizer)

    # ##: Log.
    epoch_end = time.time()
    elapsed = (epoch_end - epoch_start) / 60.0
    print(f"Training took {elapsed:.4} minutes")

    # ##: Save the training model.
    cacher.save_network(network)
