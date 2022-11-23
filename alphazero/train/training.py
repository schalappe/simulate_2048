# -*- coding: utf-8 -*-
"""
Set of class for training.
"""
import time

import tensorflow as tf
import tqdm

from alphazero.addons.config import StochasticAlphaZeroConfig
from alphazero.models.network import Network
from alphazero.module.replay import ReplayBuffer


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
    optimizer = tf.optimizers.Adam(learning_rate=config.training.learning_rate)

    # ##: Nth training.
    epoch_start = time.time()
    with tqdm.trange(config.training.epochs) as period:
        for step in period:
            # ##: Compute targets.
            sample = replay_buffer.sample()

            # ##: Train network.
            loss = network.train_step(sample, optimizer)

            # ##: Log.
            period.set_description(f"Training: {step + 1}")
            period.set_postfix(loss=loss)

    # ##: Log.
    epoch_end = time.time()
    elapsed = (epoch_end - epoch_start) / 60.0
    print(f"Training took {elapsed:.4} minutes")
