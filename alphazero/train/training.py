# -*- coding: utf-8 -*-
"""
Set of class for training.
"""
import time

from alphazero.addons.config import StochasticAlphaZeroConfig
from alphazero.addons.optimizer import GCAdam
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
    optimizer = GCAdam(learning_rate=config.training.learning_rate)

    # ##: Nth training.
    epoch_start = time.time()
    for _ in range(config.training.epochs):
        # ##: Compute targets.
        sample = replay_buffer.sample()

        # ##: Train network.
        network.train_step(sample, optimizer)

    # ##: Log.
    epoch_end = time.time()
    elapsed = (epoch_end - epoch_start) / 60.0
    print(f"Training took {elapsed:.4} minutes")
