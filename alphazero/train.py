# -*- coding: utf-8 -*-
"""
Script for training an agent.
"""
from os.path import abspath, dirname, join

from alphazero.game.config import config_2048
from alphazero.module.replay import ReplayBuffer
from alphazero.train.self_play import run_eval, run_self_play
from alphazero.train.training import train_network

STORAGE_MODEL = join(dirname(dirname(abspath(__file__))), "zoo")


# ##: Get configuration.
config = config_2048()
config.training.store_path = STORAGE_MODEL

# ##: Prepare necessary
replay_buffer = ReplayBuffer(config.replay)
network = config.factory.network_factory()

# ##: Training Loop.
for loop in range(config.loop):
    print("-" * 88)
    print("Training loop", loop + 1)
    run_self_play(config, network, replay_buffer, loop * config.training.epochs)
    train_network(config, network, replay_buffer)

    print(f"MuZero played {config.self_play.episodes * (loop + 1)}")

    if loop > 0 and loop % config.export == 0:
        # ##: Export network.
        network.save_network(config.training.store_path)
        print("General evaluation ->  score:", run_eval(config, network))
