# -*- coding: utf-8 -*-
"""
Script for training an agent.
"""
from os.path import abspath, dirname, join

from alphazero.game.config import config_2048
from alphazero.models.network import NetworkCacher
from alphazero.module.replay import ReplayBuffer
from alphazero.train.self_play import run_eval, run_self_play
from alphazero.train.training import train_network

STORAGE_MODEL = join(dirname(dirname(abspath(__file__))), "zoo")


# ##: Get configuration.
config = config_2048()
config.training.store_path = STORAGE_MODEL

# ##: Prepare necessary
replay_buffer = ReplayBuffer(config.replay)
cacher = NetworkCacher()
cacher.save_network(config.factory.network_factory())

# ##: Training Loop.
for loop in range(config.training.training_step):
    print("-" * 88)
    print("Training loop ", loop + 1)

    # ##: Self play.
    run_self_play(config, cacher, replay_buffer)

    # ##: Train network.
    train_network(config, cacher, replay_buffer)

    # ##: Store model.
    if loop > 0 and loop % config.training.export == 0:
        network = cacher.load_network()
        network.save_network(config.training.store_path)

print("General evaluation ->  score: ", run_eval(config, cacher))
