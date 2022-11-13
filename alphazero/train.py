# -*- coding: utf-8 -*-
"""
Script for training an agent.
"""
import concurrent.futures
import time
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
    max_cells = []
    epoch_start = time.time()
    print("Self-play is in progress ...")
    for _ in range(config.self_play.episodes // 2):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(2):
                futures.append(executor.submit(lambda: run_self_play(config, cacher, replay_buffer)))
            for future in concurrent.futures.as_completed(futures):
                max_cells.append(future.result())
    epoch_end = time.time()

    # ##: Display time
    elapsed = (epoch_end - epoch_start) / 60.0
    print("Max: {max(max_cells)} for the self-play ...")
    print(f"Self-play took {elapsed:.4} minutes")

    # ##: Train network.
    train_network(config, cacher, replay_buffer)

    # ##: Store model.
    if loop > 0 and loop % config.training.export == 0:
        network = cacher.load_network()
        network.save_network(config.training.store_path)

print("General evaluation ->  score: ", run_eval(config, cacher))
