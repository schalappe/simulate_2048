# -*- coding: utf-8 -*-
"""
Script for training an agent.
"""
from os.path import abspath, dirname, join

from module import ConfigurationDQN, DQNTraining

STORAGE_MODEL = join(dirname(dirname(abspath(__file__))), "models")


if __name__ == "__main__":
    import argparse

    # ## ----> Get arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="Which model to warn-up", required=True, type=str)
    args = parser.parse_args()

    # ## ----> Training with specific algorithm.
    if args.algo == "dqn":
        config = ConfigurationDQN(
            env_size=4,
            reward_type="affine",
            discount=0.99,
            epsilon=1.0,
            min_epsilon=0.05,
            decay_epsilon=0.999985,
            batch_size=128,
            learning_rate=1e-4,
            memory_size=100000,
            max_steps=1000000,
            target_update=10000,
        )
        dqn = DQNTraining(config, STORAGE_MODEL)
        dqn.train_model()
    print("Finish!")
