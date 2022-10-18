# -*- coding: utf-8 -*-
"""
Script for training an agent.
"""
from os.path import abspath, dirname, join

from addons import AgentConfigurationDQN, TrainingConfigurationDQN
from training import DQNTraining

STORAGE_MODEL = join(dirname(dirname(abspath(__file__))), "models")


if __name__ == "__main__":
    import argparse

    # ## ----> Get arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="Which model to warn-up", required=True, type=str)
    args = parser.parse_args()

    # ## ----> Training with specific algorithm.
    if args.algo == "dqn":
        config_agent = AgentConfigurationDQN(
            type_model="dense",
            store_model=STORAGE_MODEL,
            learning_rate=1e-4,
            discount=0.999,
            epsilon_max=1.0,
            epsilon_min=0.05,
            epsilon_decay=200,
            batch_size=32,
            memory_size=10000,
        )
        config = TrainingConfigurationDQN(
            reward_type="affine",
            observation_type="classic",
            store_history=STORAGE_MODEL,
            epoch=1000,
            update_target=10,
            agent_configuration=config_agent,
        )
        dqn = DQNTraining(config)
        dqn.train_model()
    print("Finish!")
