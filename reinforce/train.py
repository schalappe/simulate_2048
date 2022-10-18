# -*- coding: utf-8 -*-
"""
Script for training an agent.
"""
from os.path import abspath, dirname, join

from addons import AgentConfigurationDQN, TrainingConfigurationDQN
from training import DQNTraining

STORAGE_MODEL = join(dirname(dirname(abspath(__file__))), "zoo")

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
            learning_rate=5e-3,
            discount=0.95,
            epsilon_max=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.995,
        )
        config = TrainingConfigurationDQN(
            reward_type="affine",
            observation_type="classic",
            store_history=STORAGE_MODEL,
            agent_configuration=config_agent,
            epoch=1000,
            batch_size=32,
            update_target=10,
            memory_size=10000,
        )
        dqn = DQNTraining(config)
        dqn.train_model()
    print("Finish!")
