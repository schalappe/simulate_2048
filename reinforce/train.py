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
    parser.add_argument("--algo", help="Which algorithm to use", required=True, type=str)
    parser.add_argument("--model", help="Which type of model tu use", required=True, type=str)
    parser.add_argument("--obs", help="Which observation to implement", required=True, type=str)
    args = parser.parse_args()

    # ## ----> Training with specific algorithm.
    if args.algo == "dqn":
        config_agent = AgentConfigurationDQN(
            type_model=args.model,
            store_model=STORAGE_MODEL,
            learning_rate=5e-3,
            discount=0.95,
            epsilon_max=0.5,
            epsilon_min=0.01,
            epsilon_decay=0.999,
        )
        config = TrainingConfigurationDQN(
            observation_type=args.obs,
            store_history=STORAGE_MODEL,
            agent_configuration=config_agent,
            epoch=1000,
            batch_size=1024,
            update_target=10,
            memory_size=100000,
        )
        dqn = DQNTraining(config)
        dqn.train_model()
    print("Finish!")
