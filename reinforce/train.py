# -*- coding: utf-8 -*-
"""
Script for training an agent.
"""
from os.path import abspath, dirname, join

from addons import (
    AgentConfiguration,
    AgentConfigurationDQN,
    TrainingConfigurationA2C,
    TrainingConfigurationDQN,
)
from training import A2CTraining, DQNTraining

STORAGE_MODEL = join(dirname(dirname(abspath(__file__))), "zoo")

if __name__ == "__main__":
    import argparse

    # ## ----> Get arguments.
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="task")
    subparsers.required = True

    # ## ----> Sub-parser for training.
    parser_train = subparsers.add_parser("train", help="Aide de la commande `train`")
    parser_train.add_argument("--algo", help="Which algorithm to use", required=True, type=str)
    parser_train.add_argument("--model", help="Which type of model tu use", type=str, default="dueling")
    parser_train.add_argument("--obs", help="Which observation to implement", required=True, type=str)
    parser_train.add_argument("--style", required=False, type=str, default="simple")

    # ## ----> Sub-parser for multiple training.
    parser_multi = subparsers.add_parser("multi-train", help="Aide de la commande `multi-train`")
    parser_multi.add_argument("--algo", help="Which algorithm to use", required=True, type=str)
    parser_multi.add_argument("--obs", help="Which observation to implement", required=True, type=str)

    args = parser.parse_args()

    # ## ----> Train specific algorithm.
    if args.task == "train":
        if args.algo == "dqn":
            config_agent = AgentConfigurationDQN(
                type_model=args.model,
                store_model=STORAGE_MODEL,
                learning_rate=5e-3,
                discount=0.99,
                epsilon_max=0.5,
                epsilon_min=0.01,
                epsilon_decay=0.995,
            )
            config = TrainingConfigurationDQN(
                observation_type=args.obs,
                store_history=STORAGE_MODEL,
                agent_configuration=config_agent,
                epoch=1000,
                batch_size=32,
                update_target=1,
                memory_size=5000,
                agent_type=args.style,
            )
            dqn = DQNTraining(config)
            dqn.train_model()
        elif args.algo == "a2c":
            config_agent = AgentConfiguration(store_model=STORAGE_MODEL, learning_rate=5e-3, discount=0.99)
            config = TrainingConfigurationA2C(
                observation_type=args.obs, store_history=STORAGE_MODEL, agent_configuration=config_agent, epoch=100
            )
            a2c = A2CTraining(config)
            a2c.train_model()
    # ## -----> Train multiple model.
    elif args.task == "multi-train":
        if args.algo == "dqn":
            for model in ["dense", "dueling"]:
                for style in ["simple"]:
                    print(f"Training of {model} - {style}")
                    config_agent = AgentConfigurationDQN(
                        type_model=model,
                        store_model=STORAGE_MODEL,
                        learning_rate=5e-3,
                        discount=0.99,
                        epsilon_max=0.5,
                        epsilon_min=0.01,
                        epsilon_decay=0.995,
                    )
                    config = TrainingConfigurationDQN(
                        observation_type=args.obs,
                        store_history=STORAGE_MODEL,
                        agent_configuration=config_agent,
                        epoch=100,
                        batch_size=32,
                        update_target=1,
                        memory_size=10000,
                        agent_type=style,
                    )
                dqn = DQNTraining(config)
                dqn.train_model()
    print("Finish!")
