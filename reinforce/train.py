# -*- coding: utf-8 -*-
"""
Script for training an agent.
"""
from os.path import abspath, dirname, join

from addons import TrainingConfigurationA2C, TrainingConfigurationDQN
from training import A2CTraining, DQNDuelingTraining, DQNTraining, DDQNTraining, DDQNDuelingTraining

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

    # ## ----> Sub-parser for multiple training.
    parser_multi = subparsers.add_parser("multi-train", help="Aide de la commande `multi-train`")

    args = parser.parse_args()

    # ## ----> Train specific algorithm.
    if args.task == "train":

        # ##: Get configuration.
        if args.algo == "dqn":
            config = TrainingConfigurationDQN(
                store_history=STORAGE_MODEL,
                training_steps=10,
                learning_rate=3e-4,
                batch_size=64,
                discount=0.99,
                save_steps=101,
                replay_step=50,
                update_step=300,
                greedy_step=5,
                max_steps=5000,
                memory_size=10000,
            )
            train = DQNTraining(config)
        elif args.algo == "dqn-dueling":
            config = TrainingConfigurationDQN(
                store_history=STORAGE_MODEL,
                training_steps=10,
                learning_rate=3e-4,
                batch_size=64,
                discount=0.99,
                save_steps=101,
                replay_step=50,
                update_step=300,
                greedy_step=5,
                max_steps=5000,
                memory_size=10000,
            )
            train = DQNDuelingTraining(config)
        elif args.algo == "double-dqn":
            config = TrainingConfigurationDQN(
                store_history=STORAGE_MODEL,
                training_steps=10,
                learning_rate=3e-4,
                batch_size=64,
                discount=0.99,
                save_steps=101,
                replay_step=50,
                update_step=300,
                greedy_step=5,
                max_steps=5000,
                memory_size=10000,
            )
            train = DDQNTraining(config)
        elif args.algo == "double-dqn-dueling":
            config = TrainingConfigurationDQN(
                store_history=STORAGE_MODEL,
                training_steps=30000,
                learning_rate=3e-4,
                batch_size=64,
                discount=0.99,
                save_steps=101,
                replay_step=50,
                update_step=300,
                greedy_step=10000,
                max_steps=5000,
                memory_size=10000,
            )
            train = DDQNDuelingTraining(config)
        elif args.algo == "a2c":
            config = TrainingConfigurationA2C(
                store_history=STORAGE_MODEL,
                training_steps=500,
                learning_rate=3e-4,
                discount=0.99,
                save_steps=101,
                max_steps=5000,
            )
            train = A2CTraining(config)
        else:
            raise ValueError(f"This `{args.algo}` isn't implemented yet.")

        # ##: Train agent.
        train.train_model()
    # ## -----> Train multiple model.
    elif args.task == "multi-train":
        # ##: A2C
        config = TrainingConfigurationA2C(
            store_history=STORAGE_MODEL,
            training_steps=1000,
            learning_rate=3e-4,
            discount=0.99,
            save_steps=101,
            max_steps=5000,
        )
        train = A2CTraining(config)
        train.train_model()

        # ##: DQN Training.
        del train
        config = TrainingConfigurationDQN(
            store_history=STORAGE_MODEL,
            training_steps=1000,
            learning_rate=3e-4,
            batch_size=64,
            discount=0.99,
            save_steps=101,
            replay_step=50,
            update_step=300,
            greedy_step=5000,
            max_steps=5000,
            memory_size=10000,
        )
        train = DQNTraining(config)
        train.train_model()

        # ##: DQN dueling.
        del train
        config = TrainingConfigurationDQN(
            store_history=STORAGE_MODEL,
            training_steps=1000,
            learning_rate=3e-4,
            batch_size=64,
            discount=0.99,
            save_steps=101,
            replay_step=50,
            update_step=300,
            greedy_step=5000,
            max_steps=5000,
            memory_size=10000,
        )
        train = DQNDuelingTraining(config)
        train.train_model()

    print("Finish!")
