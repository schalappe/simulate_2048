# -*- coding: utf-8 -*-
"""
Script for training an agent.
"""
from reinforce.game.config import ENCODAGE_SIZE, Configuration


class Trainer:
    def __init__(self, config: Configuration):
        self._replay = config.replay_factory()
        self._cacher = config.cacher_factory(ENCODAGE_SIZE)
        self._actor = config.actor_factory(self._replay, self._cacher)
        self._learner = config.learn_factory(self._replay, self._cacher)
        self._cycles = config.cycles

    def launch_train_cycle(self):
        # ##: Learning cycle
        for cycle in range(self._cycles):
            print("-" * 88)
            print("Training loop ", cycle + 1)

            self._actor.play()
            self._learner.learn()

        print("Finish ...")


if __name__ == "__main__":
    import argparse

    from reinforce.algo.a2c_model import actor_critic

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True)
    args = parser.parse_args()

    trainer = None
    if args.algo == "A2C":
        trainer = Trainer(actor_critic())

    if trainer:
        trainer.launch_train_cycle()
    else:
        print("Not implemented yet !!!")
