# -*- coding: utf-8 -*-
"""
Script for training an agent.
"""
from reinforce.game.config import ENCODAGE_SIZE, Algorithm


def train_agent(*, algorithm: Algorithm) -> None:
    # ##: Create necessary for agent.
    replay_buffer = algorithm.replay_factory()
    cacher_network = algorithm.cacher_factory(ENCODAGE_SIZE)

    # ##: Create agent for self-play and learning.
    actor = algorithm.actor_factory(replay_buffer, cacher_network)
    learner = algorithm.learn_factory(replay_buffer, cacher_network)

    # ##: Self-play and learning cycle.
    for step in range(algorithm.cycles):
        print("-" * 88)
        print("Training loop ", step + 1)

        actor.play()
        learner.learn()

    print("Finish ...")


if __name__ == "__main__":
    import argparse

    from reinforce.algo.a2c_model import actor_critic

    train_agent(algorithm=actor_critic())
