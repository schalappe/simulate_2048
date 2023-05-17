# -*- coding: utf-8 -*-
"""
Define configuration for actor-critic algorithm
"""
import tensorflow as tf

from reinforce.actor.a2c_actor import A2CActor
from reinforce.game.config import ActorConfiguration, Configuration
from reinforce.learner.a2c_learner import A2CConfiguration, A2CLearner
from reinforce.network.actor_critic import ActorCritic
from reinforce.network.cacher import NetworkCacher
from reinforce.replay.experience_replay import ExperienceReplay


def actor_critic() -> Configuration:
    return Configuration(
        cycles=100,
        replay_factory=lambda: ExperienceReplay(capacity=5, batch_size=5),
        cacher_factory=lambda size: NetworkCacher(
            network=ActorCritic()(shape=4 * 4 * size),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        ),
        actor_factory=lambda replay, cacher: A2CActor(
            config=ActorConfiguration(episodes=5), replay=replay, cacher=cacher
        ),
        learn_factory=lambda replay, cacher: A2CLearner(
            config=A2CConfiguration(epochs=1, discount=0.999), replay=replay, cacher=cacher
        ),
    )
