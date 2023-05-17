# -*- coding: utf-8 -*-
"""
A2C actor to interact with an environment..
"""
import os
from collections import deque
from statistics import mean
from typing import Tuple

import tensorflow as tf
from tqdm import trange

from reinforce.game.config import ActorConfiguration
from reinforce.network.cacher import NetworkCacher
from reinforce.replay.replay import BufferReplay

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class A2CActor:
    """A2C actor to interact with the environment."""

    def __init__(self, config: ActorConfiguration, replay: BufferReplay, cacher: NetworkCacher):
        # ##: Actor configuration.
        self._episodes = config.episodes
        self._game = config.environment_factory()

        # ##: Shared data.
        self._cacher = cacher
        self._replay = replay

    def play(self):
        """
        Takes network, produces episodes and stores ten into replay buffer.
        """
        # ##: Keep the last episodes reward.
        episodes_reward = deque(maxlen=self._episodes)

        time_steps = trange(self._episodes)
        for _ in time_steps:
            # ##: Initialize game.
            initial_state = tf.constant(self._game.reset(), dtype=tf.float32)

            # ##: Run episode.
            action_probs, values, rewards = self.run_episode(initial_state)

            # ##: Store it.
            self._replay.store([action_probs, values, rewards])

            # ##: Statistics.
            episode_reward = int(tf.math.reduce_sum(rewards))
            episodes_reward.append(episode_reward)
            running_reward = mean(episodes_reward)

            # ##: Show statistics.
            time_steps.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

    @tf.function(reduce_retracing=True)
    def run_episode(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Runs a single episode to collect training data.

        Parameters
        ----------
        initial_state: tf.Tensor
            First state of the environment

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            all policies, values and rewards of the episode
        """

        # ##: Variable.
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        index = 0

        while tf.constant(True):
            states = states.write(index, state)

            # ##: Run the model and to get action probabilities and critic value.
            state = tf.expand_dims(state, 0)
            action_prob, _ = self._cacher.network(state)

            # Sample next action from the action probability distribution
            action = tf.random.categorical(action_prob, 1)[0, 0]
            actions = actions.write(index, tf.cast(action, tf.int32))

            # Apply action to the environment to get next state and reward
            state, reward, done = self._game.step(action)
            state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(index, reward)

            index += 1
            if tf.cast(done, tf.bool):
                break

        states = states.stack()
        actions = actions.stack()
        rewards = rewards.stack()

        return states, actions, rewards
