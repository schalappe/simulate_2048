# -*- coding: utf-8 -*-
"""
Actor-Critic learner to update the network weight.
"""
import os
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Sequence

import tensorflow as tf
from tqdm import trange

from numpy import finfo, float32
from reinforce.game.config import LearnerConfiguration
from reinforce.network.cacher import NetworkCacher
from reinforce.replay.replay import BufferReplay

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@dataclass
class A2CConfiguration(LearnerConfiguration):
    """Data needed to update network weight."""

    discount: float


class A2CLearner:
    """An actor-critic learner to update the network weights based."""

    def __init__(self, config: A2CConfiguration, replay: BufferReplay, cacher: NetworkCacher):
        # ##: Learner configuration.
        self._epochs = config.epochs
        self._discount = config.discount
        self._eps = finfo(float32).eps.item()

        # ##: Shared data.
        self._cacher = cacher
        self._replay = replay

    def _get_expected_return(self, rewards: tf.Tensor, standardize: bool = True) -> tf.Tensor:
        # ##: Initialize `returns`.
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # ##: Start from the end of `rewards` and accumulate reward sums into the `returns` array.
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self._discount * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self._eps))

        return returns

    @classmethod
    def _compute_loss(cls, action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined Actor-Critic loss."""
        huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = huber_loss(values, returns)

        return actor_loss + critic_loss

    @tf.function(reduce_retracing=True)
    def train_step(self, experiences: Sequence[Sequence[tf.Tensor]]) -> tf.Tensor:
        with tf.GradientTape() as tape:

            # ##: Unstack necessary for training.
            states, actions, rewards = experiences
            returns = self._get_expected_return(rewards)

            # ##: Get policies and values
            policies, values = self._cacher.network(states, training=True)
            actions = tf.one_hot(actions, 4)
            policies = tf.nn.softmax(policies)
            action_probs = tf.reduce_max(policies * actions, axis=1)

            # ##: Convert training data to appropriate TF tensor shapes
            action_probs, returns = [tf.expand_dims(x, 1) for x in [action_probs, returns]]

            # ##: Calculate the loss values to update network.
            loss = self._compute_loss(action_probs, values, returns)

        # ##: Compute the gradients from the loss
        grads = tape.gradient(loss, self._cacher.network.trainable_variables)

        # Apply the gradients to the model's parameters
        self._cacher.optimizer.apply_gradients(zip(grads, self._cacher.network.trainable_variables))

        return loss

    def learn(self):
        """Single training step of the learner."""

        # ##: Episodes from buffer.
        replay_episodes = self._replay.sample()

        epochs = len(replay_episodes)

        # ##: Keep the last epochs loss.
        epochs_loss = deque(maxlen=epochs)

        time_steps = trange(epochs)
        for step in time_steps:
            # ##: Run a single training step.
            loss = self.train_step(replay_episodes[step])
            loss = float(loss.numpy())

            # ##: Statistics.
            epochs_loss.append(loss)
            running_loss = mean(epochs_loss)

            # ##: Show statistics.
            time_steps.set_postfix(loss=loss, running_loss=running_loss)
