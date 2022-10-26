# -*- coding: utf-8 -*-
"""
A2C Training.
"""
import statistics
from typing import Tuple

import numpy as np
import tensorflow as tf
import tqdm

from reinforce.addons import GCAdam, TrainingConfigurationA2C
from reinforce.module import AgentA2C

from .core import Training


class A2CTraining(Training):
    """
    The Actor-Critic algorithm
    """

    def _initialize_agent(self):
        self._agent = AgentA2C()

    def _initialize_train_parameters(self):
        self._optimizer = GCAdam(learning_rate=self._config.learning_rate)
        self._loss_function = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    def _initialize_specific_parameters(self):
        self._epsilon = np.finfo(np.float32).eps.item()

    def __init__(self, config: TrainingConfigurationA2C):
        super().__init__(config)
        self._name = "a2c"
        self._config = config

    def run_episode(self, initial_state: tf.Tensor, max_steps: int = 5000) -> Tuple:
        """Runs a single episode to collect training data."""

        # ##: Buffer of experience
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        board = initial_state

        for timestep in tf.range(max_steps):
            # ##: Get action probabilities and critic value from agent.
            action_prob, value = self._agent.predict(board)

            # ##: Sample next action from the action probability distribution.
            action = tf.random.categorical(action_prob, 1)[0, 0]
            action_prob_t = tf.nn.softmax(action_prob)

            # ##: Store critic values.
            values = values.write(timestep, tf.squeeze(value))

            # ##: Store log probability of the action chosen
            action_probs = action_probs.write(timestep, action_prob_t[0, action])

            # ##: Apply action to the environment to get next state and reward
            board, reward, done = self.environment_step(action)
            board.set_shape(initial_state_shape)

            # ##: Store reward
            rewards = rewards.write(timestep, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def get_expected_return(self, rewards: tf.Tensor, standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""

        # ##: Initialize returns.
        size = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=size)

        # ##: Accumulate reward sum.
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for indice in tf.range(size):
            reward = rewards[indice]
            discounted_sum = reward + self._config.discount * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(indice, discounted_sum)
        returns = returns.stack()[::-1]

        # ##: Standardize for training stability.
        if standardize:
            returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self._epsilon)

        return returns

    def compute_loss(self, action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        # ##: Compute advantage.
        advantage = returns - values

        # ##: Compute actor loss.
        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        # ##: Compute critic loss.
        critic_loss = self._loss_function(values, returns)

        return actor_loss + critic_loss

    def train_step(self, initial_state: tf.Tensor, max_steps: int = 5000) -> float:
        """
        Runs a model training step.

        Parameters
        ----------
        initial_state: Tensor
            Initial state
        max_steps: int
            Max step for a single episode

        Returns
        -------
        float
            Total of reward
        """

        with tf.GradientTape() as tape:
            # ##: Collect training data.
            action_probs, values, rewards = self.run_episode(initial_state, max_steps)

            # ##: Calculate expected returns.
            returns = self.get_expected_return(rewards)

            # ##: Convert training data to appropriate TF tensor shapes.
            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # ##: Compute loss values.
            loss = self.compute_loss(action_probs, values, returns)

        # ##: Compute the gradients from the loss.
        grads = tape.gradient(loss, self._agent.policy.trainable_variables)

        # ##: Apply the gradients to the model's parameters
        self._optimizer.apply_gradients(zip(grads, self._agent.policy.trainable_variables))

        return tf.math.reduce_sum(rewards).numpy()

    def train_model(self):
        """
        Train the policy network.
        """
        histories = []
        with tqdm.trange(self._config.training_steps) as period:
            for step in period:
                # ##: Initialize environment and state.
                board, _ = self._game.reset()
                initial_state = tf.constant(board, dtype=tf.int32)

                # ##: Train model.
                episode_reward = self.train_step(initial_state, max_steps=self._config.max_steps)

                # ##: Log.
                histories.append(episode_reward)
                running_reward = statistics.mean(histories)
                period.set_description(f"Episode {step + 1}")
                period.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                # ##: Save model
                if step % self._config.save_steps == 0:
                    self._agent.save_model(self._config.store_history)

        # ##: End of training.
        self.close(histories)
