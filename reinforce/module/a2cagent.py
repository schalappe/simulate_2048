# -*- coding: utf-8 -*-
"""
Actor-critic methods.
"""
from os.path import join

import numpy as np
import tensorflow as tf
from numpy import ndarray

from reinforce.addons import AgentConfiguration, GCAdam
from reinforce.models import actor_and_critic_model

from .agent import TrainingAgent


class TrainingAgentA2C(TrainingAgent):
    """
    Train an agent to play 2048 Game with A2C algorithm.
    """

    def _initialize_agent(self, config: AgentConfiguration, observation_type: str):
        # ## ----> Initialization network.
        tf.keras.backend.clear_session()
        self._policy = actor_and_critic_model(input_size=(4, 4, 1))

        # ## ----> Initialization optimizer.
        self._optimizer = GCAdam(learning_rate=config.learning_rate)
        self._loss_function = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        # ## ----> Initialization A2C parameters.
        self._store_model = config.store_model
        self._discount = config.discount
        self._name = "_".join(["a2c", observation_type])

    def predict(self, state: np.ndarray) -> tuple:
        """
        Predict action probability and value of a state

        Parameters
        ----------
        state: ndarray
            State of the game

        Returns
        -------
        tuple
            Action probability and value of a state
        """
        state_tensor = tf.convert_to_tensor(tf.reshape(state, (4, 4, 1)))
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_prob, critic_value = self._policy(state_tensor)
        return action_prob, critic_value

    def select_action(self, state: ndarray) -> int:
        """
        Select an action given the state.

        Parameters
        ----------
        state: ndarray
            State of the game

        Returns
        -------
        int
            Selected action
        """
        # ## ----> Choose an optimal action.
        state_tensor = tf.convert_to_tensor(tf.reshape(state, (4, 4, 1)))
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_prob, _ = self._policy(state_tensor, training=False)
        return tf.argmax(action_prob[0]).numpy()

    def save_model(self):
        """
        Save policy model.
        """
        self._policy.save(join(self._store_model, f"model_{self._name}"))

    def optimize_model(self, sample: tuple):
        """
        Optimize the policy network.
        """
        # ## ----> Unpack training sample.
        action_probs_history, critic_value_history, rewards_history = sample

        # ## ----> Calculate expected value from rewards.
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + self._discount * discounted_sum
            returns.insert(0, discounted_sum)

        # ## ----> Calculate loss.
        with tf.GradientTape() as tape:
            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss
                critic_losses.append(self._loss_function(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))
        # ## ----> Backpropagation.
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, self._policy.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._policy.trainable_variables))
