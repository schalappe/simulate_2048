# -*- coding: utf-8 -*-
"""
Q-Learning Agent.
"""
import random
from os.path import join

import numpy as np
import tensorflow as tf
from numpy import ndarray
from numpy.random import choice

from reinforce.addons import AgentConfigurationDQN, Experience
from reinforce.models import conv_learning, dense_learning

from .agent import Agent, TrainingAgent


class AgentDQN(Agent):
    """
    Agent to play 2048 Game.
    """

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
        if random.random() < self.epsilon:
            return choice(4)
        state_tensor = tf.expand_dims(tf.convert_to_tensor(tf.reshape(state, (4, 4, 1))), 0)
        action_prob = self.policy(state_tensor, training=False)
        action = tf.argmax(action_prob[0]).numpy()
        return action


class TrainingAgentDQN(TrainingAgent):
    """
    Train an agent to play 2048 Game with DQN algorithm.
    """

    def _initialize_agent(self, config: AgentConfigurationDQN, observation_type: str):
        """
        Initialize agent.

        Parameters
        ----------
        config: AgentConfiguration
            Configuration for agent
        observation_type: str
            Type of observation give by the environment
        """
        # ## ----> Initialization network.
        if config.type_model == "conv":
            func_model = conv_learning
        else:
            func_model = dense_learning

        # ## ----> Create networks
        self._policy = func_model(input_size=(4, 4, 1))
        self._target = func_model(input_size=(4, 4, 1))
        self.update_target()

        # ## ----> Initialization optimizer.
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self._loss_function = tf.keras.losses.Huber()
        self._policy.compile(optimizer=self._optimizer, loss_weights=self._loss_function)

        # ## ----> Initialization DQN parameters.
        self._store_model = config.store_model
        self._discount = config.discount
        self._epsilon = {"min": config.epsilon_min, "value": config.epsilon_max, "decay": config.epsilon_decay}
        self._name = "_".join([config.type_model, observation_type])

    def reduce_epsilon(self):
        """
        Reduce the epsilon value.
        """
        epsilon = self._epsilon["value"] * self._epsilon["decay"]
        self._epsilon["value"] = max(self._epsilon["min"], epsilon)

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
        # ## ----> Reduce epsilon.
        self.reduce_epsilon()

        # ## ----> Choose a random action.
        if random.random() < self._epsilon["value"]:
            return np.random.choice(4)

        # ## ----> Choose an optimal action.
        state_tensor = tf.convert_to_tensor(tf.reshape(state, (4, 4, 1)))
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_prob = self._policy(state_tensor, training=False)
        return tf.argmax(action_prob[0]).numpy()

    def save_model(self):
        """
        Save policy model.
        """
        self._policy.save(join(self._store_model, f"dqn_model_{self._name}"))

    def update_target(self):
        """
        Update the weight of target model with the weight of policy model.
        """
        self._target.set_weights(self._policy.get_weights())

    # ##: TODO: Extraire le training et faire un tf.function.
    def optimize_model(self, sample: list):
        """
        Optimize the policy network.
        """
        # ## ----> Unpack training sample.
        batch = Experience(*zip(*sample))
        state_sample = np.array(list(batch.state))
        state_next_sample = np.array(list(batch.next_state))
        reward_sample = np.array(list(batch.reward))
        action_sample = np.array(list(batch.reward))

        # ## ----> Update Q-value.
        future_rewards = self._target.predict(state_next_sample)
        updated_q_values = reward_sample + self._discount * tf.reduce_max(future_rewards, axis=1)

        # ## ----> Calculate loss.
        masks = tf.one_hot(action_sample, 4)
        with tf.GradientTape() as tape:
            q_values = self._policy(state_sample)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self._loss_function(updated_q_values, q_action)

        # ## ----> Backpropagation.
        grads = tape.gradient(loss, self._policy.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._policy.trainable_variables))
