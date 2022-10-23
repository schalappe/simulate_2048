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

from reinforce.addons import AgentConfigurationDQN, Experience, GCAdam
from reinforce.models import dense_learning, dueling_dense_learning

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
        # ## ----> Choose a random action.
        if random.random() < 0.05:
            return np.random.choice(4)
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
        if config.type_model == "dense":
            func_model = dense_learning
        elif config.type_model == "dueling":
            func_model = dueling_dense_learning
        else:
            raise ValueError("The model isn't implemented yet.")

        # ## ----> Create networks
        tf.keras.backend.clear_session()
        self._policy = func_model(input_size=(4, 4, 1))
        self._target = func_model(input_size=(4, 4, 1))
        self.update_target()

        # ## ----> Initialization optimizer.
        self._optimizer = GCAdam(learning_rate=config.learning_rate)
        self._loss_function = tf.keras.losses.Huber()
        self._policy.compile(optimizer=self._optimizer, loss=self._loss_function)

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
            return choice(4)

        # ## ----> Choose an optimal action.
        state_tensor = tf.convert_to_tensor(tf.reshape(state, (4, 4, 1)))
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_prob = self._policy(state_tensor, training=False)
        return tf.argmax(action_prob[0]).numpy()

    def save_model(self):
        """
        Save policy model.
        """
        self._policy.save(join(self._store_model, f"model_simple_dqn_{self._name}"))

    def update_target(self):
        """
        Update the weight of target model with the weight of policy model.
        """
        self._target.set_weights(self._policy.get_weights())

    @classmethod
    def unpack_sample(cls, sample: list) -> tuple:
        """
        Unpack sample.

        Parameters
        ----------
        sample: list
            Sample of experience

        Returns
        -------
        tuple
            Unpack sample
        """
        # ## ----> Unpack training sample.
        batch = Experience(*zip(*sample))
        states = np.array(list(batch.state))
        states_next = np.array(list(batch.next_state))
        rewards = np.array(list(batch.reward))
        actions = np.array(list(batch.action))
        dones = np.array(list(batch.done))
        return states, states_next, rewards, actions, dones

    def optimize_model(self, sample: list):
        """
        Optimize the policy network.
        """
        # ## ----> Unpack training sample.
        state_sample, state_next_sample, reward_sample, action_sample, done_sample = self.unpack_sample(sample)

        # ## ----> Update Q-value.
        targets = self._target.predict(state_sample, verbose=0)
        next_q_values = self._target.predict(state_next_sample, verbose=0).max(axis=1)
        targets[range(len(sample)), action_sample] = reward_sample + (1 - done_sample) * next_q_values * self._discount

        # ## ----> Optimize policy.
        self._policy.fit(state_sample, targets, epochs=1, verbose=0)


class TrainingAgentDDQN(TrainingAgentDQN):
    """
    Train an agent to play 2048 Game with DQN algorithm.
    """

    def save_model(self):
        """
        Save policy model.
        """
        self._policy.save(join(self._store_model, f"model_double_dqn_{self._name}"))

    def optimize_model(self, sample: list):
        """
        Optimize the policy network.
        """
        # ## ----> Unpack training sample.
        state_sample, state_next_sample, reward_sample, action_sample, done_sample = self.unpack_sample(sample)

        # ## ----> Update Q-value.
        targets = self._target.predict(state_sample, verbose=0)
        next_q_values = self._target.predict(state_next_sample, verbose=0)[
            range(len(sample)), np.argmax(self._policy.predict(state_next_sample, verbose=0), axis=1)
        ]
        targets[range(len(sample)), action_sample] = reward_sample + (1 - done_sample) * next_q_values * self._discount

        # ## ----> Optimize policy.
        self._policy.fit(state_sample, targets, verbose=0)
