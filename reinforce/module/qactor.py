# -*- coding: utf-8 -*-
"""
Q-Learning Agent.
"""
import random

import tensorflow as tf
from numpy import ndarray
from numpy.random import choice

from reinforce.addons import INPUT_SIZE
from reinforce.models import dense_learning, dueling_dense_learning

from .core import Agent


class AgentDQN(Agent):
    """
    Train an agent to play 2048 Game with DQN algorithm.
    """

    _epsilon, _min_epsilon = 0.01, 0.01
    _name = "dqn"

    def _initialize_agent(self):
        """
        Initialize the policy for training.
        """

        # ##: Initialization network.
        tf.keras.backend.clear_session()
        self.policy = dense_learning(input_size=INPUT_SIZE)
        self.target = dense_learning(input_size=INPUT_SIZE)
        self.update_target()

    def update_target(self):
        """
        Update the weight of target model with the weight of policy model.
        """
        self.target.set_weights(self.policy.get_weights())

    def reduce_epsilon(self, decay: float):
        """
        Reduce the epsilon value.

        Parameters
        ----------
        decay: float
            Percentage discount

        """
        epsilon = self._epsilon * decay
        self._epsilon = max(self._min_epsilon, epsilon)

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
        if random.random() < self._epsilon:
            return choice(4)
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_prob = self.policy(state_tensor, training=False)
        return tf.argmax(action_prob[0]).numpy()


class AgentDQNDueling(AgentDQN):
    """
    Train an agent to play 2048 Game with DQN Dueling algorithm.
    """

    _name = "dqn-dueling"

    def _initialize_agent(self):
        """
        Initialize the policy for training.
        """

        # ##: Initialization network.
        tf.keras.backend.clear_session()
        self.policy = dueling_dense_learning(input_size=INPUT_SIZE)
        self.target = dueling_dense_learning(input_size=INPUT_SIZE)
        self.update_target()


class AgentDDQN(AgentDQN):
    """
    Train an agent to play 2048 Game with Double DQN algorithm.
    """

    _name = "double-dqn"


class AgentDDQNDueling(AgentDQNDueling):
    """
    Train an agent to play 2048 Game with Double Dueling DQN algorithm.
    """
    _name = "double-dqn-dueling"
