# -*- coding: utf-8 -*-
"""
Actor-critic methods.
"""
from typing import Union

import numpy as np
import tensorflow as tf
from numpy import ndarray

from reinforce.models import actor_and_critic_model

from .agent import Agent


class AgentA2C(Agent):
    """
    Train an agent to play 2048 Game with A2C algorithm.
    """

    def initialize_agent(self) -> None:
        # ## ----> Initialization network.
        tf.keras.backend.clear_session()
        self.policy = actor_and_critic_model(input_size=(4, 4, 1))

        # ## ----> Initialization A2C parameters.
        self.name = "a2c"

    def predict(self, state: Union[np.ndarray, tf.Tensor]) -> tuple:
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
        action_prob, critic_value = self.policy(state_tensor)
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
        action_prob, _ = self.predict(state)
        return tf.random.categorical(action_prob, 1)[0, 0]
