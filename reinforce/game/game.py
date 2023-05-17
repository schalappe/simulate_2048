# -*- coding: utf-8 -*-
"""
Environment that implements the rule of 2048.
"""
from typing import List, Tuple

import gym
import numpy as np
import tensorflow as tf

from simulate_2048 import GameBoard


class Game2048:
    """The 2048 environment."""

    def __init__(self, encodage_size: int):
        super().__init__()
        self.game = GameBoard()  # gym.make("GameBoard")
        self.encodage_size = encodage_size

    def reset(self) -> np.ndarray:
        """
        Reinitialize the environment.

        Returns
        -------
        ndarray
            Initial state of environment
        """
        obs, _ = self.game.reset()
        return self.encode(obs).astype(np.float32)

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        Flatten the observation given by the environment than encode it.

        Parameters
        ----------
        state: ndarray
            Observation given by the environment

        Returns
        -------
        ndarray:
            Encode observation
        """
        obs = state.copy()
        obs = np.reshape(obs, -1)
        obs[obs == 0] = 1
        obs = np.log2(obs)
        obs = obs.astype(np.int64)
        return np.reshape(np.eye(self.encodage_size)[obs], -1)

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns state, reward and done flag given an action.

        Parameters
        ----------
        action: ndarray
            Action to apply to environment

        Returns
        -------
        Tuple[ndarray, ndarray, ndarray]
            Next step, reward, done flag
        """

        state, reward, done, _, _ = self.game.step(action)
        return self.encode(state).astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32)

    def step(self, action: tf.Tensor) -> List[tf.Tensor]:
        """
        Returns state, reward and done flag given an action.

        Parameters
        ----------
        action: Tensor
            Action to apply to environment

        Returns
        -------
        List[Tensor]
            Next step, reward, done flag
        """
        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.int32, tf.int32])
