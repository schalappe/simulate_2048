# -*- coding: utf-8 -*-
"""
Environment that implements the rule of 2048.
"""
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from simulate_2048 import GameBoard
from simulate_2048.wrappers import EncodedObservation


class Game2048:
    """The 2048 environment."""

    def __init__(self, encodage_size: int):
        super().__init__()
        self.game = EncodedObservation(GameBoard(), block_size=encodage_size)  # gym.make("GameBoard")

    def reset(self) -> np.ndarray:
        """
        Reinitialize the environment.

        Returns
        -------
        ndarray
            Initial state of environment
        """
        obs, _ = self.game.reset()
        return obs.astype(np.float32)

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
        return state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32)

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
