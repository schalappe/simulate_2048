# -*- coding: utf-8 -*-
"""
Encode the game state in binary representation.
"""
from typing import Tuple

from numpy import eye, int64, log2, ndarray, reshape

from simulate.envs import GameBoard


class EncodedGameBoard(GameBoard):
    """Encode the game state in binary representation."""

    def __init__(self, size: int = 4, block_size: int = 31):
        """
        Initialize the encoded game board.

        Parameters
        ----------
        size : int, optional
            The size of the game board (default is 4).
        block_size : int, optional
            The size of the one-hot encoding block (default is 31).
        """
        self._block_size = block_size
        super().__init__(size)

    def reset(self, **kwargs) -> ndarray:
        """
        Reset the environment and return an encoded observation.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments for the reset method of the parent class.

        Returns
        -------
        ndarray
            The encoded initial state of the game board
        """
        obs = super().reset(**kwargs)
        return self.observation(obs)

    def step(self, action: int) -> Tuple[ndarray, float, bool]:
        """
        Apply the action, step the environment, and return an encoded observation.

        Parameters
        ----------
        action : int
            The action to apply to the game board.

        Returns
        -------
        Tuple[ndarray, float, bool]
            The encoded state of the game board, the reward, whether the game has ended.
        """
        observation, reward, terminated = super().step(action)
        return self.observation(observation), reward, terminated

    def observation(self, observation: ndarray) -> ndarray:
        """
        Encode the game board state into a binary representation.

        This method flattens the game board state provided by the environment,
        replaces zeros with ones to avoid log2 issues, computes the log base 2
        of each cell, converts the result to integers, and then one-hot encodes
        the values.

        Parameters
        ----------
        observation : ndarray
            The current state of the game board.

        Returns
        -------
        ndarray
            The binary-encoded state of the game board.

        Notes
        -----
        - The encoding process involves flattening the game board state,
          replacing zeros to avoid log2 issues, and one-hot encoding the
          logarithmic values.
        - Zero values in the state are replaced with ones to avoid issues with
          logarithm calculations.
        - The log base 2 of each cell value is taken to determine the one-hot
          encoding.
        - The resulting one-hot encoded array is flattened before being returned.
        """
        obs = observation.flatten()
        obs[obs == 0] = 1
        obs = log2(obs).astype(int64)
        encoded = eye(self._block_size, dtype=int64)[obs]
        return reshape(encoded, -1)
