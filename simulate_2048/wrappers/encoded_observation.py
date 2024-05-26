# -*- coding: utf-8 -*-
"""
Encode the game state in binary representation.
"""
import gym
from numpy import eye, int64, log2, ndarray, reshape


class EncodedObservation(gym.ObservationWrapper):
    """Encode the game state in binary representation."""

    def __init__(self, env, block_size: int = 31):
        super().__init__(env)
        self._block_size = block_size

    def __encode(self, state: ndarray) -> ndarray:
        """
        Flatten and encode the game board state.

        This method flattens the game board state provided by the environment,
        replaces zeros with ones to avoid log2 issues, computes the log base 2
        of each cell, converts the result to integers, and then one-hot encodes
        the values.

        Parameters
        ----------
        state : ndarray
            The game board state provided by the environment.

        Returns
        -------
        ndarray
            The encoded observation as a flattened one-hot encoded array.

        Notes
        -----
        - Zero values in the state are replaced with ones to avoid issues with
          logarithm calculations.
        - The log base 2 of each cell value is taken to determine the one-hot
          encoding.
        - The resulting one-hot encoded array is flattened before being returned.
        """
        obs = state.copy()
        obs = reshape(obs, -1)
        obs[obs == 0] = 1
        obs = log2(obs).astype(int64)
        encoded = eye(self._block_size, dtype=int64)[obs]
        return reshape(encoded, -1)

    def observation(self, observation: ndarray) -> ndarray:
        """
        Encode the game board state into a binary representation.

        This method takes the current state of the game board and encodes it
        using the internal `__encode` method to produce a binary-encoded
        representation of the game state.

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
        - This function acts as a wrapper around the private `__encode` method.
        """
        return self.__encode(observation)
