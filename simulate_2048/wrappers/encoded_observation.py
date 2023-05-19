# -*- coding: utf-8 -*-
"""
Encode the game state in binary representation..
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
        obs = reshape(obs, -1)
        obs[obs == 0] = 1
        obs = log2(obs)
        obs = obs.astype(int64)
        return reshape(eye(self._block_size)[obs], -1)

    def observation(self, observation: ndarray) -> ndarray:
        """
        Encode in binary the observation.

        Parameters
        ----------
        observation: ndarray
            State of the game

        Returns
        -------
        ndarray
            State of the game
        """
        return self.__encode(observation)
