# -*- coding: utf-8 -*-
"""
Wrappers for modifying the observation giving by an environment.
"""
import gym
import numpy as np
from gym import spaces


class FlattenObservation(gym.ObservationWrapper):
    """
    Class for flatten observation.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=2, high=2**32, shape=(env.size * env.size,), dtype=np.int64)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Flatten the observation given by the environment.

        Parameters
        ----------
        observation: np.ndarray
            Observation given by the environment

        Returns
        -------
        np.ndarray:
            Flatten observation
        """
        return np.reshape(observation, -1)


class LogObservation(gym.ObservationWrapper):
    """
    Class for simplifying observation.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=1, high=32, shape=(env.size, env.size), dtype=np.int64)

    def observation(self, observation) -> np.ndarray:
        """
        Return log2 of the observation given by the environment.

        Parameters
        ----------
        observation: np.ndarray
            Observation given by the environment

        Returns
        -------
        np.ndarray:
            Log2 of the observation
        """
        obs = observation.copy()
        obs[obs == 0] = 1
        obs = np.log2(obs)
        return obs.astype(int)


class FlattenLogObservation(gym.ObservationWrapper):
    """
    Class for flatten and simplifying the observation.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=1, high=32, shape=(env.size * env.size,), dtype=np.int64)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Flatten the observation given by the environment.

        Parameters
        ----------
        observation: np.ndarray
            Observation given by the environment

        Returns
        -------
        np.ndarray:
            Flatten observation
        """
        obs = observation.copy()
        obs = np.reshape(obs, -1)
        obs[obs == 0] = 1
        obs = np.log2(obs)
        return obs.astype(int)
