# -*- coding: utf-8 -*-
"""
Set of class for environment.
"""
from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Base Environment class.
    """

    @abstractmethod
    def step(self, action: int):
        """
        Applies an action or a chance outcome to the environment.

        Parameters
        ----------
        action: int
            Action to apply
        """

    @abstractmethod
    def observation(self):
        """
        Returns the observation of the environment to feed to the network.
        """

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Returns true if the environment is in a terminal state.
        """

    @abstractmethod
    def reward(self):
        """
        Returns the reward of the environment.
        """
