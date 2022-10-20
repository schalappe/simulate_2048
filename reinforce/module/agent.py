# -*- coding: utf-8 -*-
"""
Agent definition.
"""
from abc import ABC, abstractmethod
from os import listdir
from os.path import exists, isdir, isfile, sep

import tensorflow as tf
from numpy import ndarray

from reinforce.addons import AgentConfiguration


def check_model(model_path: str) -> bool:
    """
    Check if a Tensorflow model exist.

    Parameters
    ----------
    model_path: str
        Path of model

    Returns
    -------
    bool
        True if it's Tensorflow model
        False else
    """
    # ## ----> Check if path exist.
    if not exists(model_path):
        raise ValueError(f"The path `{model_path}` doesn't exist.")

    # ## ----> Check if it's keras model.
    is_keras_file = isfile(model_path) and model_path.split(sep)[-1] in ["h5", "keras"]
    is_keras_dir = isdir(model_path) and "saved_model.pb" in listdir(model_path)
    if is_keras_file or is_keras_dir:
        return True
    return False


class Agent:
    """
    Agent to play 2048 Game.
    """

    def __init__(self, model_path: str):
        if check_model(model_path):
            self.policy = tf.keras.models.load_model(model_path)
        else:
            raise TypeError(f"The directory or file `{model_path}` isn't a keras model.")

    @abstractmethod
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


class TrainingAgent(ABC):
    """
    Train an agent to play 2048 Game.
    """

    def __init__(self, config: AgentConfiguration, observation_type: str):
        self._initialize_agent(config, observation_type)

    @abstractmethod
    def _initialize_agent(self, config: AgentConfiguration, observation_type: str):
        """
        Initialize agent.

        Parameters
        ----------
        config: AgentConfiguration
            Configuration for agent
        observation_type: str
            Type of observation give by the environment
        """

    @abstractmethod
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

    @abstractmethod
    def save_model(self):
        """
        Save policy model.
        """
