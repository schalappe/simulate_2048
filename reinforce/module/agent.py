# -*- coding: utf-8 -*-
"""

"""
from abc import ABC, abstractmethod
from os import listdir
from os.path import exists, isdir, isfile, sep

import tensorflow as tf
from numpy import ndarray
from numpy.random import choice
import random

from reinforce.addons import AgentConfiguration, GCAdam


def check_model(model_path: str):
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
    def __init__(self, model_path: str, epsilon):
        if check_model(model_path):
            self.policy = tf.keras.models.load_model(model_path, custom_objects={"GCAdam": GCAdam})
            self.epsilon = epsilon
        else:
            raise TypeError(f"The directory or file `{model_path}` isn't a keras model.")

    def select_action(self, state: ndarray) -> int:
        if random.random() < self.epsilon:
            return choice(4)
        state_tensor = tf.expand_dims(tf.convert_to_tensor(tf.reshape(state, (4, 4, 1))), 0)
        action_prob = self.policy(state_tensor, training=False)
        action = tf.argmax(action_prob[0]).numpy()
        return action


class TrainingAgent(ABC):
    def __init__(self, config: AgentConfiguration, observation_type: str, reward_type: str):
        self._initialize_agent(config, observation_type, reward_type)

    @abstractmethod
    def _initialize_agent(self, config: AgentConfiguration, observation_type: str, reward_type: str):
        pass

    @abstractmethod
    def select_action(self, state: ndarray) -> int:
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def optimize_model(self):
        pass
