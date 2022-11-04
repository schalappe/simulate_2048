# -*- coding: utf-8 -*-
"""
Core of training an agent.
"""
import pickle
from abc import ABC, abstractmethod
from collections import Counter
from itertools import count
from os.path import join
from typing import Any, List, Tuple

import gym
import numpy as np
import tensorflow as tf
from numpy import max as max_array_values
from numpy import ndarray
from numpy import sum as sum_array_values
from reinforce.addons import TrainingConfiguration

from simulate_2048 import FlattenOneHotObservation


class Training(ABC):
    """
    Algorithm to train an agent.
    """

    _name: str

    def __init__(self, config: TrainingConfiguration):
        self._config = config

        # ##: Create game.
        self._game = FlattenOneHotObservation(gym.make("GameBoard", size=4))

        # ##: Create agent.
        self._agent = None
        self._initialize_agent()

        # ##: Initialization optimizer.
        self._optimizer = None
        self._loss_function = None
        self._initialize_train_parameters()

        # ##: Other parameters.
        self._initialize_specific_parameters()

    @property
    def name(self):
        """
        Name of algorithm.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        """
        Change the name of algorithm.

        Parameters
        ----------
        value: str
            New name of the algorithm
        """
        self._name = value

    @abstractmethod
    def _initialize_agent(self):
        """
        Initialize agent for training.
        """

    @abstractmethod
    def _initialize_train_parameters(self):
        """
        Initialize parameters for training.
        """

    @abstractmethod
    def _initialize_specific_parameters(self):
        """
        Initialize specific parameters for the chosen algorithm.
        """

    def save_history(self, data: list):
        """
        Save history of training.

        Parameters
        ----------
        data: list
            History of training
        """
        with open(join(self._config.store_history, f"history_{self._name}.pkl"), "wb") as file_h:
            pickle.dump(data, file_h)

    def evaluate(self):
        """
        Evaluate the policy network.
        """
        score = []
        for i in range(100):
            # ##: Reset game.
            board, _ = self._game.reset()

            # ##: Play one party
            for timestep in count():
                # ##: Perform action.
                action = self._agent.select_action(board)
                next_board, _, done, _, _ = self._game.step(action)
                board = next_board

                # ##: store max element.
                print(
                    f"Game: {i + 1} - Score: {sum_array_values(self._game.board)} - Max: {max_array_values(self._game.board)}",
                    end="\r",
                )
                if done or timestep > 5000:
                    score.append(max_array_values(self._game.board))
                    break

            print(f"Game: {i + 1} is finished, score egal {score[-1]}.")

        # ##: Print score.
        print("Evaluation is finished.")
        frequency = Counter(score)
        print(f"Result: {dict(frequency)}")

    def environment_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        """
        Returns state, reward and done flag given an action.

        Parameters
        ----------
        action: tf.Tensor
            Action choose by the agent.

        Returns
        -------
        list
            Return of environment
        """

        def _env_step(_action) -> Tuple[ndarray, ndarray, ndarray]:
            state, reward, done, _, _ = self._game.step(_action)
            return state.astype(np.int32), np.array(reward, np.float32), np.array(done, np.int32)

        return tf.numpy_function(_env_step, [action], [tf.int32, tf.float32, tf.int32])

    @abstractmethod
    def train_model(self):
        """
        Train the policy network.
        """

    def close(self, histories: Any):
        """
        Store the training history then close all.

        Parameters
        ----------
        histories: Any
            History to store
        """
        # ## ----> End of training.
        self.save_history(histories)
        self._agent.save_model(self._config.store_history)

        # ## ----> Evaluate model.
        self.evaluate()
        self._game.close()
