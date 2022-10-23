# -*- coding: utf-8 -*-
"""
A2C Training.
"""
import pickle
from collections import Counter
from itertools import count
from os.path import join

import gym
import numpy as np
import tensorflow as tf
from numpy import max as max_array_values
from numpy import sum as sum_array_values
from numpy.random import choice

from reinforce.addons import TrainingConfigurationA2C
from reinforce.module import TrainingAgentA2C
from simulate_2048 import LogObservation


class BufferMemory:
    """
    Memory buffer for training.
    """

    def __init__(self):
        self._actions, self._values, self._rewards = [], [], []

    def append(self, action, value, reward):
        """
        Add experience to the buffer.

        Parameters
        ----------
        experience: Experience
            Experience to add to the buffer
        """
        self._actions.append(action)
        self._values.append(value)
        self._rewards.append(reward)

    def get(self) -> tuple:
        """
        Return all experiences from the buffer.

        Returns
        -------
        list
            List of experiences
        """
        return self._actions, self._values, self._rewards

    def clear(self):
        """
        Delete experience saved.
        """
        self._actions.clear()
        self._values.clear()
        self._rewards.clear()


class A2CTraining:
    """
    The Actor-Critic algorithm
    """

    def __init__(self, config: TrainingConfigurationA2C):
        # ## ----> Create game.
        self.__initialize_game(config.observation_type)

        # ## ----> Create agent.
        self._agent = TrainingAgentA2C(config.agent_configuration, config.observation_type)

        # ## ----> Directory for history.
        self._store_history = config.store_history
        self._name = "_".join(["a2c", config.observation_type])

        # ## ----> Parameters for training
        self._epoch = config.epoch
        self._buffer = BufferMemory()

    def __initialize_game(self, observation_type):
        if observation_type == "log":
            self.game = LogObservation(gym.make("GameBoard", size=4))
        else:
            self.game = gym.make("GameBoard", size=4)

    def save_history(self, data: list):
        """
        Save history of training.
        Parameters
        ----------
        data: list
            History of training
        """
        with open(join(self._store_history, f"history_{self._name}.pkl"), "wb") as file_h:
            pickle.dump(data, file_h)

    def replay(self):
        """
        Use experience to train policy network.
        """
        sample = self._buffer.get()
        self._agent.optimize_model(sample)

    def evaluate(self):
        """
        Evaluate the policy network.
        """
        score = []
        for i in range(10):
            # ## ----> Reset game.
            board, _ = self.game.reset()

            # ## ----> Play one party
            for timestep in count():
                # ## ----> Perform action.
                action = self._agent.select_action(board)
                next_board, _, done, _, _ = self.game.step(action)
                board = next_board

                # ## ----> store max element.
                print(f"Game: {i + 1} - Score: {sum_array_values(board)}", end="\r")
                if done or timestep > 5000:
                    score.append(max_array_values(self.game.board))
                    break

            print(f"Game: {i + 1} is finished, score egal {score[-1]}.")

        # ## ----> Print score.
        print("Evaluation is finished.")
        frequency = Counter(score)
        print(f"Result: {dict(frequency)}")

    def train_model(self):
        """
        Train the policy network.
        """
        max_cell, history = 0, []
        for step in range(self._epoch):
            print(f"Start game {step + 1}")
            done, total_reward = False, 0

            # ## ----> Initialize environment and state.
            board, _ = self.game.reset()

            for timestep in count():
                # ## ----> Compute action probability.
                action_prob, critic_value = self._agent.predict(board)

                # ## ----> Perform action.
                action = choice(4, p=np.squeeze(action_prob))
                next_board, reward, done, _, _ = self.game.step(action)

                # ## ----> Store in memory.
                self._buffer.append(
                    action=tf.math.log(action_prob[0, action]), value=critic_value[0, 0], reward=reward
                )
                print(f"Game: {timestep + 1} - Reward: {total_reward}", end="\r")
                board = next_board
                total_reward += reward

                # ## ----> Stop game and save history
                if done or timestep > 5000:
                    max_cell = max_array_values(self.game.board)
                    history.append(
                        [sum_array_values(self.game.board), max_array_values(self.game.board), total_reward]
                    )
                    break

            # ## ----> Perform one step of the optimization on the policy network.
            print(f"Max cell: {max_cell}, Total reward: {total_reward:.2f} at episode {step+1}")
            self.replay()

            # ## ----> Clean buffer.
            self._buffer.clear()

        # ## ----> End of training.
        self.save_history(history)
        self._agent.save_model()

        # ## ----> Evaluate model.
        self.evaluate()
        self.game.close()
