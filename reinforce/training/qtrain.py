# -*- coding: utf-8 -*-
"""
Q-Learning Algorithm.
"""
import pickle
from collections import Counter
from itertools import count
from os.path import join

import gym
from numpy import max as max_array_values
from numpy import sum as sum_array_values

from reinforce.addons import Experience, TrainingConfigurationDQN
from reinforce.module import AgentDQN
from simulate_2048 import LogObservation


class DQNTraining:
    """
    The Deep Q Learning algorithm
    """

    def __init__(self, config: TrainingConfigurationDQN,):
        # ## ----> Create game.
        self.__initialize_game(config.observation_type, config.reward_type)

        # ## ----> Create agent.
        self._agent = AgentDQN(config.agent_configuration, config.observation_type, config.reward_type)

        # ## ----> Directory for history.
        self._store_history = config.store_history
        self._name = "_".join([config.agent_configuration.type_model, config.observation_type, config.reward_type])

        # ## ----> Parameters for training
        self._epoch = config.epoch
        self._update = config.update_target

    def __initialize_game(self, observation_type, reward_type):
        if observation_type == "log":
            self.game = LogObservation(gym.make("GameBoard", size=4, type_reward=reward_type))
        else:
            self.game = gym.make("GameBoard", size=4, type_reward=reward_type)

    def save_history(self, data: list):
        """
        Save history of training.
        Parameters
        ----------
        data: list
            History of training
        """
        with open(join(self._store_history, f"dqn_history_{self._name}.pkl"), "wb") as file_h:
            pickle.dump(data, file_h)

    def evaluate(self):
        """
        Evaluate the policy network.
        """
        score = []
        for i in range(10):
            # ## ----> Reset game.
            board, _ = self.game.reset()

            # ## ----> Play one party
            for move in count():
                # ## ----> Perform action.
                action = self._agent.select_action(board)
                next_board, _, done, _, _ = self.game.step(action)
                board = next_board

                # ## ----> store max element.
                print(f"Game: {i + 1} - Score: {max_array_values(board)} - move: {move}", end="\r")
                if done:
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
        max_cells, history = [], []
        for step in range(self._epoch):
            # ## ----> Initialize environment and state.
            board, _ = self.game.reset()

            for move in count():
                # ## ----> Select and perform action.
                action = self._agent.select_action(board)
                next_board, reward, done, _, _ = self.game.step(action)

                # ## ----> Store in memory.
                self._agent.remember(Experience(board, next_board, action, reward, done))
                board = next_board

                # ## ----> Perform one step of the optimization on the policy network.
                self._agent.optimize_model()
                if done:
                    max_cells.append(max_array_values(self.game.board))
                    history.append([sum_array_values(self.game.board), max_array_values(self.game.board), move])
                    break

            # ## ----> Update the target network.
            if step % self._update == 0:
                self._agent.update_target()
                print(f"Max cell: {max_cells[-1]} at episode {step}")

        # ## ----> End of training.
        print(f"Training finish. Max value: {max(max_cells)}")
        self.save_history(history)
        self._agent.save_model()

        # ## ----> Evaluate model.
        self.evaluate()
        self.game.close()
