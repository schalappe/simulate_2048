# -*- coding: utf-8 -*-
"""
Proximal Policy Optimization Algorithm.
"""
import pickle
from collections import Counter
from itertools import count
from os.path import join

import gym
from numpy import max as max_array_values
from numpy import sum as sum_array_values

from reinforce.addons import Experience, TrainingConfigurationPPO
from reinforce.module import AgentPPO
from simulate_2048 import LogObservation


class PPOTraining:
    def __init__(self, config: TrainingConfigurationPPO):
        # ## ----> Create game.
        self.__initialize_game(config.observation_type, config.reward_type)

        # ## ----> Create agent.
        self._agent = AgentPPO(config.agent_configuration, config.observation_type, config.reward_type)

        # ## ----> Directory for history.
        self._store_history = config.store_history
        self._name = "_".join([config.agent_configuration.type_model, config.observation_type, config.reward_type])

        # ## ----> Parameters for training
        self._epoch = config.epoch

    def __initialize_game(self, observation_type, reward_type):
        if observation_type == "log":
            self.game = LogObservation(gym.make("GameBoard", size=4, type_reward=reward_type))
        else:
            self.game = gym.make("GameBoard", size=4, type_reward=reward_type)

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
                logits, action = self._agent.select_action(board)

