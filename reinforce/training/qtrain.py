# -*- coding: utf-8 -*-
"""
Q-Learning Algorithm.
"""
import pickle
from collections import Counter, deque
from os.path import join

import gym
from numpy import max as max_array_values
from numpy import sum as sum_array_values
from numpy.random import choice

from reinforce.addons import Experience, TrainingConfigurationDQN
from reinforce.module import TrainingAgentDQN
from simulate_2048 import LogObservation


class ReplayMemory:
    """
    Memory buffer for Experience Replay.
    """

    def __init__(self, buffer_length: int):
        self.memory = deque(maxlen=buffer_length)

    def __len__(self) -> int:
        return len(self.memory)

    def append(self, experience: Experience):
        """
        Add experience to the buffer.

        Parameters
        ----------
        experience: Experience
            Experience to add to the buffer
        """
        self.memory.append(experience)

    def sample(self, batch_size: int) -> list:
        """
        Sample a batch of experiences from the buffer.

        Parameters
        ----------
        batch_size: int
            Number of experiences to randomly select

        Returns
        -------
        list
            List of selected experiences
        """
        # ## ----> Choose randomly indice.
        indices = choice(len(self.memory), batch_size, replace=False)
        return [self.memory[indice] for indice in indices]


class DQNTraining:
    """
    The Deep Q Learning algorithm
    """

    def __init__(
        self,
        config: TrainingConfigurationDQN,
    ):
        # ## ----> Create game.
        self.__initialize_game(config.observation_type)

        # ## ----> Create agent.
        self._agent = TrainingAgentDQN(config.agent_configuration, config.observation_type)

        # ## ----> Directory for history.
        self._store_history = config.store_history
        self._name = "_".join([config.agent_configuration.type_model, config.observation_type])

        # ## ----> Parameters for training
        self._epoch = config.epoch
        self._batch_size = config.batch_size
        self._update = config.update_target
        self._memory = ReplayMemory(config.memory_size)

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
            done = False

            # ## ----> Play one party
            while not done:
                # ## ----> Perform action.
                action = self._agent.select_action(board)
                next_board, _, done, _, _ = self.game.step(action)
                board = next_board

                # ## ----> store max element.
                print(f"Game: {i + 1} - Score: {sum_array_values(board)}", end="\r")
                if done:
                    score.append(max_array_values(self.game.board))

            print(f"Game: {i + 1} is finished, score egal {score[-1]}.")

        # ## ----> Print score.
        print("Evaluation is finished.")
        frequency = Counter(score)
        print(f"Result: {dict(frequency)}")

    def replay(self):
        """
        Use experience to train policy network.
        """
        sample = self._memory.sample(self._batch_size)
        self._agent.optimize_model(sample)

    def train_model(self):
        """
        Train the policy network.
        """
        max_cell, history, = (
            0,
            [],
        )
        for step in range(self._epoch):
            total_reward = 0
            # ## ----> Initialize environment and state.
            board, _ = self.game.reset()

            done = False
            while not done:
                # ## ----> Select and perform action.
                action = self._agent.select_action(board)
                next_board, reward, done, _, _ = self.game.step(action)

                # ## ----> Store in memory.
                self._memory.append(Experience(board, next_board, action, reward, done))
                board = next_board
                total_reward += reward

                # ## ----> Perform one step of the optimization on the policy network.
                if len(self._memory) >= self._batch_size:
                    self.replay()

                # ## ----> Save game history
                if done:
                    max_cell = max_array_values(self.game.board)
                    history.append(
                        [sum_array_values(self.game.board), max_array_values(self.game.board), total_reward]
                    )

            # ## ----> Update the target network.
            if step % self._update == 0:
                self._agent.update_target()
            print(f"Max cell: {max_cell}, Total reward: {total_reward} at episode {step+1}")

        # ## ----> End of training.
        self.save_history(history)
        self._agent.save_model()

        # ## ----> Evaluate model.
        self.evaluate()
        self.game.close()
