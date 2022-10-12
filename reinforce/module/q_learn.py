# -*- coding: utf-8 -*-
"""
Q-Learning Algorithm.
"""
import pickle
from itertools import count
from os.path import join

import gym
import numpy as np
import tensorflow as tf

from reinforce.addons import ConfigurationDQN, Experience
from reinforce.models import ReplayMemory, deep_q_learning
from simulate_2048 import FlattenLogObservation


class DQNTraining:
    """
    The Deep Q Learning algorithm
    """

    def __init__(self, config: dict, models_path: str):
        # ## ----> Check configuration.
        self.game = None
        self.epsilon = None
        self.store_directory = models_path

        # ## ----> Initialize parameters.
        self.__load_configuration(config)

        # ## ----> Necessary for training.
        self.loss_function = tf.keras.losses.Huber()

    def __load_configuration(self, config: ConfigurationDQN):
        # ## ----> Load from dictionary.
        self.config = config
        self.epsilon = config.epsilon

        # ## ----> Create other variables.
        self.memory = ReplayMemory(self.config.memory_size)
        self.policy_net = deep_q_learning(input_size=self.config.env_size**2)
        self.target_net = deep_q_learning(input_size=self.config.env_size**2)
        self.game = FlattenLogObservation(
            gym.make("GameBoard", size=self.config.env_size, type_reward=self.config.reward_type)
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate, clipnorm=1.0)

    def reload_game(self):
        """
        Reload game 2048.
        """
        self.game = FlattenLogObservation(
            gym.make("GameBoard", size=self.config.env_size, type_reward=self.config.reward_type)
        )

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action according to the state.

        Parameters
        ----------
        state: np.ndarray
            Game board

        Returns
        -------
        int
            Selected action
        """

        # ## ----> Choose an action.
        if self.epsilon > tf.random.uniform([], 0, 1):
            action = np.random.choice(4)
        else:
            action_prob = self.policy_net(state, training=False)
            action = tf.argmax(action_prob[0]).numpy()

        return action

    def save_model(self):
        """
        Save the trained model.
        """
        # ## ----> Save model.
        self.policy_net.save(join(self.store_directory, "dqn.h5"))

    def save_history(self, data: list):
        """
        Save history of training.
        Parameters
        ----------
        data: list
            History of training
        """
        with open(join(self.store_directory, "dqn_history.pkl"), "wb") as file_h:
            pickle.dump(data, file_h)

    def optimize_model(self):
        """
        Optimize the policy network.
        """
        # ## ----> Train if enough data.
        if len(self.memory) <= self.config.batch_size:
            return

        # ## ----> Get sample for training.
        sample = self.memory.sample(self.config.batch_size)

        # ## ----> Unpack training sample.
        batch = Experience(*zip(*sample))
        state_sample = np.array(list(batch.state))
        state_next_sample = np.array(list(batch.new_state))
        reward_sample = np.array(list(batch.reward))
        action_sample = np.array(list(batch.reward))

        # ## ----> Update Q-value.
        future_rewards = self.target_net.predict(state_next_sample)
        updated_q_values = reward_sample + self.config.discount * tf.reduce_max(future_rewards, axis=1)

        # ## ----> Calculate loss.
        masks = tf.one_hot(action_sample, 4)
        with tf.GradientTape() as tape:
            q_values = self.policy_net(state_sample)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.loss_function(updated_q_values, q_action)

        # ## ----> Backpropagation.
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

    def train_model(self):
        """
        Train the policy network.
        """
        max_cells, history = [], []
        for step in range(self.config.max_steps):
            # ## ----> Initialize environment and state.
            board, _ = self.game.reset()

            for move in count():
                # ## ----> Select and perform action.
                action = self.select_action(board)
                next_board, reward, done, _, _ = self.game.step(action)

                # ## ----> Decrease epsilon value.
                self.epsilon = max(self.epsilon * self.config.decay_epsilon, self.config.min_epsilon)

                # ## ----> Store in memory.
                self.memory.append(Experience(board, action, reward, done, next_board))
                board = next_board

                # ## ----> Perform one step of the optimization on the policy network.
                self.optimize_model()
                if done:
                    max_cells.append(np.max(self.game.board))
                    history.append([np.max(self.game.board), move])
                    break

            # ## ----> Update the target network.
            if step % self.config.target_update == 0:
                self.target_net.set_weights(self.policy_net.get_weights())
                print(f"Max cell: {max_cells[-1]} at episode {step}", end="\r")

        # ## ----> End of training.
        print(f"Training finish. Max value: {max(max_cells)}")
        self.game.close()
        self.save_history(history)
        self.save_model()
