# -*- coding: utf-8 -*-
"""
A2C Training.
"""
import pickle
from collections import Counter
from itertools import count
from os.path import join

import gym
import statistics
import numpy as np
import tensorflow as tf
from numpy import max as max_array_values
from numpy import sum as sum_array_values
from numpy import ndarray
from typing import List, Tuple
from reinforce.addons import GCAdam
import tqdm

from reinforce.addons import TrainingConfigurationA2C
from reinforce.module import AgentA2C
from simulate_2048 import LogObservation


class A2CTraining:
    """
    The Actor-Critic algorithm
    """

    def __init__(self, config: TrainingConfigurationA2C):
        # ## ----> Create game.
        self.__initialize_game(config.observation_type)

        # ## ----> Create agent.
        self._agent = AgentA2C()

        # ## ----> Initialization optimizer.
        self._optimizer = GCAdam(learning_rate=config.learning_rate)
        self._loss_function = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        # ## ----> Directory for history.
        self._store_history = config.store_history
        self._name = "a2c"

        # ## ----> Parameters for training
        self._epoch = config.epoch
        self._discount = config.discount
        self._epsilon = np.finfo(np.float32).eps.item()

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
                print(f"Game: {i + 1} - Score: {sum_array_values(self.game.board)}", end="\r")
                if done or timestep > 5000:
                    score.append(max_array_values(self.game.board))
                    break

            print(f"Game: {i + 1} is finished, score egal {score[-1]}.")

        # ## ----> Print score.
        print("Evaluation is finished.")
        frequency = Counter(score)
        print(f"Result: {dict(frequency)}")

    def environment_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        """Returns state, reward and done flag given an action."""
        def _env_step(_action) -> Tuple[ndarray, ndarray, ndarray]:
            state, reward, done, _, _ = self.game.step(_action)
            return state.astype(np.int32), np.array(reward, np.float32), np.array(done, np.int32)
        return tf.numpy_function(_env_step, [action],  [tf.int32, tf.float32, tf.int32])

    def run_episode(self, initial_state: tf.Tensor, max_steps: int = 5000) -> Tuple:
        """Runs a single episode to collect training data."""

        # ## ----> Buffer of experience
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        board = initial_state

        for t in tf.range(max_steps):
            # ## ----> Get action probabilities and critic value from agent.
            action_prob, value = self._agent.predict(board)

            # ## ----> Sample next action from the action probability distribution.
            action = tf.random.categorical(action_prob, 1)[0, 0]
            action_prob_t = tf.nn.softmax(action_prob)

            # ## ----> Store critic values.
            values = values.write(t, tf.squeeze(value))

            # ## ----> Store log probability of the action chosen
            action_probs = action_probs.write(t, action_prob_t[0, action])

            # ## ----> Apply action to the environment to get next state and reward
            board, reward, done = self.environment_step(action)
            board.set_shape(initial_state_shape)

            # ## ----> Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def get_expected_return(self, rewards: tf.Tensor, standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""

        # ## ----> Initialize returns.
        size = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=size)

        # ## ----> Accumulate reward sum.
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for indice in tf.range(size):
            reward = rewards[indice]
            discounted_sum = reward + self._discount * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(indice, discounted_sum)
        returns = returns.stack()[::-1]

        # ## ----> Standardize for training stability.
        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self._epsilon))

        return returns

    def compute_loss(self, action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        # ## ----> Compute advantage.
        advantage = returns - values

        # ## ----> Compute actor loss.
        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        # ## ----> Compute critic loss.
        critic_loss = self._loss_function(values, returns)

        return actor_loss + critic_loss

    def train_step(self, initial_state: tf.Tensor, max_steps: int = 5000) -> Tuple:
        """Runs a model training step."""

        with tf.GradientTape() as tape:
            # ## ----> Collect training data.
            action_probs, values, rewards = self.run_episode(initial_state, max_steps)

            # ## ----> Calculate expected returns.
            returns = self.get_expected_return(rewards)

            # ## ----> Convert training data to appropriate TF tensor shapes.
            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # ## ----> Compute loss values.
            loss = self.compute_loss(action_probs, values, returns)

        # ## ----> Compute the gradients from the loss.
        grads = tape.gradient(loss, self._agent.policy.trainable_variables)

        # Apply the gradients to the model's parameters
        self._optimizer.apply_gradients(zip(grads, self._agent.policy.trainable_variables))

        return tf.math.reduce_sum(rewards)

    def train_model(self):
        """
        Train the policy network.
        """
        histories = []
        with tqdm.trange(self._epoch) as period:
            for step in period:
                # ## ----> Initialize environment and state.
                board, _ = self.game.reset()
                initial_state = tf.constant(board, dtype=tf.int32)

                # ## ----> Train model.
                episode_reward = int(self.train_step(initial_state, max_steps=1000))

                # ## ----> Log.
                histories.append(episode_reward)
                running_reward = statistics.mean(histories)
                period.set_description(f"Episode {step + 1}")
                period.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

        # ## ----> End of training.
        self.save_history(histories)
        self._agent.save_model(self._store_history)

        # ## ----> Evaluate model.
        self.evaluate()
        self.game.close()
