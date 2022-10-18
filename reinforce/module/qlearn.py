# -*- coding: utf-8 -*-
"""
Q-Learning Agent.
"""
import math
import random
from collections import deque
from os.path import join

import numpy as np
import tensorflow as tf
from numpy import ndarray

from reinforce.addons import AgentConfigurationDQN, Experience, GCAdam
from reinforce.models import conv_learning, dense_learning

from .agent import TrainingAgent


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
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[indice] for indice in indices]


class AgentDQN(TrainingAgent):
    def __initialize_model(self, type_model: str):
        if type_model == "conv":
            func_model = conv_learning
        else:
            func_model = dense_learning

        # ## ----> Create networks
        self._policy = func_model(input_size=(4, 4, 1))
        self._target = func_model(input_size=(4, 4, 1))

    def __initialize_optimizer(self, learning_rate: float, batch_size: int):
        self._batch_size = batch_size
        self._optimizer = GCAdam(learning_rate=learning_rate)
        self._loss_function = tf.keras.losses.Huber()

    def __initialize_dqn_parameters(self, config: AgentConfigurationDQN):
        self._step_done = 0
        self._store_model = config.store_model
        self._discount = config.discount
        self._memory = ReplayMemory(config.memory_size)
        self._epsilon = {"min": config.epsilon_min, "max": config.epsilon_max, "decay": config.epsilon_decay}

    def _initialize_agent(self, config: AgentConfigurationDQN, observation_type: str, reward_type: str):
        # ## ----> Initialization network.
        self.__initialize_model(config.type_model)

        # ## ----> Initialization optimizer.
        self.__initialize_optimizer(config.learning_rate, config.batch_size)

        # ## ----> Initialization DQN parameters.
        self.__initialize_dqn_parameters(config)
        self._name = "_".join([config.type_model, observation_type, reward_type])

    def select_action(self, state: ndarray) -> int:
        # ## ----> Compute the threshold for choosing random action.
        thresh = self._epsilon["min"] + (self._epsilon["max"] - self._epsilon["min"]) * math.exp(
            -1.0 * (self._step_done / self._epsilon["decay"])
        )
        self._step_done += 1

        # ## ----> Choose a random action.
        if random.random() < thresh:
            return np.random.choice(4)

        # ## ----> Choose an optimal action.
        state_tensor = tf.convert_to_tensor(tf.reshape(state, (4, 4, 1)))
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_prob = self._policy(state_tensor, training=False)
        return tf.argmax(action_prob[0]).numpy()

    def save_model(self):
        self._policy.compile(optimizer=self._optimizer, loss_weights=self._loss_function)
        self._policy.save(join(self._store_model, f"dqn_model_{self._name}"))

    def remember(self, experience: Experience):
        self._memory.append(experience)

    def update_target(self):
        self._target.set_weights(self._policy.get_weights())

    def optimize_model(self):
        """
        Optimize the policy network.
        """
        # ## ----> Train if enough data.
        if len(self._memory) <= self._batch_size:
            return

        # ## ----> Get sample for training.
        sample = self._memory.sample(self._batch_size)

        # ## ----> Unpack training sample.
        batch = Experience(*zip(*sample))
        state_sample = np.array(list(batch.state))
        state_next_sample = np.array(list(batch.next_state))
        reward_sample = np.array(list(batch.reward))
        action_sample = np.array(list(batch.reward))

        # ## ----> Update Q-value.
        future_rewards = self._target.predict(state_next_sample)
        updated_q_values = reward_sample + self._discount * tf.reduce_max(future_rewards, axis=1)

        # ## ----> Calculate loss.
        masks = tf.one_hot(action_sample, 4)
        with tf.GradientTape() as tape:
            q_values = self._policy(state_sample)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self._loss_function(updated_q_values, q_action)

        # ## ----> Backpropagation.
        grads = tape.gradient(loss, self._policy.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._policy.trainable_variables))
