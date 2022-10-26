# -*- coding: utf-8 -*-
"""
Q-Learning Algorithm.
"""
import statistics

import numpy as np
import tensorflow as tf
import tqdm

from reinforce.addons import Experience, GCAdam, ReplayMemory, TrainingConfigurationDQN
from reinforce.module import AgentDQN, AgentDQNDueling, AgentDDQN, AgentDDQNDueling

from .core import Training


class DQNTraining(Training):
    """
    The Deep Q Learning algorithm
    """

    def _initialize_agent(self):
        self._agent = AgentDQN()

    def _initialize_train_parameters(self):
        self._optimizer = GCAdam(learning_rate=self._config.learning_rate)
        self._loss_function = tf.keras.losses.Huber()

    def _initialize_specific_parameters(self):
        self._memory = ReplayMemory(self._config.memory_size)

    def __init__(self, config: TrainingConfigurationDQN):
        super().__init__(config)
        self._name = "dqn"
        self._config = config

    def unpack_sample(self) -> tuple:
        """
        Unpack sample.

        Returns
        -------
        tuple
            Unpack sample
        """
        # ##: Unpack training sample.
        batch = Experience(*zip(*self._memory.sample(self._config.batch_size)))
        states = tf.stack(list(batch.state), 0)
        states_next = tf.stack(list(batch.next_state), 0)
        rewards = tf.stack(list(batch.reward), 0)
        actions = np.array(list(batch.action))
        dones = tf.stack(list(float(t) for t in batch.done), 0)
        return states, states_next, rewards, actions, dones

    def get_expected_values(self, sample_reward: tf.Tensor, sample_next_states: tf.Tensor, sample_dones: tf.Tensor) -> tf.Tensor:
        """
        Compute expected Q-values.

        Parameters
        ----------
        sample_reward: Tensor
            Batch of rewards
        sample_next_states: Tensor
            Batch of next states
        sample_dones: Tensor
            Batch of dones

        Returns
        -------
        Tensor
            Expected Q-values
        """
        # ##: Update Q-value.
        targets = self._agent.target.predict(sample_next_states, verbose=0)
        next_q_values = sample_reward + (1 - sample_dones) * self._config.discount * tf.reduce_max(targets, axis=1)

        return next_q_values

    def train_step(self, sample_action: tf.Tensor, sample_state: tf.Tensor, expected_values: tf.Tensor):
        """
        Runs a model training step.

        Parameters
        ----------
        sample_action: Tensor
            Batch of actions
        sample_state: Tensor
            Batch of states
        expected_values: Tensor
            Expected Q-values
        """
        # ##: Create a mask to calculate loss on the updated Q-values.
        masks = tf.one_hot(sample_action, 4)

        with tf.GradientTape() as tape:
            # ##: Train the model on the states and updated Q-values
            q_values = self._agent.policy(sample_state)

            # ##: Apply the masks to the Q-values to get the Q-value for action taken.
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

            # Calculate loss between new Q-value and old Q-value
            loss = self._loss_function(expected_values, q_action)

        # ##: Backpropagation
        grads = tape.gradient(loss, self._agent.policy.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._agent.policy.trainable_variables))

    def train_model(self):
        """
        Train the policy network.
        """
        timestep, histories = 0, []
        with tqdm.trange(self._config.training_steps) as period:
            for step in period:
                episode_reward = 0

                # ##: Initialize environment and state.
                board, _ = self._game.reset()
                board = tf.constant(board, dtype=tf.int32)
                initial_state_shape = board.shape

                for _ in range(self._config.max_steps):
                    # ##: Select and perform action.
                    action = self._agent.select_action(board)
                    self._agent.reduce_epsilon(self._config.decay)
                    next_board, reward, done = self.environment_step(action)

                    # ##: Store in memory.
                    self._memory.append(Experience(board, next_board, action, reward, done))
                    board = next_board
                    board.set_shape(initial_state_shape)
                    episode_reward += reward.numpy()

                    # ##: Perform one step of the optimization on the policy network.
                    if len(self._memory) >= self._config.batch_size and timestep % self._config.replay_step:
                        # ##: Unpack training sample.
                        state_sample, state_next_sample, reward_sample, action_sample, done_sample = self.unpack_sample()

                        # ##: Compute next Q-value.
                        target_values = self.get_expected_values(reward_sample, state_next_sample, done_sample)

                        # ##: Train the policy network.
                        self.train_step(action_sample, state_sample, target_values)

                    # ##: Update the target network.
                    if timestep % self._config.update_step == 0:
                        self._agent.update_target()

                    # ##: Increase timestep
                    timestep += 1

                    # ##: Save game history
                    if tf.cast(done, tf.bool):
                        break

                # ##: Log.
                histories.append(episode_reward)
                running_reward = statistics.mean(histories)
                period.set_description(f"Episode {step + 1}")
                period.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                # ##: Save model
                if step % self._config.save_steps == 0:
                    self._agent.save_model(self._config.store_history)

        # ##: End of training.
        self.close(histories)


class DQNDuelingTraining(DQNTraining):
    """
    The Deep Q Learning Dueling algorithm
    """

    def _initialize_agent(self):
        self._agent = AgentDQNDueling()


class DDQNTraining(DQNTraining):
    """
    The Double Deep Q Learning algorithm.
    """

    def _initialize_agent(self):
        self._agent = AgentDDQN()

    def get_expected_values(self, sample_reward: tf.Tensor, sample_next_states: tf.Tensor, sample_dones: tf.Tensor) -> tf.Tensor:
        """
        Compute expected Q-values.

        Parameters
        ----------
        sample_reward: Tensor
            Batch of rewards
        sample_next_states: Tensor
            Batch of next states
        sample_dones: Tensor
            Batch of dones

        Returns
        -------
        Tensor
            Expected Q-values
        """
        # ##: Update Q-value.
        targets = self._agent.target.predict(sample_next_states, verbose=0)[range(self._config.batch_size), tf.argmax(self._agent.policy.predict(sample_next_states, verbose=0), axis=1)]
        next_q_values = sample_reward + (1 - sample_dones) * self._config.discount * targets

        return next_q_values


class DDQNDuelingTraining(DDQNTraining):
    """
    The Double Deep Q Learning Dueling algorithm.
    """

    def _initialize_agent(self):
        self._agent = AgentDDQNDueling()
