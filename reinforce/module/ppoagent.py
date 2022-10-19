# -*- coding: utf-8 -*-
"""
Proximal Policy Agent.
"""
from numpy import ndarray
import tensorflow as tf

from reinforce.addons import AgentConfigurationPPO
from reinforce.models import dense_policy

from .agent import TrainingAgent


class TrainingAgentPPO(TrainingAgent):
    """
    Train an agent to play 2048 Game with Proximal policy algorithm.
    """
    def __initialize_optimizer(self, actor_learning_rate: float, critic_learning_rate: float):
        self._optimizer_actor = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
        self._optimizer_critic = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)

    def _initialize_agent(self, config: AgentConfigurationPPO, observation_type: str, reward_type: str):
        """
        Initialize agent.

        Parameters
        ----------
        config: AgentConfiguration
            Configuration for agent
        observation_type: str
            Type of observation give by the environment
        reward_type: str
            Type of reward give by the environment.
        """
        # ## ----> Initialize models.
        self._actor, self._critic = dense_policy(input_size=(4, 4, 1))

        # ## ----> Initialization optimizer.
        self.__initialize_optimizer(config.learning_rate, config.second_learning_rate)

        self._name = "_".join([observation_type, reward_type])
        self._clip_ratio = config.clip_ratio

    def select_action(self, state: ndarray) -> tuple:
        """
        Select an action given the state.

        Parameters
        ----------
        state: ndarray
            State of the game

        Returns
        -------
        tuple
            Probability for each action and selected action
        """
        probs = self._actor(state)
        action = tf.squeeze(tf.random.categorical(logits=probs, num_samples=1), axis=1)
        return probs, action

    def save_model(self):
        pass

    def train_policy(self, observations: ndarray, actions: ndarray, probabilities: ndarray, advantages: ndarray):
        actions = tf.one_hot(actions, 4)
        actions = tf.reshape(actions, [-1, 4])
        actions = tf.cast(actions, tf.float64)

        with tf.GradientTape() as tape:
            new_probabilities = tf.reduce_sum(tf.math.log(self._actor(observations)) * actions, axis=1)
            ratio = tf.exp(new_probabilities - probabilities)
            min_advantages = tf.where(
                advantages > 0,
                (1 + self._clip_ratio) * advantages,
                (1 - self._clip_ratio) * advantages
            )
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_advantages))

        policy_grads = tape.gradient(policy_loss, self._actor.trainable_variables)
        self._optimizer_actor.apply_gradients(zip(policy_grads, self._actor.trainable_variables))

        kl = tf.reduce_mean(probabilities - tf.reduce_sum(tf.math.log(self._actor(observations)) * actions, axis=1))
        kl = tf.reduce_sum(kl)
        return kl

    def train_value(self, observations, targets):
        with tf.GradientTape() as tape:
            value_loss = tf.reduce_mean((targets - self._critic(observations)) ** 2)
        value_grads = tape.gradient(value_loss, self._critic.trainable_variables)
        self._optimizer_critic.apply_gradients(zip(value_grads, self._critic.trainable_variables))
