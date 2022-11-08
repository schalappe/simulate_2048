# -*- coding: utf-8 -*-
"""
Set of class for network use by Alpha Zero.
"""
from math import exp
from os.path import join
from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf
from numpy import ndarray

from alphazero.addons.types import NetworkOutput

from .core import PolicyNetwork


class Network:
    """
    An instance of the network used by AlphaZero.
    """

    def __init__(self, size: int):
        shape = (4 * 4 * size,)
        self.encodage_size = size
        self.model = PolicyNetwork()(shape)

    def encode(self, state: ndarray) -> ndarray:
        """
        Flatten the observation given by the environment than encode it.

        Parameters
        ----------
        state: ndarray
            Observation given by the environment

        Returns
        -------
        ndarray:
            Encode observation
        """
        obs = state.copy()
        obs = np.reshape(obs, -1)
        obs[obs == 0] = 1
        obs = np.log2(obs)
        obs = obs.astype(int)
        return np.reshape(np.eye(self.encodage_size)[obs], -1)

    def predictions(self, state: ndarray) -> NetworkOutput:
        """
        Returns the network predictions for a state.

        Parameters
        ----------
        state: ndarray
            The current state of the game

        Returns
        -------
        NetworkOutput
            The value of given state and the probabilities distribution over all moves
        """
        # ##: Transform the state for network.
        observation = self.encode(state)

        # ##: Use model for values and action probability distribution.
        obs_tensor = tf.convert_to_tensor(observation)
        obs_tensor = tf.expand_dims(obs_tensor, 0)
        probs, value = self.model(obs_tensor)

        # ##: Generate output.
        policy = [exp(probs[0][a]) for a in range(4)]
        return NetworkOutput(float(value[0]), {action: policy[action] / sum(policy) for action in range(4)})

    def train_step(self, batch: Sequence, optimizer: tf.keras.optimizers.Optimizer):
        """
        Train a single step of training.

        Parameters
        ----------
        batch: Sequence
            A list of observations, values and policies
        optimizer: Optimizer
            An optimizer function for back-propagation

        """
        # ##: Unpack batch into observations, values and policies.
        observations, target_values, target_policies = zip(*batch)

        # ##: Encode observation, then turn observations, values and policies into tensor.
        observations = tf.stack([self.encode(obs) for obs in observations])
        target_values = tf.stack(target_values)
        target_policies = tf.stack(target_policies)

        with tf.GradientTape() as tape:
            # ##: Model predictions.
            policies, values = self.model(observations, training=True)

            # ##: Compute loss.
            values_loss = tf.losses.huber(target_values, values)
            policies_loss = tf.losses.kl_divergence(target_policies, policies)
            loss = values_loss + policies_loss

        # ##: Optimize model.
        grads_model = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads_model, self.model.trainable_variables))

    def save_network(self, store_path: str):
        """
        Save a model.

        Parameters
        ----------
        store_path: str
            Path where store model
        """
        model_path = join(store_path, "alphazero.h5")
        self.model.save(model_path)


class NetworkCacher:
    """
    An object to share the network between the self-play and training jobs.
    """

    def __init__(self):
        self._networks = {}

    def save_network(self, step: int, network: Network):
        """
        Save a network in the cacher.

        Parameters
        ----------
        step: int
            The training step
        network: Network
            The network to store
        """
        self._networks[step] = network

    def load_network(self) -> Tuple[int, Network]:
        """
        Return the latest stored network.

        Returns
        -------
        tuple
            The latest training step and his network.
        """
        training_step = max(self._networks.keys())
        return training_step, self._networks[training_step]
