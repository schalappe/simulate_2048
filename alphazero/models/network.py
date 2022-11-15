# -*- coding: utf-8 -*-
"""
Set of class for network use by Alpha Zero.
"""
from collections import deque
from os.path import join
from typing import Sequence

import numpy as np
import tensorflow as tf
from numba import jit
from numpy import ndarray

from alphazero.addons.config import ENCODAGE_SIZE
from alphazero.addons.types import NetworkOutput

from .core import PolicyNetwork


@jit(nopython=True, cache=True)
def encode(state: ndarray, encodage_size: int) -> ndarray:
    """
    Flatten the observation given by the environment than encode it.

    Parameters
    ----------
    state: ndarray
        Observation given by the environment
    encodage_size: int
        Size of encodage

    Returns
    -------
    ndarray:
        Encode observation
    """
    obs = state.copy()
    obs = np.reshape(obs, -1)
    obs[obs == 0] = 1
    obs = np.log2(obs)
    obs = obs.astype(np.int64)
    return np.reshape(np.eye(encodage_size)[obs], -1)


class Network:
    """
    An instance of the network used by AlphaZero.
    """

    def __init__(self, size: int):
        shape = (4 * 4 * size,)
        self.encodage_size = size
        self.model = PolicyNetwork()(shape)

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
        observation = encode(state, self.encodage_size)

        # ##: Use model for values and action probability distribution.
        obs_tensor = tf.convert_to_tensor(observation)
        obs_tensor = tf.expand_dims(obs_tensor, 0)
        probs, value = self.model(obs_tensor)

        # ##: Generate output.
        # policy = [exp(probs[0][a]) for a in range(4)]
        return NetworkOutput(float(value[0]), {action: probs[0][action].numpy() for action in range(4)})

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
        observations = tf.stack([encode(obs, self.encodage_size) for obs in observations])
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
    An object to share the network weights between the self-play and training jobs.
    """

    def __init__(self, max_weights: int = 100):
        self._networks = deque(maxlen=max_weights)

    def save_network(self, network: Network):
        """
        Save the network weights.

        Parameters
        ----------
        network: Network
            Model to save
        """
        self._networks.append(network.model.get_weights())

    def load_network(self) -> Network:
        """
        Use the last weight to create a new model.

        Returns
        -------
        Network
            Model with the last weight
        """
        network = Network(ENCODAGE_SIZE)
        network.model.set_weights(self._networks[-1])
        return network
