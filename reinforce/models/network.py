# -*- coding: utf-8 -*-
"""
Set of class for network use by Alpha Zero.
"""
from abc import ABCMeta, abstractmethod
from os.path import join
from typing import Sequence, Union

import numpy as np
import tensorflow as tf
from numba import njit, prange
from numpy import ndarray

from reinforce.addons.types import NetworkOutput

from .core import PolicyNetwork


@njit(fastmath=True)
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


@njit(parallel=True, fastmath=True)
def encode_multiple(states: ndarray, encodage_size: int) -> ndarray:
    """
    Encode observation given by environment.

    Parameters
    ----------
    states: ndarray
        Observation given by the environment
    encodage_size: int
        Size of encodage

    Returns
    -------
    ndarray:
        Encode observation
    """
    _len = states.shape[0]
    size = states.shape[1]
    observations = np.zeros((_len, encodage_size * size))

    for i in prange(_len):
        observations[i] = encode(states[i], encodage_size)

    return observations


def scale_gradient(tensor: tf.Tensor, scale: Union[float, ndarray]) -> tf.Tensor:
    """
    Scales the gradient for the backward pass.

    Parameters
    ----------
    tensor: Tensor
        A Tensor
    scale: float or ndarray
        The scale factor

    Returns
    -------
    Tensor
        A tensor with the same type as the input tensor
    """
    dtype = tensor.dtype.base_dtype
    scale = tf.convert_to_tensor(scale, dtype=dtype)
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


class Network(metaclass=ABCMeta):
    """
    An instance of the network used by AlphaZero.
    """

    def __init__(self, size: int, size_or_path: Union[str, tuple]):
        self.encodage_size = size
        self.model = self._load_model(size_or_path)

    @abstractmethod
    def _load_model(self, size_or_path: Union[str, tuple]) -> tf.keras.Model:
        """
        Load model.

        Returns
        -------
        PolicyNetwork
            Model to use
        """

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
        obs_tensor = tf.convert_to_tensor(observation, dtype=tf.float16)
        obs_tensor = tf.expand_dims(obs_tensor, 0)
        probs, value = self.model(obs_tensor)

        # ##: Generate output.
        return NetworkOutput(float(value[0]), {action: probs[0][action] for action in range(4)})


class TrainNetwork(Network):
    """
    An instance of the network used by AlphaZero.
    """

    def __init__(self, size: int):
        super().__init__(size, (4 * 4 * size,))

    def _load_model(self, size_or_path: tuple) -> PolicyNetwork:
        """
        Load model.

        Returns
        -------
        PolicyNetwork
            Model to use
        """
        return PolicyNetwork()(size_or_path)

    def train_step(self, batch: Sequence, optimizer: tf.keras.optimizers.Optimizer) -> float:
        """
        Train a single step of training.

        Parameters
        ----------
        batch: Sequence
            A list of observations, values and policies
        optimizer: Optimizer
            An optimizer function for back-propagation

        Returns
        -------
        float
            Loss
        """
        # ##: Unpack batch into observations, values and policies.
        observations, target_values, target_policies, weights = zip(*batch)

        # ##: Encode observation.
        observations = encode_multiple(np.array(list(observations)), self.encodage_size)

        # ##: Turn observations, values and policies into tensor.
        observations = tf.stack(observations)
        target_values = tf.stack(target_values)
        target_policies = tf.stack(target_policies)

        with tf.GradientTape() as tape:
            # ##: Model predictions.
            policies, values = self.model(observations, training=True)

            # ##: Compute loss.
            values_loss = tf.losses.huber(target_values, values)
            values_loss = scale_gradient(values_loss, weights)

            policies_loss = tf.losses.kullback_leibler_divergence(target_policies, policies)
            policies_loss = scale_gradient(policies_loss, weights)

            loss = tf.reduce_sum(values_loss) + tf.reduce_sum(policies_loss)

        # ##: Optimize model.
        grads_model = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads_model, self.model.trainable_variables))
        return loss.numpy()

    def save_network(self, store_path: str, index: int):
        """
        Save a model.

        Parameters
        ----------
        store_path: str
            Path where store model
        index: int
            Number of model
        """
        model_path = join(store_path, f"alphazero_{index}.h5")
        self.model.save(model_path)


class FinalNetwork(Network):
    """
    An instance of the network used by AlphaZero.
    """

    def __init__(self, size: int, path: str):
        super().__init__(size, path)

    def _load_model(self, size_or_path) -> tf.keras.Model:
        """
        Load model.

        Returns
        -------
        PolicyNetwork
            Model to use
        """
        return tf.keras.models.load_model(size_or_path)
