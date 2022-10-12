# -*- coding: utf-8 -*-
"""
Set of function for Q-Learning
"""
import collections
from typing import List

import numpy as np
import tensorflow as tf

from reinforce.addons import Experience


class ReplayMemory:
    """
    Memory buffer for Experience Replay.
    """

    def __init__(self, buffer_length: int):
        self.memory = collections.deque(maxlen=buffer_length)

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

    def sample(self, batch_size: int) -> List:
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


def deep_hidden_layers(head: tf.keras.layers.Layer, units: int) -> tf.keras.layers.Layer:
    """
    Add dense layer.

    Parameters
    ----------
    head: Layer
        Previous layer
    units: int
        Number of neurons in this layer

    Returns
    -------
    Layer:
        New layer
    """
    block = tf.keras.layers.Dense(units=units, kernel_initializer="he_uniform", activation="relu")(head)
    block = tf.keras.layers.Dropout(rate=0.2)(block)
    block = tf.keras.layers.BatchNormalization()(block)
    return block


def deep_q_learning(input_size: list) -> tf.keras.Model:
    """
    Create Q-Network for Q-Learning.

    Parameters
    ----------
    input_size: int
        Input dimension

    Returns
    -------
    Model:
        New model
    """
    # ## ----> Create input layer.
    inputs = tf.keras.layers.Input(shape=input_size)

    # ## ----> All hidden layers.
    hidden = deep_hidden_layers(head=inputs, units=32)
    hidden = deep_hidden_layers(head=hidden, units=64)
    hidden = deep_hidden_layers(head=hidden, units=64)
    hidden = deep_hidden_layers(head=hidden, units=32)

    # ## ----> Create output layer.
    outputs = tf.keras.layers.Dense(units=4, activation="linear")(hidden)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
