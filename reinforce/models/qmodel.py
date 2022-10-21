# -*- coding: utf-8 -*-
"""
Set of function for Q-Learning
"""
from typing import Union

import tensorflow as tf


def dense_hidden_layers(head: tf.keras.layers.Layer, units: int) -> tf.keras.layers.Layer:
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
    Layer
        New layer
    """
    block = tf.keras.layers.Dense(units=units, activation="relu")(head)
    # block = tf.keras.layers.Dense(units=units, kernel_initializer="he_uniform", activation="relu")(head)
    # block = tf.keras.layers.Dropout(rate=0.2)(block)
    return block


def dense_learning(input_size: Union[list or tuple]) -> tf.keras.Model:
    """
    Create Q-Network for Q-Learning.

    Parameters
    ----------
    input_size: int
        Input dimension

    Returns
    -------
    Model
        New model
    """
    # ## ----> Create input layer.
    inputs = tf.keras.layers.Input(shape=input_size)

    # ## ----> All hidden layers.
    hidden = tf.keras.layers.Flatten()(inputs)
    for _ in range(4):
        hidden = dense_hidden_layers(head=hidden, units=256)

    # ## ----> Create output layer.
    outputs = tf.keras.layers.Dense(units=4, activation="linear")(hidden)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def dueling_dense_learning(input_size: Union[list or tuple]) -> tf.keras.Model:
    """
    Create Q-Network for Q-Learning.

    Parameters
    ----------
    input_size: int
        Input dimension

    Returns
    -------
    Model
        New model
    """
    # ## ----> Create input layer.
    inputs = tf.keras.layers.Input(shape=input_size)

    # ## ----> All hidden layers.
    hidden = tf.keras.layers.Flatten()(inputs)
    for _ in range(4):
        hidden = dense_hidden_layers(head=hidden, units=256)

    # ## ----> Value state.
    value_output = tf.keras.layers.Dense(units=1, activation="linear")(hidden)

    # ## ----> Advantage.
    advantage_output = tf.keras.layers.Dense(units=4, activation="linear")(hidden)

    # ## ----> Create output layer.
    outputs = tf.keras.layers.Add()([value_output, advantage_output])

    return tf.keras.Model(inputs=inputs, outputs=outputs)
