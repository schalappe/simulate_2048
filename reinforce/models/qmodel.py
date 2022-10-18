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
    Layer:
        New layer
    """
    block = tf.keras.layers.Dense(units=units, kernel_initializer="he_uniform", activation="relu")(head)
    block = tf.keras.layers.Dropout(rate=0.2)(block)
    return block


def conv_hidden_layers(head: tf.keras.layers.Layer, filters: int, kernel: int, strides: int) -> tf.keras.layers.Layer:
    block = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding="same")(head)
    block = tf.keras.layers.BatchNormalization()(block)
    block = tf.keras.layers.ReLU()(block)
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
    Model:
        New model
    """
    # ## ----> Create input layer.
    inputs = tf.keras.layers.Input(shape=input_size)

    # ## ----> All hidden layers.
    hidden = tf.keras.layers.Flatten()(inputs)
    for _ in range(5):
        hidden = dense_hidden_layers(head=hidden, units=256)

    # ## ----> Create output layer.
    outputs = tf.keras.layers.Dense(units=4, activation="linear")(hidden)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def conv_learning(input_size: Union[list or tuple]) -> tf.keras.Model:

    # ## ----> Create input layer.
    inputs = tf.keras.layers.Input(shape=input_size)

    # ## ----> All hidden layers.
    hidden = conv_hidden_layers(head=inputs, filters=16, kernel=2, strides=1)
    for _ in range(4):
        hidden = conv_hidden_layers(head=hidden, filters=256, kernel=2, strides=1)
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(units=256, activation="relu")(hidden)
    hidden = tf.keras.layers.Dropout(0.2)(hidden)

    # ## ----> Create output layer.
    outputs = tf.keras.layers.Dense(units=4, activation="linear")(hidden)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
