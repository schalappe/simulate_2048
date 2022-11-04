# -*- coding: utf-8 -*-
"""
Core function for all model.
"""
import tensorflow as tf


def dense_hidden_layers(head: tf.keras.layers.Layer, units: int, activation: str = "relu") -> tf.keras.layers.Layer:
    """
    Add dense layer.

    Parameters
    ----------
    head: Layer
        Previous layer
    units: int
        Number of neurons in this layer
    activation: str
        Output activation function

    Returns
    -------
    Layer
        New layer
    """
    if activation == "relu":
        initializer = tf.keras.initializers.HeUniform()
    else:
        initializer = "glorot_uniform"
    block = tf.keras.layers.Dense(units=units, activation=activation, kernel_initializer=initializer)(head)
    block = tf.keras.layers.Dropout(rate=0.1)(block)
    return block


def dense_hidden_block(head: tf.keras.layers.Layer, size: int, activation: str = "relu") -> tf.keras.layers.Layer:
    """
    Block of dense layers.

    Parameters
    ----------
    head: Layer
        Previous layer
    size: int
        Size of the hidden block
    activation: str
        Output activation function

    Returns
    -------
    Layer
        Dense block
    """
    hidden = dense_hidden_layers(head=head, units=size, activation=activation)
    for _ in range(9):
        hidden = dense_hidden_layers(head=hidden, units=size, activation=activation)
    return hidden
