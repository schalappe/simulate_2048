# -*- coding: utf-8 -*-
"""
Core function for all model.
"""
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
    initializer = tf.keras.initializers.HeUniform()
    block = tf.keras.layers.Dense(units=units, activation="relu", kernel_initializer=initializer)(head)
    block = tf.keras.layers.Dropout(rate=0.1)(block)
    return block


def dense_hidden_block(head: tf.keras.layers.Layer, size: int) -> tf.keras.layers.Layer:
    """
    Block of dense layers.

    Parameters
    ----------
    head: Layer
        Previous layer
    size: int
        Size of the hidden block

    Returns
    -------
    Layer
        Dense block
    """
    initializer = tf.keras.initializers.HeUniform()
    hidden = tf.keras.layers.Dense(units=size, activation="relu", kernel_initializer=initializer)(head)
    for _ in range(9):
        hidden = dense_hidden_layers(head=hidden, units=size)
    return hidden
