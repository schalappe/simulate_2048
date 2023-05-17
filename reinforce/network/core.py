# -*- coding: utf-8 -*-
"""
Core function for all model.
"""
import tensorflow as tf


def dense_layer(head: tf.keras.layers.Layer, units: int) -> tf.keras.layers.Layer:
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
    return block


def dense_block(head: tf.keras.layers.Layer, size: int, depth: int = 10) -> tf.keras.layers.Layer:
    """
    Block of dense layers.

    Parameters
    ----------
    head: Layer
        Previous layer
    size: int
        Size of the hidden block
    depth: int
        Depth the hidden network

    Returns
    -------
    Layer
        Dense block
    """
    hidden = dense_layer(head=head, units=size)
    for _ in range(depth - 1):
        hidden = dense_layer(head=hidden, units=size)
    return hidden
