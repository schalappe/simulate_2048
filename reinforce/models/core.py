# -*- coding: utf-8 -*-
"""
Core function for all model.
"""
import tensorflow as tf


def conv_hidden_layers(head: tf.keras.layers.Layer, filters: int, kernel: int, strides: int) -> tf.keras.layers.Layer:
    """
    Add convolution layer.
    Parameters
    ----------
    head: Layer
        Previous layer
    filters: int
        Filter's size
    kernel: int
        Kernel's size
    strides: int
        Stride's size
    Returns
    -------
    Layer
        New layer
    """
    block = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding="same")(head)
    block = tf.keras.layers.BatchNormalization()(block)
    block = tf.keras.layers.ReLU()(block)
    return block


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
    return block


def conv_hidden_block(head: tf.keras.layers.Layer, size: int) -> tf.keras.layers.Layer:
    """
    Block of convolution layers.

    Parameters
    ----------
    head: Layer
        Previous layer
    size: int
        Size of the hidden block

    Returns
    -------
    Layer
        Convolution block
    """
    hidden = conv_hidden_layers(head=head, filters=size, kernel=2, strides=1)
    for _ in range(2):
        hidden = conv_hidden_layers(head=hidden, filters=size, kernel=2, strides=1)
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = dense_hidden_layers(head=hidden, units=size)

    return hidden


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
    hidden = tf.keras.layers.Flatten()(head)
    for _ in range(4):
        hidden = dense_hidden_layers(head=hidden, units=size)
    return hidden


def hidden_block(head: tf.keras.layers.Layer, size: int, dtype: str) -> tf.keras.layers.Layer:
    """
    Hidden block.

    Parameters
    ----------
    head: Layer
        Previous layer
    size: int
        Size of the hidden block
    dtype: str
        Type of layer to use

    Returns
    -------
    Layer
        hidden block of layer
    """
    if dtype == "conv":
        hidden = conv_hidden_block(head, size)
    else:
        hidden = dense_hidden_block(head, size)
    return hidden
