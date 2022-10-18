# -*- coding: utf-8 -*-
"""
Set of function for Proximal Policy Optimization
"""
from typing import Any, Tuple, Union

import tensorflow as tf


def dense_hidden_layers(head: tf.keras.layers.Layer, units: int, activation: Any) -> tf.keras.layers.Layer:
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
    block = tf.keras.layers.Dense(units=units, activation=activation)(head)
    block = tf.keras.layers.Dropout(rate=0.2)(block)
    return block


def hidden_mlp(hidden_layers):
    for _ in range(4):
        hidden_layers = dense_hidden_layers(head=hidden_layers, units=256, activation=tf.tanh)
    return hidden_layers


def dense_policy(input_size: Union[list or tuple]) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Create Q-Network for Q-Learning.

    Parameters
    ----------
    input_size: tuple or list
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

    # ## ----> Actor model.
    logit = hidden_mlp(hidden)
    output_actor = tf.keras.layers.Dense(units=4, activation=None)(logit)
    actor = tf.keras.Model(inputs=inputs, outputs=output_actor)

    # ## ----> Critic model.
    value = hidden_mlp(hidden)
    output_critic = tf.keras.layers.Dense(units=1, activation=None)(value)
    critic = tf.keras.Model(inputs=inputs, outputs=tf.squeeze(output_critic, axis=1))

    return actor, critic
