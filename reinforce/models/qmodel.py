# -*- coding: utf-8 -*-
"""
Set of function for Q-Learning.
"""
from typing import Union

import tensorflow as tf

from .core import hidden_block


def dense_learning(input_size: Union[list or tuple], dtype: str = "conv") -> tf.keras.Model:
    """
    Create Q-Network for Q-Learning.

    Parameters
    ----------
    input_size: list or tuple
        Input dimension
    dtype: str
        Type of layer to use

    Returns
    -------
    Model
        New model
    """
    # ## ----> Create input layer.
    inputs = tf.keras.layers.Input(shape=input_size)

    # ## ----> All hidden layers.
    hidden = hidden_block(head=inputs, size=256, dtype=dtype)

    # ## ----> Create output layer.
    outputs = tf.keras.layers.Dense(units=4)(hidden)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def dueling_dense_learning(input_size: Union[list or tuple], dtype: str = "conv") -> tf.keras.Model:
    """
    Create Q-Network for Q-Learning.

    Parameters
    ----------
    input_size: list or tuple
        Input dimension
    dtype: str
        Type of layer to use

    Returns
    -------
    Model
        New model
    """
    # ## ----> Create input layer.
    inputs = tf.keras.layers.Input(shape=input_size)

    # ## ----> All hidden layers.
    hidden = hidden_block(head=inputs, size=256, dtype=dtype)

    # ## ----> Value state.
    value_output = tf.keras.layers.Dense(units=1)(hidden)

    # ## ----> Advantage.
    advantage_output = tf.keras.layers.Dense(units=4)(hidden)

    # ## ----> Create output layer.
    outputs = tf.keras.layers.Add()([value_output, advantage_output])

    return tf.keras.Model(inputs=inputs, outputs=outputs)
