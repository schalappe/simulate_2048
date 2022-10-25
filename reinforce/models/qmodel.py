# -*- coding: utf-8 -*-
"""
Set of function for Deep Q-Learning.
"""

import tensorflow as tf

from .core import dense_hidden_block


def dense_learning(input_size: int) -> tf.keras.Model:
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
    # ##: Create input layer.
    inputs = tf.keras.layers.Input(shape=(input_size,))

    # ##: All hidden layers.
    hidden = dense_hidden_block(head=inputs, size=256)

    # ##: Create output layer.
    outputs = tf.keras.layers.Dense(units=4)(hidden)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def dueling_dense_learning(input_size: int) -> tf.keras.Model:
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
    # ##: Create input layer.
    inputs = tf.keras.layers.Input(shape=(input_size,))

    # ##: All hidden layers.
    hidden = dense_hidden_block(head=inputs, size=256)

    # ##: Value state.
    value_output = tf.keras.layers.Dense(units=1)(hidden)

    # ##: Advantage.
    advantage_output = tf.keras.layers.Dense(units=4)(hidden)

    # ##: Create output layer.
    outputs = tf.keras.layers.Add()([value_output, advantage_output])

    return tf.keras.Model(inputs=inputs, outputs=outputs)
