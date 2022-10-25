# -*- coding: utf-8 -*-
"""
Set of function for Proximal Policy Optimization.
"""
from typing import Tuple

import tensorflow as tf

from .core import dense_hidden_block


def dense_policy(input_size: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
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
    # ##: Create input layer.
    inputs = tf.keras.layers.Input(shape=input_size)

    # ##: All hidden layers.
    hidden = tf.keras.layers.Flatten()(inputs)

    # ##: Actor model.
    logit = dense_hidden_block(hidden, size=256)
    output_actor = tf.keras.layers.Dense(units=4, activation="softmax")(logit)
    actor = tf.keras.Model(inputs=inputs, outputs=output_actor)

    # ##: Critic model.
    value = dense_hidden_block(hidden, size=256)
    output_critic = tf.keras.layers.Dense(units=1, activation="linear")(value)
    critic = tf.keras.Model(inputs=inputs, outputs=tf.squeeze(output_critic, axis=1))

    return actor, critic
