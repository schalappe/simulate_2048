# -*- coding: utf-8 -*-
"""
Set of function for Actor Critic network.
"""
from typing import Union

import tensorflow as tf

from .core import hidden_block


def actor_and_critic_model(input_size: Union[list or tuple], dtype: str = "conv") -> tf.keras.Model:
    """
    Create an actor and critic model.

    Parameters
    ----------
    input_size: list or tuple
        Size of input
    dtype: str
        Which layer to use conv or dense

    Returns
    -------
    Model
        A2C model
    """
    # ## ----> Create input layer.
    inputs = tf.keras.layers.Input(shape=input_size)

    # ## ----> All hidden layers.
    hidden = hidden_block(head=inputs, size=256, dtype=dtype)

    # ## ----> Actor output.
    actor = tf.keras.layers.Dense(4, activation="softmax")(hidden)

    # ## ----> Critic output.
    critic = tf.keras.layers.Dense(1)(hidden)

    # ## ----> Model.
    return tf.keras.Model(inputs=inputs, outputs=[actor, critic])
