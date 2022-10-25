# -*- coding: utf-8 -*-
"""
Set of function for Actor Critic network.
"""
import tensorflow as tf

from .core import dense_hidden_block


def actor_and_critic_model(input_size: int) -> tf.keras.Model:
    """
    Create an actor and critic model.

    Parameters
    ----------
    input_size: int
        Size of input

    Returns
    -------
    Model
        A2C model
    """
    # ##: Create input layer.
    inputs = tf.keras.layers.Input(shape=(input_size,))

    # ##: All hidden layers.
    hidden = dense_hidden_block(head=inputs, size=256)

    # ##: Actor output.
    actor = tf.keras.layers.Dense(4)(hidden)

    # ##: Critic output.
    critic = tf.keras.layers.Dense(1)(hidden)

    # ##: Model.
    return tf.keras.Model(inputs=inputs, outputs=[actor, critic])
