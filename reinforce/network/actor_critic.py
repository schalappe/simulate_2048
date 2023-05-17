# -*- coding: utf-8 -*-
"""
Set of class that define a neural network.
"""
import tensorflow as tf

from .core import dense_block, dense_layer


class ActorCritic:
    """Combined actor-critic network."""

    def __call__(self, shape: int) -> tf.keras.Model:
        # ##: Create input layer.
        inputs = tf.keras.layers.Input(shape=(shape,))

        # ##: All hidden layers.
        hidden = dense_block(head=inputs, size=32, depth=10)

        # ##: Actor output.
        hidden_actor = dense_layer(head=hidden, units=16)
        actor = tf.keras.layers.Dense(4)(hidden_actor)

        # ##: Critic output.
        hidden_critic = dense_layer(head=hidden, units=16)
        critic = tf.keras.layers.Dense(1)(hidden_critic)

        # ##: Model.
        return tf.keras.Model(inputs=inputs, outputs=[actor, critic], name="policy")
