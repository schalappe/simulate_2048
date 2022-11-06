# -*- coding: utf-8 -*-
"""
Core function for all model.
"""
from typing import List, Tuple, Union

import tensorflow as tf


class PolicyNetwork:
    """
    Policy network class.
    """
    @classmethod
    def dense_layer(cls, head: tf.keras.layers.Layer, units: int) -> tf.keras.layers.Layer:
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

    def dense_block(self, head: tf.keras.layers.Layer, size: int, depth: int = 10) -> tf.keras.layers.Layer:
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
        hidden = self.dense_layer(head=head, units=size)
        for _ in range(depth - 1):
            hidden = self.dense_layer(head=hidden, units=size)
        return hidden

    def __call__(self, shape: Union[List, Tuple]) -> tf.keras.Model:
        # ##: Create input layer.
        inputs = tf.keras.layers.Input(shape=shape)

        # ##: All hidden layers.
        hidden = self.dense_block(head=inputs, size=256, depth=19)

        # ##: Actor output.
        hidden_actor = self.dense_layer(head=hidden, units=256)
        actor = tf.keras.layers.Dense(4)(hidden_actor)

        # ##: Critic output.
        hidden_critic = self.dense_layer(head=hidden, units=256)
        critic = tf.keras.layers.Dense(1, activation="tanh")(hidden_critic)

        # ##: Model.
        return tf.keras.Model(inputs=inputs, outputs=[actor, critic], name="policy")
