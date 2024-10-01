# -*- coding: utf-8 -*-
from typing import Tuple

from keras import Model, layers
from numpy import prod


def identity_block_dense(input_tensor: layers.Layer, units: int) -> layers.Layer:
    """
    Creates an identity block with dense layers.

    Parameters
    ----------
    input_tensor : layers.Layer
        Input layer to the identity block.
    units : int
        Number of units for the dense layers.

    Returns
    -------
    layers.Layer
        Output tensor after applying the identity block.
    """
    x = layers.Dense(units)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(units)(x)
    x = layers.BatchNormalization()(x)

    # Add the input_tensor to the output of the dense block
    x = layers.Add()([x, input_tensor])
    x = layers.ReLU()(x)

    return x


def build_representation_model(input_shape: Tuple[int, ...], encodage_size: int = 512) -> Model:
    inputs = layers.Input(shape=input_shape)

    # ##: Initial dense layer to process the input.
    x = layers.Dense(512, activation="relu")(inputs)

    # ##: Add identity blocks.
    for _ in range(5):
        x = identity_block_dense(x, 512)

    # ##: Output dense layer for hidden state.
    outputs = layers.Dense(encodage_size, activation="relu", name="representation_model")(x)

    return Model(inputs=inputs, outputs=outputs)


def build_dynamics_model(state_shape: Tuple[int, ...], action_size: int) -> Model:
    # ##: Input layer for the state.
    input_state = layers.Input(shape=state_shape)
    dense_state = layers.Dense(512, activation="relu")(input_state)

    # ##: Input layer for the action.
    input_action = layers.Input(shape=(action_size,))
    dense_action = layers.Dense(512, activation="relu")(input_action)

    # ##: Initial dense layer to process the input.
    x = layers.Add()([dense_state, dense_action])

    # ##: Add identity blocks.
    for _ in range(5):
        x = identity_block_dense(x, 512)

    # ##: Hidden neuron layer for the next state.
    hidden_state = layers.Dense(512, activation="relu")(x)
    next_state = layers.Dense(prod(state_shape), activation="relu", name="next_state")(hidden_state)
    next_state = layers.Reshape(state_shape)(next_state)

    # ##: Hidden neuron layer for the reward.
    hidden_reward = layers.Dense(512, activation="relu")(x)
    reward = layers.Dense(1, name="reward")(hidden_reward)

    return Model([input_state, input_action], [next_state, reward], name="dynamics_model")


def build_prediction_model(state_shape: Tuple[int, ...], action_size: int) -> Model:
    inputs = layers.Input(shape=state_shape)

    # ##: Initial dense layer to process the input.
    x = layers.Dense(512, activation="relu")(inputs)

    # ##: Add identity blocks.
    for _ in range(5):
        x = identity_block_dense(x, 512)

    # ##: Hidden neuron layer for the policy.
    hidden_policy = layers.Dense(512, activation="relu")(x)
    policy = layers.Dense(action_size, activation="softmax", name="policy")(hidden_policy)

    # ##: Hidden neuron layer for the value.
    hidden_value = layers.Dense(512, activation="relu")(x)
    value = layers.Dense(1, name="value")(hidden_value)

    return Model(inputs, [policy, value], name="prediction_model")
