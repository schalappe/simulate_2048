# -*- coding: utf-8 -*-
"""
Provides functions to construct neural network models for reinforcement learning tasks.
"""
from keras import Model
from keras import layers


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


def build_model_with_identity_blocks(
    state_size: int, action_size: int = 4, num_blocks: int = 3, units: int = 256
) -> Model:
    """
    Construct a deep Q-Network model with dense identity blocks.

    Parameters
    ----------
    state_size : int
        The size of the input state (496 for the 2048 game with binary encoding).
    action_size : int, optional
        The number of possible actions (default is 4).
    num_blocks : int, optional
        The number of identity blocks to include in the model.
    units : int, optional
        The number of units for the dense layers.

    Returns
    -------
    Model
        A Keras Model instance representing the DQN with dense identity blocks.
    """
    inputs = layers.Input(shape=(state_size,))

    # Initial dense layer to process the input
    x = layers.Dense(units, activation="relu")(inputs)

    # Add identity blocks
    for _ in range(num_blocks):
        x = identity_block_dense(x, units)

    # Output dense layer for Q-value prediction
    outputs = layers.Dense(action_size)(x)

    return Model(inputs=inputs, outputs=outputs)


def build_attention_model(game_size: int, encodage_size: int, action_size: int = 4) -> Model:
    """
    Construct an Attention-based model for Q-learning.

    This function creates a deep neural network incorporating self-attention mechanisms,
    designed for reinforcement learning tasks, particularly Q-learning.

    Parameters
    ----------
    game_size : int
        The size of the game board (assuming a square board).
    encodage_size : int
        The size of the encoding for each cell in the game board.
    action_size : int, optional
        The number of possible actions in the environment (default is 4).

    Returns
    -------
    Model
        A Keras Model instance representing the Attention-based Q-network.

    Notes
    -----
    The model architecture:
    1. Input layer
    2. Reshape layer to prepare for attention
    3. Multiple attention blocks, each containing:
       - Multi-head self-attention layer
       - Add and normalize layers
       - Feed-forward network
       - Add and normalize layers
    4. Global average pooling
    5. Dense layers
    6. Output layer producing Q-values for each action

    The self-attention mechanism allows the model to focus on different parts of the input state,
    potentially capturing long-range dependencies across the game board.
    """
    inputs = layers.Input(shape=(game_size * game_size * encodage_size,))

    # ##: Reshape input to add sequence dimension.
    x = layers.Reshape((game_size * game_size, encodage_size))(inputs)

    # ##: Multiple attention blocks.
    for _ in range(3):
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        attention_output = attention(x, x)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)

        # ##: Feed-forward network
        ff = layers.Dense(128, activation="relu")(x)
        ff = layers.Dense(31, activation="relu")(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)

    # ##: Global pooling and dense layers.
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)

    outputs = layers.Dense(action_size)(x)
    return Model(inputs=inputs, outputs=outputs)
