# -*- coding: utf-8 -*-
"""
Provides functions to construct neural network models for reinforcement learning tasks.
"""
from typing import Tuple
from keras import Model
from keras import layers


def build_resnet_model(state_size: Tuple[int, int, int], action_size: int = 4) -> Model:
    """
    Construct a ResNet-based model for Q-learning.

    This function creates a deep neural network based on the ResNet architecture,
    designed for reinforcement learning tasks, particularly Q-learning.

    Parameters
    ----------
    state_size : Tuple[int, int, int]
        The dimensions of the input state (height, width, channels).
    action_size : int, optional
        The number of possible actions in the environment (default is 4).

    Returns
    -------
    Model
        A Keras Model instance representing the ResNet-based Q-network.

    Notes
    -----
    The model architecture:
    1. Initial convolutional layer
    2. Five residual blocks
    3. Global average pooling
    4. Dense layers
    5. Output layer producing Q-values for each action

    The residual blocks help in training deeper networks by introducing skip connections,
    which mitigate the vanishing gradient problem.
    """

    def residual_block(input_layer: layers.Layer, filters: int, kernel_size: int = 3):
        """
            Create a residual block for the ResNet architecture.
        Parameters
        ----------
            input_layer : layers.Layer
                The input layer to the residual block.
            filters : int
                The number of filters in the convolutional layers.
            kernel_size : int, optional
                The size of the convolutional kernel (default is 3).
        Returns
        -------
            layers.Layer
                The output of the residual block.
        """
        y = layers.Conv2D(filters, kernel_size, padding="same")(input_layer)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Conv2D(filters, kernel_size, padding="same")(y)
        y = layers.BatchNormalization()(y)
        return layers.ReLU()(layers.Add()([input_layer, y]))

    inputs = layers.Input(shape=state_size)

    # Convolution layers
    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ResNet layers
    for _ in range(5):  # 5 residual blocks
        x = residual_block(x, 32)
    # Dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(action_size)(x)

    return Model(inputs=inputs, outputs=outputs)


def build_attention_model(state_size: Tuple[int, int, int], action_size: int = 4) -> Model:
    """
    Construct an Attention-based model for Q-learning.

    This function creates a deep neural network incorporating self-attention mechanisms,
    designed for reinforcement learning tasks, particularly Q-learning.

    Parameters
    ----------
    state_size : Tuple[int, int, int]
        The dimensions of the input state (height, width, channels).
    action_size : int, optional
        The number of possible actions in the environment (default is 4).

    Returns
    -------
    Model
        A Keras Model instance representing the Attention-based Q-network.

    Notes
    -----
    The model architecture:
    1. Initial convolutional layers
    2. Reshape layer to prepare for attention
    3. Multi-head self-attention layer
    4. Global average pooling
    5. Dense layers
    6. Output layer producing Q-values for each action

    The self-attention mechanism allows the model to focus on different parts of the input state,
    potentially capturing long-range dependencies.
    """
    inputs = layers.Input(shape=state_size)

    # Convolution layers
    x = layers.Conv2D(64, 2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 2, padding="same", activation="relu")(x)
    x = layers.Reshape((-1, 128))(x)

    # Self-attention layers
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)
    x = attention(x, x)

    # Dense layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(action_size)(x)

    return Model(inputs=inputs, outputs=outputs)
