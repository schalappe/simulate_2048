# -*- coding: utf-8 -*-
"""
Provides a function to make predictions using a trained Keras model.
"""
from __future__ import annotations
from numpy import ndarray
from keras import Model
import keras


def make_prediction(model: Model, state: ndarray) -> ndarray:
    """
    Use a trained model to make a prediction based on the current game state.

    This function takes a Keras model and a game state as input, and returns the model's prediction.
    The state is converted to a tensor and expanded to match the expected input shape of the model.

    Parameters
    ----------
    model : Model
        A trained Keras model capable of processing game state data.
    state : ndarray
        The current state of the game, represented as a NumPy array.

    Returns
    -------
    ndarray
        The output of the model, representing the prediction or decision based on the input game state.

    Notes
    -----
    The function assumes that the input state is compatible with the model's expected input shape and data type.
    The state is converted to a float16 tensor before being passed to the model.

    Example
    -------
    >>> import numpy as np
    >>> game_state = np.array([...])  # Your game state here
    >>> trained_model = keras.models.load_model('your_model.h5')
    >>> prediction = make_prediction(trained_model, game_state)
    """
    obs_tensor = keras.ops.convert_to_tensor(state, dtype="float16")
    obs_tensor = keras.ops.expand_dims(obs_tensor, 0)
    return model(obs_tensor, training=False)
