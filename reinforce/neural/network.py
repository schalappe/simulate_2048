"""
Unified neural network wrapper for MuZero-style models.

This module provides the Network class that unifies the three MuZero models
(representation, dynamics, prediction) into a single interface.
"""

from __future__ import annotations

from keras import KerasTensor, Model, models, ops, utils
from numpy import ndarray

NUM_ACTIONS = 4  # 2048 game: left, up, right, down


def ndarray_to_tensor(inputs: ndarray, expand: bool = True) -> KerasTensor:
    """
    Converts a NumPy array to a Keras tensor.

    This function accepts a 2D NumPy array as input and converts it into a Keras Tensor.
    The resulting Keras Tensor is automatically expanded to match the shape (batch_size, num_features).

    Parameters
    ----------
    inputs : ndarray
        A 2D NumPy array with shape (num_samples, num_features).
    expand : bool
        Whether or not expand the tensor.

    Returns
    -------
    KerasTensor
        A Keras tensor with shape (1, num_samples, num_features) or simply (num_samples, num_features)
            if the input is a single sample.

    Note
    ----
    The dtype of the output Keras Tensor is 'float16'.

    Examples
    --------
    >>> import numpy as np
    >>> ndarray_to_tensor(np.array([[3.0], [4.0]]))
    <KerasTensor: shape=(1, 2), dtype=float16, name='ndarray_to_tensor/ExpandDims', tensorflow_grad>
    """
    obs_tensor = ops.convert_to_tensor(inputs, dtype='float16')

    if expand:
        return ops.expand_dims(obs_tensor, 0)
    return obs_tensor


class Network:
    """
    Unified wrapper for MuZero-style neural network models.

    This class provides a clean interface for the three models:
    - Representation (encoder): observation -> hidden state
    - Dynamics: (state, action) -> (next_state, reward)
    - Prediction: state -> (policy, value)

    Attributes
    ----------
    _encoder : Model
        The representation model.
    _dynamic : Model
        The dynamics model.
    _predictor : Model
        The prediction model.
    """

    def __init__(self, encoder: Model, dynamic: Model, predictor: Model):
        """
        Initialize the Network with pre-built models.

        Parameters
        ----------
        encoder : Model
            The representation model.
        dynamic : Model
            The dynamics model.
        predictor : Model
            The prediction model.
        """
        self._encoder = encoder
        self._dynamic = dynamic
        self._predictor = predictor

    @classmethod
    def from_path(cls, encoder_path: str, dynamic_path: str, predictor_path: str) -> Network:
        """
        Load a Network from saved model files.

        Parameters
        ----------
        encoder_path : str
            Path to the saved encoder model.
        dynamic_path : str
            Path to the saved dynamics model.
        predictor_path : str
            Path to the saved predictor model.

        Returns
        -------
        Network
            A Network instance with loaded models.
        """
        return cls(
            encoder=models.load_model(encoder_path),
            dynamic=models.load_model(dynamic_path),
            predictor=models.load_model(predictor_path),
        )

    def representation(self, observation: ndarray) -> ndarray:
        """
        Encode an observation into a hidden state.

        Parameters
        ----------
        observation : ndarray
            The raw observation.

        Returns
        -------
        ndarray
            The hidden state representation.
        """
        return self._encoder(ndarray_to_tensor(observation))[0]

    def dynamics(self, state: ndarray, action: int) -> tuple[ndarray, float]:
        """
        Predict the next state and reward given current state and action.

        Parameters
        ----------
        state : ndarray
            The current hidden state.
        action : int
            The action to take (0-3 for 2048).

        Returns
        -------
        Tuple[ndarray, float]
            The predicted next state and reward.
        """
        next_state, reward = self._dynamic(
            [
                ndarray_to_tensor(state),
                ndarray_to_tensor(utils.to_categorical([action], num_classes=NUM_ACTIONS), expand=False),
            ]
        )
        return next_state[0], reward[0][0]

    def prediction(self, state: ndarray) -> tuple[ndarray, float]:
        """
        Predict policy and value from a hidden state.

        Parameters
        ----------
        state : ndarray
            The hidden state.

        Returns
        -------
        Tuple[ndarray, float]
            The policy probabilities and value estimate.
        """
        policy, value = self._predictor(ndarray_to_tensor(state))
        return policy[0], value[0][0]
