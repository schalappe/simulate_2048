# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

from keras import KerasTensor, Model, models, ops, utils
from numpy import ndarray


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
    obs_tensor = ops.convert_to_tensor(inputs, dtype="float16")

    if expand:
        return ops.expand_dims(obs_tensor, 0)
    return obs_tensor


class Network:

    def __init__(self, encoder: Model, dynamic: Model, predictor: Model):
        self._encoder = encoder
        self._dynamic = dynamic
        self._predictor = predictor

    @classmethod
    def from_path(cls, encoder_path: str, dynamic_path: str, predictor_path: str) -> Network:
        return cls(
            encoder=models.load_model(encoder_path),
            dynamic=models.load_model(dynamic_path),
            predictor=models.load_model(predictor_path),
        )

    def representation(self, observation: ndarray) -> ndarray:
        return self._encoder(ndarray_to_tensor(observation))[0]

    def dynamics(self, state: ndarray, action: int) -> Tuple[ndarray, float]:
        next_state, reward = self._dynamic(
            [ndarray_to_tensor(state), ndarray_to_tensor(utils.to_categorical([action], num_classes=4), expand=False)]
        )
        return next_state[0], reward[0][0]

    def prediction(self, state: ndarray) -> Tuple[ndarray, float]:
        policy, value = self._predictor(ndarray_to_tensor(state))
        return policy[0], value[0][0]
