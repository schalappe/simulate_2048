"""
Provides functions for binary encoding of state observations, including a standard encoding function
and a flatten-and-encode variant.
"""

from __future__ import annotations

from numpy import eye, int64, log2, ndarray


def encode(state: ndarray, encodage_size: int) -> ndarray:
    """
    Binary encode the observation given by the environment.

    This function takes a state observation and encodes it into a binary representation
    using the specified encoding size. It applies log2 transformation and one-hot encoding.

    Parameters
    ----------
    state : ndarray
        Observation given by the environment. Should be a numpy array of positive integers.
    encodage_size : int
        Maximum value to encode, determining the size of the one-hot encoded output.

    Returns
    -------
    ndarray
        Binary encoded observation. A 2D array where each row is a one-hot encoded value.

    Notes
    -----
    - The function applies log2 to non-zero elements of the input state.
    - Values in the state should be less than 2^encodage_size to avoid index errors.
    - The output will have shape (state.size, encodage_size).

    Example
    -------
    >>> import numpy as np
    >>> state = np.array([1, 2, 4, 8])
    >>> encode(state, encodage_size=4)
    array([[0, 1, 0, 0],
           [1, 0, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]])
    """
    obs = state.astype('float64')
    obs = log2(obs, where=obs != 0, out=obs)
    obs = obs.astype(int64, copy=False)
    return eye(encodage_size, dtype=int64)[obs]


def encode_flatten(state: ndarray, encodage_size: int) -> ndarray:
    """
    Flatten the observation given by the environment and then binary encode it.

    This function first flattens the input state array and then applies binary encoding
    using the specified encoding size. It combines the functionality of flattening and
    encoding in one step.

    Parameters
    ----------
    state : ndarray
        Observation given by the environment. Can be a multi-dimensional array of positive integers.
    encodage_size : int
        Maximum value to encode, determining the size of the one-hot encoded output for each element.

    Returns
    -------
    ndarray
        Flattened and binary encoded observation. A 1D array where each group of 'encodage_size'
        elements represents a one-hot encoded value.

    Notes
    -----
    - The function internally uses the 'encode' function after flattening the input state.
    - The output will have a total size of state.size * encodage_size.

    Example
    -------
    >>> import numpy as np
    >>> state = np.array([[1, 2], [4, 8]])
    >>> encode_flatten(state, encodage_size=4)
    array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    """
    obs = state.ravel().astype('float64')
    return encode(obs, encodage_size).ravel()
