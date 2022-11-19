# -*- coding: utf-8 -*-
"""
Set of useful function for 2048 Simulation.
"""
from typing import Tuple

import numpy as np
from numba import njit, prange
from numpy import ndarray


@njit
def merge_column(column) -> Tuple:
    """
    Merge value in a column and compute score.

    Parameters
    ----------
    column:
        One column of the game board

    Returns
    -------
    tuple
        score and new column
    """
    result, score = [], []

    i = 1
    while i < len(column):
        if column[i] == column[i - 1]:
            score.append(column[i] + column[i - 1])
            result.append(column[i] + column[i - 1])
            i += 2
        else:
            result.append(column[i - 1])
            i += 1

    if i == len(column):
        result.append(column[i - 1])

    return score, result


@njit
def slide_and_merge(board: ndarray, size: int = 4) -> Tuple:
    """
    Slide board to the left and merge cells. Then compute score for agent.

    Parameters
    ----------
    board: ndarray
        Game board
    size: int
        Size of game board

    Returns
    -------
    tuple
        score and next board
    """
    result = np.zeros((4, 4), dtype=np.int64)
    _internal_score = np.zeros(4, dtype=np.int64)

    # ## ----> Loop over board
    for index in prange(4):
        row = board[index]
        row = np.extract(row > 0, row)
        _score, _result_row = merge_column(row)
        _internal_score[index] = np.sum(np.asarray(_score))
        row = padding(np.array(_result_row), size)
        result[index] = row

    score = np.sum(_internal_score)

    return score, result


@njit
def padding(array: ndarray, size=4) -> ndarray:
    """
    Pad an array with zero.

    Parameters
    ----------
    array: ndarray
        Array to pad
    size: int
        size of new array

    Returns
    -------
    ndarray
        Padded array
    """
    result = np.zeros(size)
    if len(array) == 0:
        return result
    result[:array.shape[0]] = array
    return result


@njit
def compute_penalties(board: ndarray) -> float:
    """
    Compute penalties for moved cells.

    Parameters
    ----------
    board: ndarray
        Game board

    Returns
    -------
    float
        Penalties
    """
    penalties = 0.0
    for row in board:
        idx = 0
        for idx_v, valeur in enumerate(row):
            if valeur != 0:
                if idx_v != idx:
                    penalties += 0.1 * valeur
                idx += 1
    return penalties
