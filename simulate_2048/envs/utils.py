# -*- coding: utf-8 -*-
"""
Slide and merge columns.
"""
from typing import List, Tuple

import numpy as np
from numpy import ndarray


def merge_column(column) -> Tuple[List[int], List[int]]:
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
    result[: array.shape[0]] = array
    return result


def slide_and_merge(board: ndarray, size: int = 4) -> Tuple[float, ndarray]:
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
    score = 0.0

    for index in range(4):
        row = board[index]
        row = np.extract(row > 0, row)
        score_row, result_row = merge_column(row)
        score += np.sum(np.asarray(score_row))
        row = padding(np.array(result_row), size)
        result[index] = row

    return score, result


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
    for index in range(len(board)):
        idx = 0
        for idx_v, valeur in enumerate(board[index]):
            if valeur != 0:
                if idx_v != idx:
                    penalties += 0.1 * valeur
                idx += 1
    return penalties
