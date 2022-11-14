# -*- coding: utf-8 -*-
"""
Set of useful function for 2048 Simulation.
"""
from typing import Tuple

import numpy as np
from numpy import ndarray
from numba import jit


@jit(nopython=True, cache=True)
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
    result, _internal_score = [], []

    # ## ----> Loop over board
    for row in board:
        row = np.extract(row > 0, row)
        _score, _result_row = merge_column(row)
        _internal_score.extend(_score)
        row = np.pad(np.array(_result_row), (0, size - len(_result_row)), "constant", constant_values=(0,))
        result.append(row)

    score = sum(_internal_score) if _internal_score else 0

    return score, np.array(result, dtype=np.int64)


@jit(nopython=True, cache=True)
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
