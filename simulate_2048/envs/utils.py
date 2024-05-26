# -*- coding: utf-8 -*-
"""
Slide and merge columns for a 2048 game-like board.
"""
from typing import Tuple

import numpy as np
from numpy import ndarray


def merge_column(column: ndarray) -> Tuple[int, ndarray]:
    """
    Merge adjacent equal values in a column and compute the total score.

    This function takes a numpy array representing a column of the 2048 game and
    merges adjacent equal values. The merged values are doubled and added to the
    score. The function returns the total score from the merges and the new column
    configuration.

    Parameters
    ----------
    column : ndarray
        One column of the game board

    Returns
    -------
    Tuple[int, ndarray]
        A tuple containing:
        - An integer representing the total score obtained from merging values.
        - A NumPy array of integers representing the new column after merging.

    Notes
    -----
    - Adjacent equal values are merged into one value, which is double the original
      value, and this merged value contributes to the score.
    - If a value is merged, it is not available for further merges in the same step.
    - The function handles columns with any length, including empty columns.
    - Zeroes are not considered for merging.

    Examples
    --------
    >>> merge_column(np.array([2, 2, 4, 4, 8]))
    (12, array([4, 8, 8]))

    >>> merge_column(np.array([2, 2, 2, 2]))
    (8, array([4, 4]))

    >>> merge_column(np.array([2, 0, 2, 2]))
    (4, array([2, 4]))

    >>> merge_column(np.array([]))
    (0, array([], dtype=int))
    """
    # ##: Initialize the score.
    score = np.int64(0)
    merged_values = []

    # ##: Iterate over the column and merge values.
    i = 0
    while i < len(column):
        if i < len(column) - 1 and column[i] == column[i + 1] and column[i] != 0:
            merged_value = column[i] * 2
            score += merged_value
            merged_values.append(merged_value)
            i += 2
        else:
            merged_values.append(column[i])
            i += 1

    # ##: Convert result to NumPy array.
    result = np.array(merged_values, dtype=int)

    return score, result


def pad_array(array: ndarray, size: int = 4) -> ndarray:
    """
    Pad an array with zeros to a specified size.

    Parameters
    ----------
    array : ndarray
        The array to pad.
    size : int, optional
        The desired size of the padded array (default is 4).

    Returns
    -------
    ndarray
        The padded array.

    Notes
    -----
    - If the input array length is greater than or equal to the specified size,
      it is returned unchanged.
    - If the input array length is less than the specified size, zeros are appended
      to the end of the array until it reaches the desired size.
    - The size parameter determines the final length of the array after padding.
    - The input array can be of any dimension, but only the first axis (rows) is
      padded.
    - If the input array is empty, a zero-filled array of the specified size is returned.

    Examples
    --------
    >>> pad_array(np.array([1, 2, 3]), size=5)
    array([1, 2, 3, 0, 0])

    >>> pad_array(np.array([1, 2, 3]), size=2)
    array([1, 2, 3])

    >>> pad_array(np.array([]), size=4)
    array([0, 0, 0, 0])
    """
    if array.size >= size:
        return array

    padded_array = np.zeros(size, dtype=array.dtype)
    padded_array[: len(array)] = array
    return padded_array


def slide_and_merge(board: ndarray, size: int = 4) -> Tuple[float, ndarray]:
    """
    Slide the game board to the left, merge adjacent cells, and compute the score.

    Parameters
    ----------
    board : ndarray
        The game board represented as a 2D NumPy array.
    size : int, optional
        The size of the game board (default is 4).

    Returns
    -------
    Tuple[float, ndarray]
        A tuple containing:
        - The total score obtained after sliding and merging cells.
        - The updated game board after sliding and merging.

    Notes
    -----
    - This function slides all cells in each row of the game board to the left,
      merging adjacent cells with the same value.
    - The merging process doubles the value of merged cells and contributes to the score.
    - After sliding and merging, zeros are added to the right to maintain the board size.
    - The score is computed based on the merged cell values.
    - The function operates only on the rows of the game board.
    - If the input board is empty, a zero-filled board of the specified size is returned.

    Examples
    --------
    >>> board = np.array([[2, 2, 0, 0],
    ...                   [0, 4, 4, 0],
    ...                   [8, 8, 8, 8],
    ...                   [2, 0, 0, 2]])
    >>> slide_and_merge(board)
    (32.0, array([[4, 0, 0, 0],
                  [8, 0, 0, 0],
                  [16, 16, 0, 0],
                  [4, 0, 0, 0]]))
    """
    result = np.zeros_like(board)
    score = 0.0

    for index, row in enumerate(board):
        non_zero_cells = row[row > 0]
        score_row, result_row = merge_column(non_zero_cells)
        score += score_row
        padded_row = pad_array(result_row, size)
        result[index] = padded_row

    return score, result


def compute_penalties(board: ndarray) -> float:
    """
    Compute penalties for moved cells on the game board.

    This function calculates the penalties based on the movement of non-zero
    cells in each row of the game board. A penalty is incurred if a non-zero
    cell has moved from its original position.

    Parameters
    ----------
    board : np.ndarray
        The game board represented as a 2D NumPy array.

    Returns
    -------
    float
        The total penalties incurred due to moving cells.

    Notes
    -----
    - Penalties are calculated as 10% of the value of each non-zero cell that
      has moved from its original position.
    - The function operates only on the rows of the game board.
    - The game board is assumed to be non-empty and a 2D NumPy array.

    Examples
    --------
    >>> board = np.array([[2, 0, 4, 0],
    ...                   [0, 4, 0, 8],
    ...                   [8, 0, 0, 0],
    ...                   [2, 2, 2, 2]])
    >>> compute_penalties(board)
    0.5
    """

    penalties = 0.0
    for row in board:
        non_zero_indices = np.nonzero(row)[0]
        expected_indices = np.arange(len(non_zero_indices))
        moved_indices = non_zero_indices[non_zero_indices != expected_indices]
        penalties += 0.1 * np.sum(row[moved_indices])

    return penalties
