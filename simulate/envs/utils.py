# -*- coding: utf-8 -*-
"""
Slide and merge columns for a 2048 game-like board.
"""
from typing import List, Tuple

from numpy import array, array_equal, ndarray, rot90, zeros_like


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
        Total score obtained from merging values and the new column after merging.

    Notes
    -----
    - Adjacent equal values are merged into one value, which is double the original
      value, and this merged value contributes to the score.
    - If a value is merged, it is not available for further merges in the same step.
    - The function handles columns with any length, including empty columns.
    - Zeroes are not considered for merging.

    Examples
    --------
    >>> merge_column(array([2, 2, 4, 4, 8]))
    (12, array([4, 8, 8]))

    >>> merge_column(array([2, 2, 2, 2]))
    (8, array([4, 4]))

    >>> merge_column(array([2, 0, 2, 2]))
    (4, array([2, 4]))

    >>> merge_column(array([]))
    (0, array([], dtype=int))
    """
    # ##: Handle empty columns.
    non_zero = column[column != 0]
    if len(non_zero) <= 1:
        return 0, non_zero

    # ##: Initialize the score.
    result = []
    score = 0

    # ##: Iterate over the column and merge values.
    i = 0
    while i < len(non_zero) - 1:
        if non_zero[i] == non_zero[i + 1]:
            merged = non_zero[i] * 2
            result.append(merged)
            score += merged
            i += 2
        else:
            result.append(non_zero[i])
            i += 1

    if i == len(non_zero) - 1:
        result.append(non_zero[-1])

    return score, array(result, dtype=column.dtype)


def slide_and_merge(board: ndarray) -> Tuple[float, ndarray]:
    """
    Slide the game board to the left, merge adjacent cells, and compute the score.

    Parameters
    ----------
    board : ndarray
        The game board represented as a 2D NumPy array.

    Returns
    -------
    Tuple[float, ndarray]
        Total score obtained after sliding and merging cells, and the updated game board.

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
    >>> board = array([[2, 2, 0, 0],
    ...                   [0, 4, 4, 0],
    ...                   [8, 8, 8, 8],
    ...                   [2, 0, 0, 2]])
    >>> slide_and_merge(board)
    (32.0, array([[4, 0, 0, 0],
                  [8, 0, 0, 0],
                  [16, 16, 0, 0],
                  [4, 0, 0, 0]]))
    """
    result = zeros_like(board)
    score = 0.0

    for i, row in enumerate(board):
        score_row, merged_row = merge_column(row)
        score += score_row
        result[i, : len(merged_row)] = merged_row

    return score, result


def illegal_actions(state: ndarray) -> List[int]:
    """
    Returns the illegal actions for the current state of the game board.

    This function checks all possible actions (left, up, right, down) to determine
    which actions do not result in a change in the game board. If an action does not
    result in a different board state, it is considered an illegal action.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.

    Returns
    -------
    List[int]
        A list of illegal actions (0 for left, 1 for up, 2 for right, 3 for down).

    Notes
    -----
    - The game board is assumed to be a 2D NumPy array.
    - The function rotates the board to align the action direction with merging logic.
    - Illegal actions are those that do not result in a different board state after merging.
    """
    illegal_moves = []

    # Check all possible moves
    for action in range(4):
        rotated_board = rot90(state, k=action)
        _, updated_board = slide_and_merge(rotated_board)
        if array_equal(rotated_board, updated_board):
            illegal_moves.append(action)

    return illegal_moves
