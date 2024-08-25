# -*- coding: utf-8 -*-
"""
Slide and merge columns for a 2048 game-like board.
"""
from typing import List

from numpy import ndarray, rot90


def illegal_actions(state: ndarray) -> List[int]:
    """
    Returns the illegal actions for the current state of the game board.

    This function checks all possible actions (left, up, right, down) to determine which actions do not
    result in a change in the game board. If an action does not result in a different board state, it is
    considered an illegal action.

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
    for action in range(4):
        rotated = rot90(state, k=action)
        if can_move(rotated):
            continue
        illegal_moves.append(action)

    return illegal_moves


def legal_actions(state: ndarray) -> List[int]:
    """
    Determine the legal actions for the current state of the game board.

    This function checks all possible actions (left, up, right, down) to identify which actions would
    result in a change to the game board. These actions are considered legal moves.

    Parameters
    ----------
    state : ndarray
        The current state of the game board, represented as a 2D NumPy array.

    Returns
    -------
    List[int]
        A list of legal actions, where:
        - 0 represents left
        - 1 represents up
        - 2 represents right
        - 3 represents down

    Notes
    -----
    - The function uses rotation to check each direction efficiently.
    - An action is considered legal if it changes the board state.
    - This function is useful for determining valid moves in the game logic.
    """
    legal_moves = []
    for action in range(4):
        rotated = rot90(state, k=action)
        if can_move(rotated):
            legal_moves.append(action)
    return legal_moves


def can_move(board: ndarray) -> bool:
    """
    Check if any tile can move in the current direction (left).

    This function determines whether a move to the left is possible for the given board configuration. It checks
    for two conditions that allow movement:
    1. An empty cell (0) with a non-empty cell to its right.
    2. Two adjacent cells with the same non-zero value.

    Parameters
    ----------
    board : ndarray
        The game board to check, represented as a 2D NumPy array.

    Returns
    -------
    bool
        True if a move to the left is possible, False otherwise.

    Notes
    -----
    - This function checks for left movement. For other directions, rotate the
      board before calling this function.
    - It's a key component in determining legal moves and game end conditions.

    Examples
    --------
    >>> board1 = array([[2, 0, 0, 0],
    ...                    [2, 2, 0, 0],
    ...                    [4, 4, 0, 0],
    ...                    [8, 16, 32, 64]])
    >>> can_move(board1)
    True  # Movement is possible (2s can merge, 4s can merge)

    >>> board2 = array([[2, 4, 8, 16],
    ...                    [32, 64, 128, 256],
    ...                    [512, 1024, 2048, 4096],
    ...                    [8192, 16384, 32768, 65536]])
    >>> can_move(board2)
    False  # No movement possible to the left
    """
    rows, cols = board.shape
    for row in range(rows):
        for col in range(cols - 1):
            if board[row, col] == 0 and board[row, col + 1] != 0:
                return True
            if board[row, col] != 0 and board[row, col] == board[row, col + 1]:
                return True
    return False
