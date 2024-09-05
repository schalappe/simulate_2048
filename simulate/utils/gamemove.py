# -*- coding: utf-8 -*-
"""
Game move utilities for the 2048 game simulator.

This module provides functions for determining legal and illegal moves in the 2048 game. It includes
utilities for analyzing the game board state, identifying possible actions, and determining which moves
are valid or invalid based on the current configuration.
"""
from typing import List

from numpy import ndarray, rot90


def illegal_actions(state: ndarray) -> List[int]:
    """
    Determine illegal actions for the current game board state.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.

    Returns
    -------
    List[int]
        A list of illegal actions (0: left, 1: up, 2: right, 3: down).

    Notes
    -----
    - An action is considered illegal if it doesn't change the board state.
    - The function uses board rotation to check all directions efficiently.
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
    Determine legal actions for the current game board state.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.

    Returns
    -------
    List[int]
        A list of legal actions (0: left, 1: up, 2: right, 3: down).

    Notes
    -----
    - Legal actions are those that result in a change to the game board.
    - This function is the complement of `illegal_actions`.
    """
    legal_moves = []
    for action in range(4):
        rotated = rot90(state, k=action)
        if can_move(rotated):
            legal_moves.append(action)
    return legal_moves


def can_move(board: ndarray) -> bool:
    """
    Check if any tile can move left on the given board.

    Parameters
    ----------
    board : ndarray
        The game board to check.

    Returns
    -------
    bool
        True if a left move is possible, False otherwise.

    Notes
    -----
    - This function only checks for left movement.
    - For other directions, rotate the board before calling this function.
    - A move is possible if there's an empty cell to the left of a non-empty cell,
      or if two adjacent cells have the same non-zero value.
    """
    rows, cols = board.shape
    for row in range(rows):
        for col in range(cols - 1):
            if board[row, col] == 0 and board[row, col + 1] != 0:
                return True
            if board[row, col] != 0 and board[row, col] == board[row, col + 1]:
                return True
    return False
