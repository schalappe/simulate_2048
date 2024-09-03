# -*- coding: utf-8 -*-
"""
Gameboard utilities for the 2048 game simulator.

This module provides core functionality for simulating the 2048 game, including board manipulation,
state transitions, and game logic. t contains functions for merging tiles, applying moves,
generating new game states, and determining game termination.
"""
from typing import List, Optional, Tuple

from numpy import all as np_all
from numpy import any as np_any
from numpy import argwhere, array, ndarray, rot90, zeros_like
from numpy.random import default_rng

from .gamemove import can_move


def merge_column(column: ndarray) -> Tuple[int, ndarray]:
    """
    Merge adjacent equal values in a column and compute the total score.

    Parameters
    ----------
    column : ndarray
        A 1D array representing one column of the game board.

    Returns
    -------
    score : int
        The total score obtained from merging.
    merged_column : ndarray
        The new column configuration after merging.

    Notes
    -----
    - Zeros (empty cells) are ignored and removed before merging.
    - Merging occurs from the start of the column towards the end.
    - Each value can only be merged once per function call.

    Examples
    --------
    >>> merge_column(np.array([2, 2, 4, 4, 8]))
    (12, array([4, 8, 8]))

    >>> merge_column(np.array([2, 2, 2, 2]))
    (8, array([4, 4]))

    >>> merge_column(np.array([2, 0, 2, 2]))
    (4, array([4, 2]))
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
    score : float
        The total score obtained from all merges.
    updated_board : ndarray
        The updated game board after sliding and merging.

    Notes
    -----
    - The function operates on rows, effectively sliding left.
    - For other directions, rotate the board before calling this function.
    - Empty cells (zeros) are added to the right side of each row after merging.

    Examples
    --------
    >>> board = np.array([[2, 2, 0, 0],
    ...                   [0, 4, 4, 0],
    ...                   [8, 8, 8, 8],
    ...                   [2, 0, 0, 2]])
    >>> score, new_board = slide_and_merge(board)
    >>> print(f"Score: {score}")
    Score: 32.0
    >>> print(new_board)
    [[ 4  0  0  0]
     [ 8  0  0  0]
     [16 16  0  0]
     [ 4  0  0  0]]
    """
    result = zeros_like(board)
    score = 0.0

    for i, row in enumerate(board):
        score_row, merged_row = merge_column(row)
        score += score_row
        result[i, : len(merged_row)] = merged_row

    return score, result


def latent_state(state: ndarray, action: int) -> Tuple[ndarray, float]:
    """
    Compute the next state after applying an action, without adding a new tile.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.
    action : int
        The action to apply (0: left, 1: up, 2: right, 3: down).

    Returns
    -------
    new_state : ndarray
        The new state of the board after applying the action.
    reward : float
        The reward obtained from this action.

    Notes
    -----
    This function does not add a new tile to the board after the move.
    """
    rotated_board = rot90(state, k=action)
    reward, updated_board = slide_and_merge(rotated_board)
    return rot90(updated_board, k=-action), reward


def after_state(state: ndarray) -> List[Tuple[ndarray, float]]:
    """
    Generate all possible next states after a move, including new tile placements.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.

    Returns
    -------
    list of tuple
        A list of tuples, each containing:
        - A possible next state of the board (ndarray)
        - The probability of that state occurring (float)

    Notes
    -----
    - If there are no empty cells, it returns the current state with 100% probability.
    - Probabilities account for both empty cell selection and new tile value (2 or 4).

    Examples
    --------
    >>> state = np.array([[2, 0], [0, 4]])
    >>> possible_states = after_state(state)
    >>> len(possible_states)
    4
    >>> for new_state, prob in possible_states:
    ...     print(f"Probability: {prob:.2f}")
    ...     print(new_state)
    ...     print()
    Probability: 0.45
    [[2 2]
     [0 4]]

    Probability: 0.45
    [[2 0]
     [2 4]]

    Probability: 0.05
    [[2 4]
     [0 4]]

    Probability: 0.05
    [[2 0]
     [4 4]]
    """
    # ##: Find empty cells.
    empty_cells = argwhere(state == 0)
    num_empty_cells = len(empty_cells)

    # ##: If no empty cells, return the post-action state with 100% probability.
    if num_empty_cells == 0:
        return [(state, 1.0)]

    # ##: Generate probable next states.
    probable_states = []
    for cell in empty_cells:
        for new_value in [2, 4]:
            new_state = state.copy()
            new_state[tuple(cell)] = new_value

            # Calculate the probability of this state
            prob = (0.9 if new_value == 2 else 0.1) / num_empty_cells
            probable_states.append((new_state, prob))

    return probable_states


def fill_cells(state: ndarray, number_tile: int, seed: Optional[int] = None) -> ndarray:
    """
    Fill empty cells with new tiles (2 or 4).

    Parameters
    ----------
    state : ndarray
        The current state of the game board.
    number_tile : int
        Number of new tiles to add.
    seed : int, optional
        Random number generator seed for reproducibility.

    Returns
    -------
    ndarray
        The updated state of the game board with new tiles added.

    Notes
    -----
    - New tiles have a 90% chance of being 2 and a 10% chance of being 4.
    - If there are fewer empty cells than requested, it fills all available cells.
    """
    rng = default_rng(seed)
    # ##: Only there still available places
    if not state.all():
        # ##: Randomly choose cell value between 2 and 4.
        values = rng.choice([2, 4], size=number_tile, p=[0.9, 0.1])

        # ##: Randomly choose cells positions in board.
        available_cells = argwhere(state == 0)
        chosen_indices = rng.choice(len(available_cells), size=number_tile, replace=False)

        # ##: Fill empty cells.
        state[tuple(available_cells[chosen_indices].T)] = values
    return state


def next_state(state: ndarray, action: int, seed: Optional[int] = None) -> Tuple[ndarray, float]:
    """
    Compute the next state and reward after applying an action.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.
    action : int
        The action to apply (0: left, 1: up, 2: right, 3: down).
    seed : int, optional
        Random number generator seed for reproducibility.

    Returns
    -------
    new_state : ndarray
        The new state of the game board after the action and adding a new tile.
    reward : float
        The reward (score) obtained from this action.

    Notes
    -----
    - If the action results in no change, the reward is 0 and no new tile is added.
    - A new tile (2 or 4) is added to a random empty cell after a valid move.

    Examples
    --------
    >>> state = np.array([[2, 2, 0, 0],
    ...                   [0, 4, 4, 0],
    ...                   [0, 0, 2, 0],
    ...                   [0, 0, 0, 2]])
    >>> new_state, reward = next_state(state, action=0, seed=42)
    >>> print(f"Reward: {reward}")
    Reward: 8.0
    >>> print(new_state)
    [[4 0 0 2]
     [8 0 0 0]
     [2 0 0 0]
     [2 0 0 0]]
    """
    rotated = rot90(state, k=action)
    if can_move(rotated):
        # ##: Applied action and get reward.
        reward, updated_board = slide_and_merge(rotated)
        state = rot90(updated_board, k=-action)

        # ##: Fill randomly one cell.
        state = fill_cells(state=state, number_tile=1, seed=seed)
        return state, reward
    return state, 0


def is_done(state: ndarray) -> bool:
    """
    Check if the game has ended by determining if any moves are possible.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.

    Returns
    -------
    bool
        True if the game is over (no moves possible), False otherwise.

    Notes
    -----
    The game is over when there are no empty cells AND no adjacent cells have the same value.

    Examples
    --------
    >>> is_done(np.array([[2, 4, 8, 16],
    ...                   [32, 64, 128, 256],
    ...                   [512, 1024, 2048, 4],
    ...                   [8, 16, 32, 64]]))
    True

    >>> is_done(np.array([[2, 2, 4, 8],
    ...                   [16, 32, 64, 128],
    ...                   [256, 512, 1024, 2048],
    ...                   [4, 8, 16, 32]]))
    False
    """
    return bool(
        np_all(state != 0) and not np_any(state[:-1] == state[1:]) and not np_any(state[:, :-1] == state[:, 1:])
    )
