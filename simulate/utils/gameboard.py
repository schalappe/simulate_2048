# -*- coding: utf-8 -*-
"""
Simulator for the 2048 game.
"""
from typing import List, Tuple, Optional

from numpy import all as np_all
from numpy import any as np_any
from numpy import argwhere, array, ndarray, rot90, zeros_like
from numpy.random import default_rng

from .gamemove import can_move


def merge_column(column: ndarray) -> Tuple[int, ndarray]:
    """
    Merge adjacent equal values in a column and compute the total score.

    This function implements the core merging mechanic of the 2048 game for a single column.
    It combines adjacent equal values, moving them towards the start of the column,
    and calculates the score gained from these merges.

    Parameters
    ----------
    column : np.ndarray
        A 1D array representing one column of the game board.

    Returns
    -------
    Tuple[int, np.ndarray]
        A tuple containing:
        - The total score obtained from merging (int)
        - The new column configuration after merging (np.ndarray)

    Notes
    -----
    - Zeros (empty cells) are ignored and removed before merging.
    - Merging occurs from the start of the column towards the end.
    - Each value can only be merged once per call of this function.
    - The returned column may be shorter than the input if merges occurred.

    Examples
    --------
    >>> merge_column(array([2, 2, 4, 4, 8]))
    (12, array([4, 8, 8]))

    >>> merge_column(array([2, 2, 2, 2]))
    (8, array([4, 4]))

    >>> merge_column(array([2, 0, 2, 2]))
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

    This function applies the core 2048 game mechanics to the entire board:
    sliding all tiles to the left, merging where possible, and calculating the
    total score from all merges.

    Parameters
    ----------
    board : np.ndarray
        The game board represented as a 2D NumPy array.

    Returns
    -------
    Tuple[float, np.ndarray]
        A tuple containing:
        - The total score obtained from all merges (float)
        - The updated game board after sliding and merging (np.ndarray)

    Notes
    -----
    - The function operates on rows, effectively sliding left.
    - For other directions, the board should be rotated before calling this function.
    - Empty cells (zeros) are added to the right side of each row after merging.
    - The shape of the returned board is always the same as the input board.

    Examples
    --------
    >>> board = array([[2, 2, 0, 0],
    ...                   [0, 4, 4, 0],
    ...                   [8, 8, 8, 8],
    ...                   [2, 0, 0, 2]])
    >>> score, new_board = slide_and_merge(board)
    >>> score
    32.0
    >>> new_board
    array([[4, 0, 0, 0],
           [8, 0, 0, 0],
           [16, 16, 0, 0],
           [4, 0, 0, 0]])
    """
    result = zeros_like(board)
    score = 0.0

    for i, row in enumerate(board):
        score_row, merged_row = merge_column(row)
        score += score_row
        result[i, : len(merged_row)] = merged_row

    return score, result


def latent_state(state: ndarray, action: int) -> ndarray:
    """
    Compute the next state after applying an action, without adding a new tile.

    This function rotates the board according to the action, applies the slide and merge operation, and
    then rotates the board back to its original orientation.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.
    action : int
        The action to apply (0: left, 1: up, 2: right, 3: down).

    Returns
    -------
    ndarray
        The new state of the board after applying the action, before adding a new tile.

    Notes
    -----
    - This function does not add a new tile to the board after the move.
    - It's useful for calculating possible next states in game tree search algorithms.
    """
    rotated_board = rot90(state, k=action)
    _, updated_board = slide_and_merge(rotated_board)
    return rot90(updated_board, k=-action)


def after_state(state: ndarray) -> List[Tuple[ndarray, float]]:
    """
    Generate all possible next states after a move, including new tile placements.

    This function calculates all possible board states that could result from the next move, considering the
    random placement of a new tile (2 or 4).

    Parameters
    ----------
    state : ndarray
        The current state of the game board.

    Returns
    -------
    List[Tuple[ndarray, float]]
        A list of tuples, each containing:
        - A possible next state of the board (np.ndarray)
        - The probability of that state occurring (float)

    Notes
    -----
    - If there are no empty cells, it returns the current state with 100% probability.
    - The probabilities account for both the likelihood of choosing an empty cell
      and the probability of a 2 (90%) or 4 (10%) being placed.
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

    This function randomly selects empty cells in the game board and fills them with new tiles, simulating
    the game's tile-adding mechanic after each move.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.
    number_tile : int
        Number of new tiles to add.
    seed : int, optional
        Random number generator seed for reproducibility (default is None).

    Returns
    -------
    ndarray
        The updated state of the game board with new tiles added.

    Notes
    -----
    - New tiles have a 90% chance of being 2 and a 10% chance of being 4.
    - If there are fewer empty cells than requested new tiles, it fills all available cells.
    - If there are no empty cells, the function returns the input state unchanged.
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

    This function applies the given action to the current state, calculates the resulting board configuration
    and score, and adds a new tile to the board.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.
    action : int
        The action to apply (0: left, 1: up, 2: right, 3: down).
    seed : int, optional
        Random number generator seed for reproducibility (default is None).

    Returns
    -------
    Tuple[ndarray, float]
        A tuple containing:
        - The new state of the game board after the action and adding a new tile (np.ndarray)
        - The reward (score) obtained from this action (float)

    Notes
    -----
    - If the action results in no change to the board, the reward is 0 and no new tile is added.
    - A new tile (2 or 4) is added to a random empty cell after a valid move.
    - The seed ensures reproducibility of random tile placement.
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
    Check if the game is finished (no more moves possible).

    This function determines whether the game has ended by checking if there are any possible moves left on the board.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.

    Returns
    -------
    bool
        True if the game is finished (no more moves possible), False otherwise.

    Notes
    -----
    - The game is considered finished if there are no empty cells and no adjacent cells with the
    same value (i.e., no possible merges).
    - This function checks both horizontal and vertical adjacencies.
    """
    return np_all(state != 0) and not np_any(state[:-1] == state[1:]) and not np_any(state[:, :-1] == state[:, 1:])
