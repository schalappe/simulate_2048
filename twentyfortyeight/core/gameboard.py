"""
Core functionality for simulating the 2048 game, including board manipulation and game logic.
"""

from numpy import all as np_all
from numpy import any as np_any
from numpy import argwhere, array, ndarray, rot90, zeros_like
from numpy.random import PCG64DXSM, default_rng

from twentyfortyeight.core.gamemove import can_move

# ##>: Tile spawn probabilities for 2048 game (90% for 2, 10% for 4).
TILE_SPAWN_PROBS: dict[int, float] = {2: 0.9, 4: 0.1}

# ##>: Pre-computed tile values and probabilities for fast sampling.
_TILE_VALUES = [2, 4]
_TILE_PROBS = [0.9, 0.1]

# ##>: Module-level generator for performance (avoids repeated initialization).
_GENERATOR = default_rng(PCG64DXSM())


def merge_column(column: ndarray) -> tuple[int, ndarray]:
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


def slide_and_merge(board: ndarray) -> tuple[float, ndarray]:
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
    """
    result = zeros_like(board)
    score = 0.0

    for i, row in enumerate(board):
        score_row, merged_row = merge_column(row)
        score += score_row
        result[i, : len(merged_row)] = merged_row

    return score, result


def latent_state(state: ndarray, action: int) -> tuple[ndarray, float]:
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


def after_state(state: ndarray) -> list[tuple[ndarray, float]]:
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

            prob = TILE_SPAWN_PROBS[new_value] / num_empty_cells
            probable_states.append((new_state, prob))

    return probable_states


def after_state_lazy(state: ndarray) -> tuple[ndarray, list[tuple[int, int]], int]:
    """
    Prepare lazy generation data for stochastic outcomes.

    Unlike ``after_state()``, this does NOT create any state copies upfront. It returns
    the base state and empty cell positions, allowing outcomes to be generated on-demand
    via ``generate_outcome()``.

    Parameters
    ----------
    state : ndarray
        The current state of the game board (post-action, pre-tile-spawn).

    Returns
    -------
    tuple
        A tuple containing:
        - state : ndarray - The base state reference (not copied).
        - empty_cells : List[Tuple[int, int]] - Positions of empty cells as (row, col).
        - num_empty : int - Number of empty cells.

    Notes
    -----
    This is designed for MCTS with progressive widening, where only a subset of
    possible outcomes are actually explored. Using lazy generation avoids creating
    states that will never be visited.
    """
    empty_cells = argwhere(state == 0)
    return state, [(int(c[0]), int(c[1])) for c in empty_cells], len(empty_cells)


def generate_outcome(state: ndarray, cell: tuple[int, int], value: int, num_empty: int) -> tuple[ndarray, float]:
    """
    Generate a single stochastic outcome on demand.

    Creates a copy of the state with a new tile placed at the specified cell.
    Use with ``after_state_lazy()`` to generate outcomes incrementally.

    Parameters
    ----------
    state : ndarray
        The base state (will be copied, not mutated).
    cell : Tuple[int, int]
        Position (row, col) where the tile will be placed.
    value : int
        Tile value to place (2 or 4).
    num_empty : int
        Total number of empty cells for probability calculation. Must be > 0.

    Returns
    -------
    tuple
        - new_state : ndarray - A new board state with the tile added.
        - probability : float - The probability of this outcome.

    Raises
    ------
    ValueError
        If num_empty <= 0 (no empty cells to place a tile).

    Notes
    -----
    - The original state is NOT modified.
    - Probability = P(value) / num_empty, where P(2)=0.9 and P(4)=0.1.
    """
    if num_empty <= 0:
        raise ValueError(f'num_empty must be > 0, got {num_empty}')

    new_state = state.copy()
    new_state[cell] = value
    return new_state, TILE_SPAWN_PROBS[value] / num_empty


def fill_cells(state: ndarray, number_tile: int, seed: int | None = None) -> ndarray:
    """
    Fill empty cells with new tiles (2 or 4).

    Parameters
    ----------
    state : ndarray
        The current state of the game board. **Modified in-place.**
    number_tile : int
        Number of new tiles to add.
    seed : int, optional
        Random number generator seed for reproducibility.

    Returns
    -------
    ndarray
        The same array reference with new tiles added.

    Notes
    -----
    - New tiles have a 90% chance of being 2 and a 10% chance of being 4.
    - If there are fewer empty cells than requested, it fills all available cells.
    - **This function mutates the input array.** Pass ``state.copy()`` if the original must be preserved.
    """
    # ##>: Use module-level generator for performance unless seed is specified.
    rng = default_rng(seed) if seed is not None else _GENERATOR

    # ##: Only if there are still available places.
    if not state.all():
        values = rng.choice(_TILE_VALUES, size=number_tile, p=_TILE_PROBS)

        # ##: Randomly choose cell positions in board.
        available_cells = argwhere(state == 0)
        chosen_indices = rng.choice(len(available_cells), size=number_tile, replace=False)

        # ##: Fill empty cells.
        state[tuple(available_cells[chosen_indices].T)] = values
    return state


def next_state(state: ndarray, action: int, seed: int | None = None) -> tuple[ndarray, float]:
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
    """
    return bool(
        np_all(state != 0) and not np_any(state[:-1] == state[1:]) and not np_any(state[:, :-1] == state[:, 1:])
    )
