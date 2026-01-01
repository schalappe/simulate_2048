"""
Game move utilities for the 2048 game simulator, providing functions for determining legal
and illegal moves.
"""

from numpy import ndarray


def _can_move_direction(board: ndarray, direction: int) -> bool:
    """
    Check if a move is possible in a specific direction without rotation.

    Parameters
    ----------
    board : ndarray
        The game board to check.
    direction : int
        Direction to check (0: left, 1: up, 2: right, 3: down).

    Returns
    -------
    bool
        True if the move is possible, False otherwise.
    """
    if direction == 0:  # Left: empty cell left of non-empty, or equal adjacent
        left_cols, right_cols = board[:, :-1], board[:, 1:]
        can_slide = (left_cols == 0) & (right_cols != 0)
        can_merge = (left_cols != 0) & (left_cols == right_cols)
    elif direction == 1:  # Up: empty cell above non-empty, or equal adjacent
        top_rows, bottom_rows = board[:-1, :], board[1:, :]
        can_slide = (top_rows == 0) & (bottom_rows != 0)
        can_merge = (top_rows != 0) & (top_rows == bottom_rows)
    elif direction == 2:  # Right: empty cell right of non-empty, or equal adjacent
        left_cols, right_cols = board[:, :-1], board[:, 1:]
        can_slide = (right_cols == 0) & (left_cols != 0)
        can_merge = (right_cols != 0) & (left_cols == right_cols)
    else:  # Down: empty cell below non-empty, or equal adjacent
        top_rows, bottom_rows = board[:-1, :], board[1:, :]
        can_slide = (bottom_rows == 0) & (top_rows != 0)
        can_merge = (bottom_rows != 0) & (top_rows == bottom_rows)

    return bool(can_slide.any() or can_merge.any())


def legal_actions_mask(state: ndarray) -> tuple[bool, bool, bool, bool]:
    """
    Get a boolean mask for all four directions in a single pass.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.

    Returns
    -------
    tuple[bool, bool, bool, bool]
        Mask for (left, up, right, down) where True means action is legal.

    Notes
    -----
    Optimized to compute horizontal and vertical adjacencies only once,
    then derive all four directions from them.
    """
    # ##>: Compute horizontal adjacency once for left/right.
    left_cols, right_cols = state[:, :-1], state[:, 1:]
    h_can_merge = (left_cols != 0) & (left_cols == right_cols)

    # ##>: Compute vertical adjacency once for up/down.
    top_rows, bottom_rows = state[:-1, :], state[1:, :]
    v_can_merge = (top_rows != 0) & (top_rows == bottom_rows)

    # ##>: Check slide conditions per direction.
    left = (left_cols == 0) & (right_cols != 0)
    right = (right_cols == 0) & (left_cols != 0)
    up = (top_rows == 0) & (bottom_rows != 0)
    down = (bottom_rows == 0) & (top_rows != 0)

    return (
        bool(left.any() or h_can_merge.any()),
        bool(up.any() or v_can_merge.any()),
        bool(right.any() or h_can_merge.any()),
        bool(down.any() or v_can_merge.any()),
    )


def illegal_actions(state: ndarray) -> list[int]:
    """
    Determine illegal actions for the current game board state.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.

    Returns
    -------
    list[int]
        A list of illegal actions (0: left, 1: up, 2: right, 3: down).

    Notes
    -----
    - An action is considered illegal if it doesn't change the board state.
    - Uses vectorized operations to check all directions without rotations.
    """
    mask = legal_actions_mask(state)
    return [i for i in range(4) if not mask[i]]


def legal_actions(state: ndarray) -> list[int]:
    """
    Determine legal actions for the current game board state.

    Parameters
    ----------
    state : ndarray
        The current state of the game board.

    Returns
    -------
    list[int]
        A list of legal actions (0: left, 1: up, 2: right, 3: down).

    Notes
    -----
    - Legal actions are those that result in a change to the game board.
    - Uses vectorized operations to check all directions without rotations.
    """
    mask = legal_actions_mask(state)
    return [i for i in range(4) if mask[i]]


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
    # ##>: Compare all adjacent horizontal pairs with vectorized operations.
    left_cols = board[:, :-1]
    right_cols = board[:, 1:]

    # ##>: Condition 1: Empty cell left of non-empty cell (can slide).
    can_slide = (left_cols == 0) & (right_cols != 0)
    if can_slide.any():
        return True

    # ##>: Condition 2: Two adjacent equal non-zero values (can merge).
    can_merge = (left_cols != 0) & (left_cols == right_cols)
    return bool(can_merge.any())
