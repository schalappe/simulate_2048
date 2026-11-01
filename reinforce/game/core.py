"""
JAX-native core game logic for 2048.

This module provides pure JAX implementations of all 2048 game operations. All functions are
JIT-compilable and can be vectorized with vmap for massive parallelization on GPU.

Key design principles:
1. No Python control flow on traced values (use lax.cond, lax.scan, etc.)
2. All functions are stateless and pure
3. Explicit random state management via PRNGKey
"""

import jax
import jax.numpy as jnp
from jax import lax

# ##>: Type aliases for clarity.
Array = jax.Array
PRNGKey = jax.Array

# ##>: Tile spawn probabilities (90% for 2, 10% for 4).
TILE_PROBS = jnp.array([0.9, 0.1])
TILE_VALUES = jnp.array([2, 4])

# ##>: Board dimensions.
BOARD_SIZE = 4
NUM_ACTIONS = 4

# ##>: Binary observation from the paper (Appendix C): 31 bits per tile, 16 tiles.
TILE_BITS = 31
OBSERVATION_SIZE = BOARD_SIZE * BOARD_SIZE * TILE_BITS


def _compact_row(row: Array) -> Array:
    """Compact non-zero elements to the left."""
    # ##>: Create a sorting key: non-zero elements get their index, zero gets max.
    n = row.shape[0]
    is_nonzero = row != 0
    # ##>: Non-zero elements keep position, zeros go to end.
    sort_key = jnp.where(is_nonzero, jnp.arange(n), n)
    sorted_indices = jnp.argsort(sort_key)
    return row[sorted_indices]


@jax.jit
def merge_row(row: Array) -> tuple[Array, Array]:
    """
    Merge a single row to the left.

    This implements the 2048 merge logic: slide non-zero tiles left, then merge adjacent
    equal tiles, then slide again.

    Parameters
    ----------
    row : Array
        A 1D array of shape (4,) representing one row.

    Returns
    -------
    tuple[Array, Array]
        - merged_row: The row after merging, shape (4,)
        - score: The score from this merge (sum of merged tile values)

    Notes
    -----
    Uses a functional approach compatible with JAX tracing.
    """
    # ##>: Step 1: Compact non-zero values to the left.
    compacted = _compact_row(row)

    # ##>: Step 2: Merge adjacent equal tiles.
    # ##>: Process each pair: (0,1), (1,2), (2,3) but only merge once.
    def merge_pair(carry, idx):
        arr, score, merged_prev = carry

        curr = arr[idx]
        next_val = arr[idx + 1]

        # ##>: Merge if: current == next, both non-zero, and previous wasn't merged.
        can_merge = (curr == next_val) & (curr != 0) & (~merged_prev)

        # ##>: Update array: double current, zero next if merging.
        new_curr = jnp.where(can_merge, curr * 2, curr)
        new_next = jnp.where(can_merge, 0, next_val)
        new_arr = arr.at[idx].set(new_curr)
        new_arr = new_arr.at[idx + 1].set(new_next)

        # ##>: Update score.
        merge_score = jnp.where(can_merge, curr * 2, 0)

        return (new_arr, score + merge_score, can_merge), None

    # ##>: Process indices 0, 1, 2 (pairs: 0-1, 1-2, 2-3).
    (merged, total_score, _), _ = lax.scan(
        merge_pair,
        (compacted, jnp.int32(0), jnp.bool_(False)),
        jnp.arange(3),
    )

    # ##>: Step 3: Compact again after merging.
    final = _compact_row(merged)

    return final, total_score


@jax.jit
def slide_and_merge(board: Array) -> tuple[Array, Array]:
    """
    Slide the entire board left and merge tiles.

    Parameters
    ----------
    board : Array
        A 2D array of shape (4, 4) representing the game board.

    Returns
    -------
    tuple[Array, Array]
        - new_board: The board after sliding and merging
        - score: Total score from all merges

    Notes
    -----
    Uses vmap to process all 4 rows in parallel.
    """
    # ##>: Apply merge_row to each row in parallel.
    new_board, scores = jax.vmap(merge_row)(board)
    return new_board, jnp.sum(scores)


@jax.jit
def latent_state(state: Array, action: int) -> tuple[Array, Array]:
    """
    Compute the next state after applying an action, without adding a new tile.

    This is the deterministic part of the transition - the "afterstate" before
    the stochastic tile spawn.

    Parameters
    ----------
    state : Array
        The current state of the game board, shape (4, 4).
    action : int
        The action to apply (0: left, 1: up, 2: right, 3: down).

    Returns
    -------
    tuple[Array, Array]
        - new_state: The board after applying the action
        - reward: The score from merges

    Notes
    -----
    Uses rotation to convert all directions to "slide left" operation.
    """

    # ##>: Apply each possible action and select based on action index.
    def apply_action_k(k: int) -> tuple[Array, Array]:
        rotated = jnp.rot90(state, k=k)
        updated, reward = slide_and_merge(rotated)
        return jnp.rot90(updated, k=-k), reward

    # ##>: Compute all 4 possible results.
    results = [apply_action_k(k) for k in range(4)]
    all_states = jnp.stack([r[0] for r in results], axis=0)
    all_rewards = jnp.stack([r[1] for r in results], axis=0)

    # ##>: Select based on action.
    new_state = all_states[action]
    reward = all_rewards[action]

    return new_state, reward


@jax.jit
def fill_cells(state: Array, key: PRNGKey) -> Array:
    """
    Fill empty cells with new tiles (2 or 4).

    Parameters
    ----------
    state : Array
        The current state of the game board, shape (4, 4).
    key : PRNGKey
        JAX random key for stochastic tile selection.

    Returns
    -------
    Array
        The board with new tiles added.

    Notes
    -----
    - 90% chance of spawning a 2, 10% chance of spawning a 4.
    - If the board is full, returns unchanged.
    """
    # ##>: Find empty cells.
    empty_mask = state == 0
    num_empty = jnp.sum(empty_mask)

    # ##>: Handle case where board is full.
    def fill_tiles(args: tuple[Array, PRNGKey]) -> Array:
        board, rng = args
        k1, k2 = jax.random.split(rng)

        # ##>: Sample which empty cell to fill.
        # ##>: Convert 2D mask to 1D indices.
        flat_mask = empty_mask.flatten()
        cumsum = jnp.cumsum(flat_mask)

        # ##>: Sample position among empty cells.
        pos_idx = jax.random.randint(k1, (), 0, num_empty)

        # ##>: Find the actual flat index.
        flat_idx = jnp.argmax(cumsum > pos_idx)

        # ##>: Sample tile value (2 or 4).
        tile_val = jax.random.choice(k2, TILE_VALUES, p=TILE_PROBS)

        # ##>: Place tile.
        flat_board = board.flatten()
        flat_board = flat_board.at[flat_idx].set(tile_val)

        return flat_board.reshape(4, 4)

    def no_fill(args: tuple[Array, PRNGKey]) -> Array:
        board, _ = args
        return board

    # ##>: Only fill if there are empty cells.
    new_state = lax.cond(num_empty > 0, fill_tiles, no_fill, (state, key))

    return new_state


@jax.jit
def next_state(state: Array, action: int, key: PRNGKey) -> tuple[Array, Array]:
    """
    Compute the full transition: apply action and add random tile.

    Parameters
    ----------
    state : Array
        The current state of the game board, shape (4, 4).
    action : int
        The action to apply (0: left, 1: up, 2: right, 3: down).
    key : PRNGKey
        JAX random key for tile spawning.

    Returns
    -------
    tuple[Array, Array]
        - new_state: The board after action and tile spawn
        - reward: The score from merges
    """
    # ##>: Apply action.
    after_action, reward = latent_state(state, action)

    # ##>: Check if the board changed (valid move).
    board_changed = jnp.any(after_action != state)

    # ##>: Only add tile if the move was valid.
    def add_tile(s: Array) -> Array:
        return fill_cells(s, key)

    def keep_state(s: Array) -> Array:
        return s

    new_state = lax.cond(board_changed, add_tile, keep_state, after_action)

    # ##>: Reward is 0 if move was invalid. Cast to float32 for consistency.
    reward = jnp.where(board_changed, reward.astype(jnp.float32), jnp.float32(0.0))

    return new_state, reward


@jax.jit
def legal_actions_mask(state: Array) -> Array:
    """
    Get a boolean mask indicating which actions are legal.

    Parameters
    ----------
    state : Array
        The current state of the game board, shape (4, 4).

    Returns
    -------
    Array
        Boolean array of shape (4,) where True means the action is legal.
        Order: [left, up, right, down]

    Notes
    -----
    An action is legal if it would change the board state.
    """

    # ##>: Check each direction by applying the action and comparing.
    def check_action(action: Array) -> Array:
        after, _ = latent_state(state, action)
        return jnp.any(after != state)

    # ##>: Vectorize over all 4 actions.
    return jax.vmap(check_action)(jnp.arange(4))


@jax.jit
def is_done(state: Array) -> Array:
    """
    Check if the game is over (no legal moves).

    Parameters
    ----------
    state : Array
        The current state of the game board, shape (4, 4).

    Returns
    -------
    Array
        Boolean scalar, True if game is over.
    """
    mask = legal_actions_mask(state)
    return ~jnp.any(mask)


@jax.jit
def create_initial_board(key: PRNGKey) -> Array:
    """
    Create a new game board with 2 random tiles.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.

    Returns
    -------
    Array
        A new 4x4 board with 2 tiles placed.
    """
    k1, k2 = jax.random.split(key)
    board = jnp.zeros((4, 4), dtype=jnp.int32)
    board = fill_cells(board, k1)
    board = fill_cells(board, k2)
    return board


@jax.jit
def board_to_exponents(board: Array) -> Array:
    """
    Compress a board to tile exponents for storage.

    Parameters
    ----------
    board : Array
        Game boards with tile values, shape (..., 4, 4).

    Returns
    -------
    Array
        Flattened exponents as uint8, shape (..., 16). Empty cells are 0 and tile 2^k is k.
    """
    exponents = jnp.where(board > 0, jnp.log2(jnp.maximum(board, 1).astype(jnp.float32)), 0.0)
    flat = jnp.round(exponents).astype(jnp.uint8)
    return flat.reshape(*board.shape[:-2], BOARD_SIZE * BOARD_SIZE)


@jax.jit
def encode_exponents(exponents: Array) -> Array:
    """
    Encode stored tile exponents as the binary network input of the paper (Appendix C).

    Each tile becomes 31 bits where bit k is set for tile 2^k. Empty cells are all zeros.

    Parameters
    ----------
    exponents : Array
        Tile exponents, shape (..., 16).

    Returns
    -------
    Array
        Observation as float32, shape (..., 496).
    """
    exponents = exponents.astype(jnp.int32)
    bits = jax.nn.one_hot(exponents, TILE_BITS, dtype=jnp.float32) * (exponents > 0)[..., None]
    return bits.reshape(*exponents.shape[:-1], OBSERVATION_SIZE)


@jax.jit
def encode_observation(board: Array) -> Array:
    """
    Encode boards for neural network input.

    Parameters
    ----------
    board : Array
        Game boards with tile values, shape (..., 4, 4).

    Returns
    -------
    Array
        Observation as float32, shape (..., 496). See `encode_exponents`.
    """
    return encode_exponents(board_to_exponents(board))
