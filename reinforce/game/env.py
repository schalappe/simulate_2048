"""
JAX-native environment wrapper for 2048.

Provides a stateless environment interface following the functional programming paradigm required
by JAX. All state is explicit and passed through functions.

The environment is designed to be:
1. Fully JIT-compilable for fast execution
2. Vectorizable via vmap for parallel game simulation
3. Compatible with JAX's functional random state management
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from reinforce.game.core import (
    create_initial_board,
    encode_observation,
    is_done,
    legal_actions_mask,
    max_tile,
    next_state,
)

# ##>: Type aliases.
Array = jax.Array
PRNGKey = jax.Array


class GameState(NamedTuple):
    """
    Immutable game state container.

    Attributes
    ----------
    board : Array
        The 4x4 game board, shape (4, 4), dtype int32.
    step_count : Array
        Number of steps taken in this episode, scalar int32.
    done : Array
        Whether the game is over, scalar bool.
    total_reward : Array
        Cumulative reward in this episode, scalar float32.
    """

    board: Array
    step_count: Array
    done: Array
    total_reward: Array


@jax.jit
def reset(key: PRNGKey) -> GameState:
    """
    Reset the environment to a new game.

    Parameters
    ----------
    key : PRNGKey
        JAX random key for initial tile placement.

    Returns
    -------
    GameState
        Initial game state with 2 random tiles.
    """
    board = create_initial_board(key)
    return GameState(
        board=board,
        step_count=jnp.array(0, dtype=jnp.int32),
        done=jnp.array(False),
        total_reward=jnp.array(0.0, dtype=jnp.float32),
    )


@jax.jit
def step(state: GameState, action: int, key: PRNGKey) -> tuple[GameState, Array, Array, dict]:
    """
    Take a step in the environment.

    Parameters
    ----------
    state : GameState
        Current game state.
    action : int
        Action to take (0: left, 1: up, 2: right, 3: down).
    key : PRNGKey
        JAX random key for tile spawning.

    Returns
    -------
    tuple[GameState, Array, Array, dict]
        - next_state: New game state after the action
        - reward: Reward from this step
        - done: Whether the game is over
        - info: Dictionary with additional information
    """

    # ##>: Handle already-done games.
    def done_step() -> tuple[GameState, Array, Array]:
        return state, jnp.array(0.0), jnp.array(True)

    def active_step() -> tuple[GameState, Array, Array]:
        # ##>: Apply action.
        new_board, reward = next_state(state.board, action, key)

        # ##>: Check if game is over.
        game_done = is_done(new_board)

        # ##>: Create new state.
        new_state = GameState(
            board=new_board,
            step_count=state.step_count + 1,
            done=game_done,
            total_reward=state.total_reward + reward,
        )

        return new_state, reward, game_done

    # ##>: Branch based on done status.
    new_state, reward, done = lax.cond(state.done, done_step, active_step)

    # ##>: Compute info dict.
    info = {
        'max_tile': max_tile(new_state.board),
        'step_count': new_state.step_count,
        'total_reward': new_state.total_reward,
    }

    return new_state, reward, done, info


@jax.jit
def get_observation(state: GameState) -> Array:
    """
    Get the encoded observation from a game state.

    Parameters
    ----------
    state : GameState
        Current game state.

    Returns
    -------
    Array
        Encoded observation, shape (16,).
    """
    return encode_observation(state.board)


@jax.jit
def get_legal_actions(state: GameState) -> Array:
    """
    Get the mask of legal actions for the current state.

    Parameters
    ----------
    state : GameState
        Current game state.

    Returns
    -------
    Array
        Boolean mask of shape (4,), True for legal actions.
    """
    return legal_actions_mask(state.board)


# ##>: Vectorized versions for parallel game simulation.
# ##>: These can run hundreds of games simultaneously on GPU.


def batched_reset(keys: Array) -> GameState:
    """
    Reset multiple environments in parallel.

    Parameters
    ----------
    keys : Array
        Array of PRNGKeys, shape (num_envs, 2).

    Returns
    -------
    GameState
        Batched game states with shape (num_envs, ...) for each field.
    """
    return jax.vmap(reset)(keys)


def batched_step(states: GameState, actions: Array, keys: Array) -> tuple[GameState, Array, Array, dict]:
    """
    Step multiple environments in parallel.

    Parameters
    ----------
    states : GameState
        Batched game states.
    actions : Array
        Array of actions, shape (num_envs,).
    keys : Array
        Array of PRNGKeys, shape (num_envs, 2).

    Returns
    -------
    tuple[GameState, Array, Array, dict]
        Batched results from each environment.
    """
    return jax.vmap(step)(states, actions, keys)


def batched_get_observation(states: GameState) -> Array:
    """
    Get observations from multiple game states.

    Parameters
    ----------
    states : GameState
        Batched game states.

    Returns
    -------
    Array
        Batched observations, shape (num_envs, 16).
    """
    return jax.vmap(get_observation)(states)


def batched_get_legal_actions(states: GameState) -> Array:
    """
    Get legal action masks from multiple game states.

    Parameters
    ----------
    states : GameState
        Batched game states.

    Returns
    -------
    Array
        Batched legal action masks, shape (num_envs, 4).
    """
    return jax.vmap(get_legal_actions)(states)


class Environment:
    """
    Stateful wrapper for easier interactive use.

    This class provides a more traditional OOP interface around the
    functional JAX environment. Useful for testing and debugging,
    but the pure functions should be used for training.

    Attributes
    ----------
    state : GameState
        Current game state.
    key : PRNGKey
        Current random key.
    """

    def __init__(self, seed: int = 0):
        """
        Initialize the environment.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        """
        self.key = jax.random.PRNGKey(seed)
        self.state = self.reset_env()

    def reset_env(self) -> GameState:
        """Reset to a new game."""
        self.key, subkey = jax.random.split(self.key)
        self.state = reset(subkey)
        return self.state

    def step_env(self, action: int) -> tuple[GameState, float, bool, dict]:
        """Take a step in the environment."""
        self.key, subkey = jax.random.split(self.key)
        self.state, reward, done, info = step(self.state, action, subkey)
        return self.state, float(reward), bool(done), info

    def observation(self) -> Array:
        """Get current observation."""
        return get_observation(self.state)

    def legal_actions(self) -> Array:
        """Get legal action mask."""
        return get_legal_actions(self.state)

    @property
    def board(self) -> Array:
        """Get current board."""
        return self.state.board

    @property
    def done(self) -> bool:
        """Check if game is over."""
        return bool(self.state.done)
