"""
Self-play compiled on the device.

`play_moves` plays a fixed number of moves for a batch of games in one jitted `lax.scan`: encode, search,
select, step, and restart games that ended, with no host round-trip per move. Games run to their real end;
`max_game_moves` only stops runaway games and marks them truncated. `GameCollector` stitches the returned
moves into finished `Game`s on the host. Evaluation uses the same path with temperature 0 and no root noise.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from reinforce.game.core import (
    board_to_exponents,
    create_initial_board,
    encode_observation,
    is_done,
    legal_actions_mask,
    next_state,
)
from reinforce.mcts.policy import select_action
from reinforce.mcts.stochastic_mctx import run_mcts
from reinforce.neural.network import NetworkParams
from reinforce.training.config import TrainConfig
from reinforce.training.learner import apply_fns_for
from reinforce.training.replay_buffer import Game

# ##>: Type aliases. Move records live on the device, then on the host once collected.
Array = jax.Array
PRNGKey = jax.Array
RecordArray = jax.Array | np.ndarray


class ActorState(NamedTuple):
    """
    Games in progress.

    Attributes
    ----------
    boards : Array
        Current boards, shape (num_games, 4, 4).
    moves : Array
        Moves played in each current game, shape (num_games,).
    """

    boards: Array
    moves: Array


class MoveRecord(NamedTuple):
    """
    One move of every game. `play_moves` returns these stacked with a leading time axis.

    Attributes
    ----------
    boards : Array
        Exponents of the board before the move, shape (num_games, 16), uint8.
    next_boards : Array
        Exponents of the board after the move and tile spawn, before any restart, shape (num_games, 16).
    actions : Array
        Actions, shape (num_games,).
    rewards : Array
        Merge scores, shape (num_games,).
    policies : Array
        Root visit distributions, shape (num_games, num_actions).
    search_values : Array
        Raw root search values, shape (num_games,).
    terminal : Array
        True if the game ended after this move, shape (num_games,).
    truncated : Array
        True if the game hit the safety limit after this move, shape (num_games,).
    """

    boards: RecordArray
    next_boards: RecordArray
    actions: RecordArray
    rewards: RecordArray
    policies: RecordArray
    search_values: RecordArray
    terminal: RecordArray
    truncated: RecordArray


def new_actors(key: PRNGKey, num_games: int) -> ActorState:
    """
    Start a batch of new games.

    Parameters
    ----------
    key : PRNGKey
        Random key for the initial tiles.
    num_games : int
        Number of games.

    Returns
    -------
    ActorState
        Fresh games at move 0.
    """
    boards = jax.vmap(create_initial_board)(jax.random.split(key, num_games))
    return ActorState(boards=boards, moves=jnp.zeros(num_games, dtype=jnp.int32))


@partial(jax.jit, static_argnames=('config', 'num_moves', 'explore'))
def play_moves(
    params: NetworkParams,
    actors: ActorState,
    key: PRNGKey,
    temperature: Array | float,
    config: TrainConfig,
    num_moves: int,
    explore: bool = True,
) -> tuple[ActorState, MoveRecord]:
    """
    Play `num_moves` moves in every game, restarting games as they end.

    Parameters
    ----------
    params : NetworkParams
        Network parameters.
    actors : ActorState
        Games in progress.
    key : PRNGKey
        Random key.
    temperature : Array | float
        Action selection temperature (traced: changing it does not recompile).
    config : TrainConfig
        Training configuration (static).
    num_moves : int
        Moves per game in this call (static).
    explore : bool
        If False, no Dirichlet noise at the root (evaluation).

    Returns
    -------
    tuple[ActorState, MoveRecord]
        - Games in progress after the chunk
        - Moves, each field with shape (num_moves, num_games, ...)
    """
    apply_fns = apply_fns_for(config)
    num_games = actors.boards.shape[0]
    dirichlet_fraction = config.dirichlet_fraction if explore else 0.0

    def one_move(state: ActorState, move_key: PRNGKey) -> tuple[ActorState, MoveRecord]:
        search_key, action_key, spawn_key, restart_key = jax.random.split(move_key, 4)
        legal = jax.vmap(legal_actions_mask)(state.boards)
        output = run_mcts(
            params,
            apply_fns,
            encode_observation(state.boards),
            legal,
            search_key,
            codebook_size=config.codebook_size,
            num_simulations=config.num_simulations,
            discount=config.discount,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_fraction=dirichlet_fraction,
            pb_c_init=config.pb_c_init,
            pb_c_base=config.pb_c_base,
        )
        action_weights = jnp.asarray(output.action_weights)
        actions = select_action(action_key, action_weights, temperature)
        boards, rewards = jax.vmap(next_state)(state.boards, actions, jax.random.split(spawn_key, num_games))
        moves = state.moves + 1
        terminal = jax.vmap(is_done)(boards)
        truncated = (moves >= config.max_game_moves) & ~terminal
        ended = terminal | truncated

        restarted = jax.vmap(create_initial_board)(jax.random.split(restart_key, num_games))
        next_actors = ActorState(
            boards=jnp.where(ended[:, None, None], restarted, boards),
            moves=jnp.where(ended, 0, moves),
        )
        record = MoveRecord(
            boards=board_to_exponents(state.boards),
            next_boards=board_to_exponents(boards),
            actions=actions,
            rewards=rewards,
            policies=action_weights,
            search_values=jnp.asarray(output.search_tree.node_values)[:, 0],
            terminal=terminal,
            truncated=truncated,
        )
        return next_actors, record

    return jax.lax.scan(one_move, actors, jax.random.split(key, num_moves))


class GameCollector:
    """
    Stitch chunks of moves from `play_moves` into finished games.

    Parameters
    ----------
    num_games : int
        Number of games played side by side.
    """

    def __init__(self, num_games: int):
        self._pending: list[list[MoveRecord]] = [[] for _ in range(num_games)]

    def add(self, records: MoveRecord) -> list[tuple[int, Game]]:
        """
        Add one chunk and return the games that ended in it.

        Parameters
        ----------
        records : MoveRecord
            Output of `play_moves`, fields shaped (num_moves, num_games, ...).

        Returns
        -------
        list[tuple[int, Game]]
            (game slot, game) for each game that ended during this chunk.
        """
        host = MoveRecord(*(np.asarray(field) for field in records))
        ended = host.terminal | host.truncated
        finished = []
        for slot, pending in enumerate(self._pending):
            segment_start = 0
            for end in np.flatnonzero(ended[:, slot]).tolist():
                pending.append(_slice(host, slot, segment_start, end + 1))
                finished.append((slot, _to_game(pending)))
                pending.clear()
                segment_start = end + 1
            if segment_start < ended.shape[0]:
                pending.append(_slice(host, slot, segment_start, ended.shape[0]))
        return finished


def _slice(records: MoveRecord, slot: int, start: int, stop: int) -> MoveRecord:
    """Take moves [start, stop) of one game slot."""
    return MoveRecord(*(field[start:stop, slot] for field in records))


def _to_game(segments: list[MoveRecord]) -> Game:
    """Concatenate the segments of one game, which ends at the last move of the last segment."""
    boards, next_boards, actions, rewards, policies, search_values, terminal, _ = (
        np.concatenate(fields) for fields in zip(*segments, strict=True)
    )
    return Game(
        boards=np.concatenate([boards, next_boards[-1:]]),
        actions=actions.astype(np.int8),
        rewards=rewards.astype(np.float32),
        policies=policies.astype(np.float32),
        search_values=search_values.astype(np.float32),
        terminal=bool(terminal[-1]),
    )


def evaluate_games(params: NetworkParams, key: PRNGKey, config: TrainConfig, num_games: int) -> list[Game]:
    """
    Play complete greedy games without root noise.

    Parameters
    ----------
    params : NetworkParams
        Network parameters.
    key : PRNGKey
        Random key.
    config : TrainConfig
        Training configuration.
    num_games : int
        Number of games, played side by side.

    Returns
    -------
    list[Game]
        One finished game per slot.

    Raises
    ------
    ValueError
        If `num_games` is below 1.
    """
    if num_games < 1:
        raise ValueError(f'num_games must be at least 1, got {num_games}')
    key, start_key = jax.random.split(key)
    actors = new_actors(start_key, num_games)
    collector = GameCollector(num_games)
    first_games: dict[int, Game] = {}
    while len(first_games) < num_games:
        key, chunk_key = jax.random.split(key)
        actors, records = play_moves(
            params, actors, chunk_key, 0.0, config, num_moves=config.moves_per_generation, explore=False
        )
        for slot, game in collector.add(records):
            first_games.setdefault(slot, game)
    return [first_games[slot] for slot in range(num_games)]


def summarize_games(games: list[Game]) -> dict:
    """
    Summarize finished games for logs.

    Parameters
    ----------
    games : list[Game]
        Finished games (at least one).

    Returns
    -------
    dict
        Mean, max, and min score, mean and best max tile, mean length, truncated games, and how many
        games reached each tile from 2048 to 32768.
    """
    scores = [game.score for game in games]
    tiles = [game.max_tile for game in games]
    summary = {
        'games': len(games),
        'mean_score': float(np.mean(scores)),
        'max_score': float(np.max(scores)),
        'min_score': float(np.min(scores)),
        'mean_max_tile': float(np.mean(tiles)),
        'max_tile': int(np.max(tiles)),
        'mean_length': float(np.mean([len(game) for game in games])),
        'truncated': sum(1 for game in games if not game.terminal),
    }
    for tile in (2048, 4096, 8192, 16384, 32768):
        summary[f'reached_{tile}'] = sum(1 for value in tiles if value >= tile)
    return summary
