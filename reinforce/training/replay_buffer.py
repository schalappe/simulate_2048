"""
Prioritized replay buffer for Stochastic MuZero (paper, Appendix C).

Finished games enter the buffer whole: value targets are computed once over the full game, then the game
is split into sequences of up to `sequence_length` steps. Each sequence also stores the next
`num_unroll_steps` steps, so an unroll from any of its positions reads real data, and absorbing targets
after the true end of a game. Boards are stored as uint8 tile exponents (16 bytes per position);
observations are encoded on the device.

Sampling follows the paper: priority p = |search value - value target| per position, P(i) ∝ p^α, and
importance weights (N · P(i))^-β over all N stored positions. Sampling is two-level (sequence by its
priority sum, then position), which gives the same distribution as flat sampling.
"""

import json
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.random import default_rng

from reinforce.training.config import TrainConfig
from reinforce.training.losses import TrainingBatch

# ##>: Keeps every stored position reachable when search value and target agree exactly.
MIN_PRIORITY = 1e-6


class Game(NamedTuple):
    """
    A finished self-play game.

    Attributes
    ----------
    boards : np.ndarray
        Tile exponents before each move plus the final board, shape (T + 1, 16), uint8.
    actions : np.ndarray
        Actions, shape (T,).
    rewards : np.ndarray
        Merge scores, shape (T,).
    policies : np.ndarray
        Root visit distributions, shape (T, num_actions).
    search_values : np.ndarray
        Raw root search values, shape (T,).
    terminal : bool
        True if the game ended with no legal move; False if it stopped at the safety limit.
    """

    boards: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    policies: np.ndarray
    search_values: np.ndarray
    terminal: bool

    def __len__(self) -> int:
        """Return the number of moves."""
        return len(self.actions)

    @property
    def score(self) -> float:
        """Game score: the sum of merge rewards."""
        return float(np.sum(self.rewards))

    @property
    def max_tile(self) -> int:
        """Largest tile on the final board."""
        return int(2 ** int(self.boards[-1].max())) if self.boards[-1].max() > 0 else 0


def td_lambda_targets(
    rewards: np.ndarray,
    search_values: np.ndarray,
    terminal: bool,
    discount: float,
    td_steps: int,
    td_lambda: float,
) -> np.ndarray:
    """
    Compute n-step TD(λ) value targets for a whole game.

    z_t = Σ_{k<n} (1 - λ) λ^(k-1) G_t^(k) + λ^(n-1) G_t^(n), where G_t^(k) sums k discounted rewards and
    bootstraps from the search value k steps later. Returns that run past the end of the game stop early:
    after a terminal state they end with value 0; in a game stopped at the safety limit the final board was
    never searched, so they bootstrap from the last searched state, T - 1.

    Parameters
    ----------
    rewards : np.ndarray
        Rewards, shape (T,).
    search_values : np.ndarray
        Raw search values, shape (T,).
    terminal : bool
        Whether the game ended with no legal move.
    discount : float
        Discount γ.
    td_steps : int
        Number of steps n.
    td_lambda : float
        λ.

    Returns
    -------
    np.ndarray
        Value targets, shape (T,), float32.
    """
    length = len(rewards)
    positions = np.arange(length)
    # ##>: Steps available before the bootstrap state: the end of the game, or the last searched state.
    remaining = length - positions if terminal else length - 1 - positions
    values_ext = np.append(search_values.astype(np.float64), 0.0)
    rewards_ext = np.concatenate([rewards.astype(np.float64), np.zeros(td_steps)])

    reward_sum = np.zeros(length)
    targets = np.zeros(length)
    for k in range(1, td_steps + 1):
        reward_sum += np.where(k <= remaining, discount ** (k - 1) * rewards_ext[positions + k - 1], 0.0)
        steps = np.minimum(k, remaining)
        n_step_return = reward_sum + discount**steps * values_ext[positions + steps]
        weight = (1 - td_lambda) * td_lambda ** (k - 1) if k < td_steps else td_lambda ** (k - 1)
        targets += weight * n_step_return
    return targets.astype(np.float32)


class ReplayBuffer:
    """
    Ring buffer of fixed-length sequences with prioritized sampling.

    Parameters
    ----------
    config : TrainConfig
        Training configuration (capacity, sequence length, unroll steps, TD and priority parameters).
    seed : int
        Seed for sampling.
    """

    def __init__(self, config: TrainConfig, seed: int = 0):
        self.config = config
        capacity = config.replay_buffer_size
        states = config.sequence_length + config.num_unroll_steps + 1
        transitions = states - 1

        self._boards = np.zeros((capacity, states, 16), dtype=np.uint8)
        self._policies = np.zeros((capacity, states, config.action_size), dtype=np.float32)
        self._values = np.zeros((capacity, states), dtype=np.float32)
        self._value_mask = np.zeros((capacity, states), dtype=bool)
        self._policy_mask = np.zeros((capacity, states), dtype=bool)
        self._actions = np.zeros((capacity, transitions), dtype=np.int8)
        self._rewards = np.zeros((capacity, transitions), dtype=np.float32)
        self._reward_mask = np.zeros((capacity, transitions), dtype=bool)
        self._transition_mask = np.zeros((capacity, transitions), dtype=bool)
        # ##>: p^α per start position; positions past the end of a sequence stay 0.
        self._priorities = np.zeros((capacity, config.sequence_length), dtype=np.float64)
        self._priority_sums = np.zeros(capacity, dtype=np.float64)
        self._lengths = np.zeros(capacity, dtype=np.int64)

        self._size = 0
        self._position = 0
        self.games_added = 0
        self._rng = default_rng(seed)

    def __len__(self) -> int:
        """Return the number of stored sequences."""
        return self._size

    @property
    def num_positions(self) -> int:
        """Return the number of stored start positions."""
        return int(self._lengths.sum())

    def is_ready(self) -> bool:
        """Check whether the buffer holds enough sequences to start training."""
        return self._size >= self.config.min_buffer_size

    def add_game(self, game: Game) -> None:
        """
        Compute targets for a finished game and store it as sequences.

        Parameters
        ----------
        game : Game
            The finished game. Must have at least one move.
        """
        config = self.config
        length = len(game)
        if length == 0:
            raise ValueError('Cannot store a game without moves')

        targets = td_lambda_targets(
            game.rewards, game.search_values, game.terminal, config.discount, config.td_steps, config.td_lambda
        )
        priorities = (np.abs(game.search_values - targets) + MIN_PRIORITY) ** config.priority_alpha

        # ##>: Extend every per-position array past the end with absorbing values, so slicing never runs short.
        pad = config.sequence_length + config.num_unroll_steps + 1
        terminal = bool(game.terminal)
        uniform_policy = np.full((pad + 1, config.action_size), 1.0 / config.action_size, dtype=np.float32)
        boards = np.concatenate([game.boards, np.zeros((pad, 16), dtype=np.uint8)])
        policies = np.concatenate([game.policies, uniform_policy])
        values = np.concatenate([targets, np.zeros(pad + 1, dtype=np.float32)])
        value_mask = np.concatenate([np.ones(length, dtype=bool), np.full(pad + 1, terminal)])
        policy_mask = np.concatenate([np.ones(length, dtype=bool), np.zeros(pad + 1, dtype=bool)])
        actions = np.concatenate([game.actions, np.zeros(pad, dtype=game.actions.dtype)])
        rewards = np.concatenate([game.rewards, np.zeros(pad, dtype=np.float32)])
        reward_mask = np.concatenate([np.ones(length, dtype=bool), np.full(pad, terminal)])
        transition_mask = np.concatenate([np.ones(length, dtype=bool), np.zeros(pad, dtype=bool)])

        states = self._boards.shape[1]
        for start in range(0, length, config.sequence_length):
            slot = self._position
            starts = min(config.sequence_length, length - start)
            self._boards[slot] = boards[start : start + states]
            self._policies[slot] = policies[start : start + states]
            self._values[slot] = values[start : start + states]
            self._value_mask[slot] = value_mask[start : start + states]
            self._policy_mask[slot] = policy_mask[start : start + states]
            self._actions[slot] = actions[start : start + states - 1]
            self._rewards[slot] = rewards[start : start + states - 1]
            self._reward_mask[slot] = reward_mask[start : start + states - 1]
            self._transition_mask[slot] = transition_mask[start : start + states - 1]
            self._priorities[slot] = 0.0
            self._priorities[slot, :starts] = priorities[start : start + starts]
            self._priority_sums[slot] = self._priorities[slot].sum()
            self._lengths[slot] = starts

            self._position = (self._position + 1) % config.replay_buffer_size
            self._size = min(self._size + 1, config.replay_buffer_size)
        self.games_added += 1

    def sample(self, batch_size: int) -> TrainingBatch:
        """
        Sample a training batch by priority.

        Parameters
        ----------
        batch_size : int
            Number of positions.

        Returns
        -------
        TrainingBatch
            Unroll windows of K + 1 states with targets, masks, and normalized importance weights.
        """
        if self._size == 0:
            raise RuntimeError('Cannot sample from an empty replay buffer')

        sequence_totals = self._priority_sums[: self._size]
        total = sequence_totals.sum()
        sequences = self._rng.choice(self._size, size=batch_size, p=sequence_totals / total)

        cumulative = np.cumsum(self._priorities[sequences], axis=1)
        draws = self._rng.random(batch_size) * cumulative[:, -1]
        starts = np.minimum((cumulative <= draws[:, None]).sum(axis=1), self._lengths[sequences] - 1)

        probabilities = self._priorities[sequences, starts] / total
        weights = (self.num_positions * probabilities) ** (-self.config.priority_beta)
        weights = weights / weights.max()

        unroll = self.config.num_unroll_steps
        state_index = starts[:, None] + np.arange(unroll + 1)
        transition_index = starts[:, None] + np.arange(unroll)
        rows = sequences[:, None]
        return TrainingBatch(
            boards=self._boards[rows, state_index],
            actions=self._actions[rows, transition_index].astype(np.int32),
            rewards=self._rewards[rows, transition_index],
            reward_mask=self._reward_mask[rows, transition_index].astype(np.float32),
            transition_mask=self._transition_mask[rows, transition_index].astype(np.float32),
            policies=self._policies[rows, state_index],
            policy_mask=self._policy_mask[rows, state_index].astype(np.float32),
            values=self._values[rows, state_index],
            value_mask=self._value_mask[rows, state_index].astype(np.float32),
            weights=weights.astype(np.float32),
        )

    def save(self, path: Path) -> None:
        """
        Save the buffer to one file, atomically.

        The data goes to a temporary file that replaces `path` only once fully written, so a crash in the
        middle of a save leaves the previous file intact.

        Parameters
        ----------
        path : Path
            Destination `.npz` file.
        """
        temporary = path.with_name(path.name + '.tmp')
        meta = {
            'size': self._size,
            'position': self._position,
            'games_added': self.games_added,
            'rng_state': self._rng.bit_generator.state,
        }
        with temporary.open('wb') as file:
            # ##&: pyrefly matches the ** arrays against savez's allow_pickle keyword; the names never collide.
            np.savez(file, meta=np.array(json.dumps(meta)), **self._arrays())  # pyrefly: ignore[bad-argument-type]
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)

    def load(self, path: Path) -> None:
        """
        Restore a buffer written by `save` with the same configuration.

        Parameters
        ----------
        path : Path
            File written by `save`.

        Raises
        ------
        ValueError
            If the stored arrays do not match this buffer's configuration.
        """
        with np.load(path) as data:
            arrays = self._arrays()
            for name, array in arrays.items():
                stored = data[name]
                if stored.shape != array.shape or stored.dtype != array.dtype:
                    raise ValueError(
                        f'Replay buffer file {path} does not match the configuration: {name} has shape '
                        f'{stored.shape} {stored.dtype}, expected {array.shape} {array.dtype}'
                    )
                array[...] = stored
            meta = json.loads(str(data['meta']))
        self._size = meta['size']
        self._position = meta['position']
        self.games_added = meta['games_added']
        self._rng.bit_generator.state = meta['rng_state']

    def _arrays(self) -> dict[str, np.ndarray]:
        """Return the stored arrays by name."""
        return {
            'boards': self._boards,
            'policies': self._policies,
            'values': self._values,
            'value_mask': self._value_mask,
            'policy_mask': self._policy_mask,
            'actions': self._actions,
            'rewards': self._rewards,
            'reward_mask': self._reward_mask,
            'transition_mask': self._transition_mask,
            'priorities': self._priorities,
            'priority_sums': self._priority_sums,
            'lengths': self._lengths,
        }
