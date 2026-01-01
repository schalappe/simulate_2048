# -*- coding: utf-8 -*-
"""
Encoded wrapper for the 2048 game environment.
"""
from typing import Optional, Tuple

from numpy import int64, ndarray, zeros

from twentyfortyeight.core import fill_cells, is_done, next_state
from twentyfortyeight.utils import encode_flatten


class EncodedTwentyFortyEight:
    """
    Wrapper for the 2048 game that returns binary-encoded observations.

    This wrapper encodes the board state into a flattened binary representation,
    where each cell value is one-hot encoded with a configurable encoding size.

    Parameters
    ----------
    size : int
        The size of the square grid.
    block_size : int
        The encoding size for each cell (number of bits per cell).
    """

    ACTIONS = {"left": 0, "up": 1, "right": 2, "down": 3}

    def __init__(self, size: int = 4, block_size: int = 31):
        self.size = size
        self.block_size = block_size
        self._board: ndarray = zeros((size, size), dtype=int64)
        self._reward: float = 0.0
        self.reset()

    @property
    def is_finished(self) -> bool:
        """Check if the game is finished."""
        return is_done(self._board)

    @property
    def observation(self) -> ndarray:
        """
        Get the binary-encoded observation.

        Returns
        -------
        ndarray
            Flattened binary-encoded board of shape (size * size * block_size,).
        """
        return encode_flatten(self._board, encodage_size=self.block_size)

    def reset(self, seed: Optional[int] = None) -> ndarray:
        """
        Reset the game to initial state.

        Returns
        -------
        ndarray
            The encoded initial board state.
        """
        self._board = zeros((self.size, self.size), dtype=int64)
        self._board = fill_cells(state=self._board, number_tile=2, seed=seed)
        self._reward = 0.0
        return self.observation

    def step(self, action: int) -> Tuple[ndarray, float, bool]:
        """
        Apply an action and return the new encoded state.

        Parameters
        ----------
        action : int
            The action to apply (0: left, 1: up, 2: right, 3: down).

        Returns
        -------
        Tuple[ndarray, float, bool]
            The encoded observation, reward, and done flag.
        """
        self._board, self._reward = next_state(state=self._board, action=action)
        return self.observation, self._reward, self.is_finished
