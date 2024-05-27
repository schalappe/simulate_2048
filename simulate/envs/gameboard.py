# -*- coding: utf-8 -*-
"""
Class describing the 2048 game for an agent.
"""
from typing import Optional, Tuple, Union

from numpy import argwhere, array_equal, int64, ndarray, rot90, zeros
from numpy.random import Generator, default_rng

from .utils import compute_penalties, slide_and_merge


class GameBoard:
    """2048 game environment."""

    # ##: All Actions.
    ACTIONS = {"left": 0, "up": 1, "right": 2, "down": 3}

    # ##: Game variables.
    _board: Optional[ndarray] = None
    _generator: Optional[Generator] = None

    def __init__(self, size: int = 4):
        self._size = size  # ##: The size of the square grid.

        # ##: Reset game.
        self.reset()

    def _fill_cells(self, number_tile: int) -> None:
        """
        Fill empty cells with 2 or 4.

        Parameters
        ----------
        number_tile : int
            Number of cells to fill.
        """
        # ##: Only there still available places
        if not self._board.all():
            # ##: Randomly choose cell value between 2 and 4.
            values = self._generator.choice([2, 4], size=number_tile, p=[0.9, 0.1]).tolist()

            # ##: Randomly choose cells positions in board.
            available_cells = argwhere(self._board == 0)
            chosen_indices = self._generator.choice(len(available_cells), size=number_tile, replace=False)
            cells = [tuple(available_cells[idx]) for idx in chosen_indices]

            # ##: Fill empty cells.
            for cell, value in zip(cells, values):
                self._board[cell] = value

    def _is_done(self) -> bool:
        """
        Check if the game is finished.

        Returns
        -------
        bool
            True if the game is finished, False otherwise.
        """
        if not self._board.all():
            return False

        # ##: Check if there still valid action.
        for action in self.ACTIONS.values():
            rotated_board = rot90(self._board, k=action)
            _, updated_board = slide_and_merge(rotated_board)
            if not array_equal(rotated_board, updated_board):
                return False

        return True

    @property
    def is_finished(self) -> bool:
        """
        Check if the game is finished.

        Returns
        -------
        bool
            True if the game is finished, False otherwise.
        """
        return self._is_done()

    @property
    def board(self) -> ndarray:
        """
        Get the current state of the game board.

        Returns
        -------
        ndarray
            Current state of the game board.
        """
        return self._board

    @property
    def size(self) -> int:
        """
        Get the size of the game board.

        Returns
        -------
        int
            Size of the game board.
        """
        return self._size

    def reset(self, seed: Optional[int] = None) -> ndarray:
        """
        Initialize an empty board and add two random tiles.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        ndarray
            The new game board.
        """
        self._generator = default_rng(seed)
        self._board = zeros((self._size, self._size), dtype=int64)
        self._fill_cells(number_tile=2)
        return self._board

    def step(self, action: int) -> Tuple[ndarray, Union[int, float], bool]:
        """
        Apply the selected action to the board.

        Parameters
        ----------
        action : int
            Action to apply.

        Returns
        -------
        Tuple[ndarray, Union[int, float], bool]
            Updated board, reward, game state.
        """
        reward = -4

        # ##: Applied action.
        rotated_board = rot90(self._board, k=action)
        score, updated_board = slide_and_merge(rotated_board)

        # ##: Fill new cell only if the board has evolved.
        if not array_equal(rotated_board, updated_board):
            self._board = rot90(updated_board, k=4 - action)
            reward = score - compute_penalties(self._board)

            # ##: Fill randomly one cell.
            self._fill_cells(number_tile=1)

        # ##: Check if game is finished.
        done = self._is_done()

        return self._board, reward, done

    def render(self, mode: str = "human") -> None:
        """
        Render the game board.

        Parameters
        ----------
        mode : str
            Mode for rendering.
        """
        if mode == "human":
            for row in self._board.tolist():
                print(" \t".join(map(str, row)))
