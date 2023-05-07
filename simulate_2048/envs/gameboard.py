# -*- coding: utf-8 -*-
"""
Classe décrivant le jeu 2048 pour un agent
"""
from typing import Dict, List, Optional, Tuple, Union

import gym
from gym import spaces
from gym.utils import seeding
from numpy import argwhere, array_equal, int64, ndarray, reshape, rot90, zeros

from .utils import slide_and_merge


class GameBoard(gym.Env):
    """2048 game environment."""

    # ##: Available actions.
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    # ##: All Actions.
    ACTIONS = [LEFT, UP, RIGHT, DOWN]
    ACTIONS_STRING = {LEFT: "left", UP: "up", RIGHT: "right", DOWN: "down"}

    # ##: Game variables.
    _board = None

    def __init__(self, size: int = 4):
        self.size = size  # ##: The size of the square grid.
        self.observation_space = spaces.Box(low=0, high=2**32, shape=(size * size,), dtype=int64)
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # ##: Reset game.
        self.reset()

    def __random_cell_value(self, number_cell: int) -> List[int]:
        """
        Randomly choose cell value between 2 and 4.

        Parameters
        ----------
        number_cell: int
            Number of value to generate

        Returns
        -------
        list
            List of chosen value
        """
        return self._np_random.choice([2, 4], size=number_cell, p=[0.9, 0.1]).tolist()

    def __random_position(self, number_cell: int) -> Tuple:
        """
        Randomly choose cells positions in board.

        Parameters
        ----------
        number_cell: int
            Number of cells to select

        Returns
        -------
        tuple
            List of chosen cells
        """
        available_cells = argwhere(self._board == 0)
        chosen_cells = self._np_random.choice(len(available_cells), size=number_cell, replace=False)
        cell_positions = available_cells[chosen_cells]
        return tuple(map(tuple, cell_positions))

    def _fill_cells(self, number_tile):
        """
        Find empty cells and fill them with 2 or 4.

        Parameters
        ----------
        number_tile: int
            Number of cell to fill
        """
        # ##: Only there still available places
        if not self._board.all():
            values = self.__random_cell_value(number_cell=number_tile)
            cells = self.__random_position(number_cell=number_tile)

            for cell, value in zip(cells, values):
                self._board[cell] = value

    def _is_done(self) -> bool:
        """
        Check if the game is finished. The game is finished when there aren't valid action.

        Returns
        -------
        bool
            True if the game is finished
            False else
        """
        board = self._board.copy()

        # ##: Check if all cells is filled.
        if not board.all():
            return False

        # ##: Check if there still valid action.
        for action in self.ACTIONS:
            rotated_board = rot90(board, k=action)
            _, updated_board = slide_and_merge(rotated_board)
            if not updated_board.all():
                return False

        return True

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[ndarray, Optional[Dict]]:
        """
        Initialize empty board then add randomly two tiles.

        Returns
        -------
        Tuple
            New game board and information
        """
        self._np_random, seed = seeding.np_random(seed)
        self._board = zeros(shape=[self.size, self.size], dtype=int64)
        self._fill_cells(number_tile=2)

        return reshape(self._board, -1), options

    def step(self, action: int) -> Tuple[ndarray, Union[int, float], bool, bool, Dict]:
        """
        Applied the selected action to the board.

        Parameters
        ----------
        action: int
            Action to apply

        Returns
        -------
        tuple
            Update board, reward, state of the game and info
        """
        reward = -4

        # ##: Applied action.
        rotated_board = rot90(self._board, k=action)
        # penalty = compute_penalties(rotated_board)
        score, updated_board = slide_and_merge(rotated_board)

        # ##: Fill new cell only if the board has evolved.
        if not array_equal(rotated_board, updated_board):
            self._board = rot90(updated_board, k=4 - action)
            reward = score  # - penalty

            # ##: Fill randomly one cell.
            self._fill_cells(number_tile=1)

        # ##: Check if game is finished.
        done = self._is_done()

        return reshape(self._board, -1), reward, done, False, {}

    def render(self, mode="human"):
        """
        Render game board.

        Parameters
        ----------
        mode: str
            Mode
        """
        if mode == "human":
            for row in self._board.tolist():
                print(" \t".join(map(str, row)))
