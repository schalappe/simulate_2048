# -*- coding: utf-8 -*-
"""
Classe décrivant le jeu 2048 pour un agent
"""
from typing import Any, List, Tuple, Union

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from .utils import compute_penalties, slide_and_merge


class GameBoard(gym.Env):
    """
    2048 game environment.
    """

    # ##: Available actions.
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    ACTIONS_STRING = {LEFT: "left", UP: "up", RIGHT: "right", DOWN: "down"}

    # ##: All Actions.
    ACTIONS = [LEFT, UP, RIGHT, DOWN]

    def __init__(self, size: int = 4):
        self.size = size  # ##: The size of the square grid.
        self.observation_space = spaces.Box(low=0, high=2**32, shape=(size, size), dtype=np.int64)
        self.action_space = spaces.Discrete(4)  # ##: 4 actions possible.

        # ##: Initialize variables.
        self._board = None
        self._old_board = None
        self.random = None

        # ##: Reset game.
        self.seed()
        self.reset()

    def __random_cell_value(self, number_cell: int) -> List:
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
        return self.random.choice([2, 4], size=number_cell, p=[0.9, 0.1]).tolist()

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
        available_cells = np.argwhere(self.board == 0)
        chosen_cells = self.random.choice(len(available_cells), size=number_cell, replace=False)
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
        if not self.board.all():
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
        board = self.board.copy()

        # ##: Check if all cells is filled.
        if not board.all():
            return False

        # ##: Check if there still valid action.
        for action in self.ACTIONS:
            rotated_board = np.rot90(board, k=action)
            _, updated_board = slide_and_merge(rotated_board)
            if not updated_board.all():
                return False

        return True

    def seed(self, seed=None):
        """
        Generate a random number generator
        """
        self.random, seed = seeding.np_random(seed)
        return seed

    def reset(self, **kwargs) -> Tuple:
        """
        Initialize empty board then add randomly two tiles.

        Parameters
        ----------
        **kwargs

        Returns
        -------
        Tuple
            New game board and information
        """
        self._board = np.zeros(shape=[self.size, self.size], dtype=np.int64)
        self._old_board = None
        self._fill_cells(number_tile=2)

        return self.board, {}

    def step(self, action: int) -> Tuple[Any, Union[int, Tuple], bool, bool, dict]:
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
        reward = -10

        # ##: Save old board.
        self._old_board = self._board.copy()

        # ##: Applied action.
        rotated_board = np.rot90(self._board, k=action)
        # penalty = compute_penalties(rotated_board)
        score, updated_board = slide_and_merge(rotated_board)

        # ##: Fill new cell only if the board has evolved.
        if not np.array_equal(rotated_board, updated_board):
            self._board = np.rot90(updated_board, k=4 - action)
            reward = score  # - penalty

            # ##: Fill randomly one cell.
            self._fill_cells(number_tile=1)

        # ##: Check if game is finished.
        done = self._is_done()

        return self.board, reward, done, False, {}

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

    @property
    def board(self) -> np.ndarray:
        """
        Return game board.

        Returns
        -------
        np.ndarray:
            Game board
        """
        return self._board

    @board.setter
    def board(self, board: np.ndarray):
        """
        Change the current game board.

        Parameters
        ----------
        board: np.ndarray
            New game board
        """
        self._board = board

    @property
    def old_board(self) -> np.ndarray:
        """
        Return the old board.

        Returns
        -------
        np.ndarray:
            Old game board
        """
        return self._old_board
