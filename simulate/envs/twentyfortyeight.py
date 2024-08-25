# -*- coding: utf-8 -*-
"""
Class describing the 2048 game for an agent.
"""
from typing import Optional, Tuple

from numpy import int64, ndarray, zeros

from simulate.utils import fill_cells, is_done, next_state


class TwentyFortyEight:
    """
    2048 game environment.

    This class implements the core mechanics of the 2048 game, providing methods to initialize the game board,
    make moves, and check the game state.

    Attributes
    ----------
    ACTIONS : dict
        A dictionary mapping action names to their corresponding integer values.

    Methods
    -------
    is_finished : bool
        Property that checks if the game is finished.
    observation : ndarray
        Property that returns the current state of the game board.
    reset() -> ndarray
        Resets the game to its initial state.
    step(action: int) -> Tuple[ndarray, float, bool]
        Applies an action to the game state and returns the result.
    render(mode: str = "human") -> None
        Renders the current game state.
    """

    # ##: All Actions.
    ACTIONS = {"left": 0, "up": 1, "right": 2, "down": 3}

    def __init__(self, size: int = 4):
        """
        Initialize the 2048 game board.

        Parameters
        ----------
        size : int, optional
            The size of the square grid (default is 4).
        """
        self.size = size
        self._board: Optional[ndarray] = None
        self.reset()

    @property
    def is_finished(self) -> bool:
        """
        Check if the game is finished.

        Returns
        -------
        bool
            True if the game is finished (no more moves possible), False otherwise.
        """
        return is_done(self._board)

    @property
    def observation(self) -> ndarray:
        """
        Get the current state of the game board.

        Returns
        -------
        ndarray
            The current state of the game board as a 2D numpy array.
        """
        return self._board

    def reset(self) -> ndarray:
        """
        Initialize an empty board and add two random tiles.

        This method resets the game to its initial state, creating a new board with two randomly placed
        tiles (usually 2's, occasionally 4's).

        Returns
        -------
        ndarray
            The new game board as a 2D numpy array.
        """
        self._board = zeros((self.size, self.size), dtype=int64)
        self._board = fill_cells(state=self._board, number_tile=2)
        return self._board

    def step(self, action: int) -> Tuple[ndarray, float, bool]:
        """
        Apply the selected action to the board.

        This method updates the game state based on the given action, adds a new tile to the board, and
        returns the new state along with the reward and whether the game has ended.

        Parameters
        ----------
        action : int
            The action to apply, corresponding to a move direction (0: left, 1: up, 2: right, 3: down).

        Returns
        -------
        Tuple[ndarray, float, bool]
            A tuple containing:
            - The updated game board (ndarray)
            - The reward obtained from this action (float)
            - Whether the game has finished after this action (bool)
        """
        self._board, reward = next_state(state=self._board, action=action)
        return self._board, reward, self.is_finished

    def render(self, mode: str = "human") -> None:
        """
        Render the game board.

        This method prints the current state of the game board to the console.

        Parameters
        ----------
        mode : str, optional
            The mode for rendering. Currently, only "human" mode is supported, which prints to
            the console (default is "human").

        Notes
        -----
        This method provides a simple text-based visualization of the game board. For more advanced rendering,
        consider implementing a graphical interface.
        """
        if mode == "human":
            for row in self._board.tolist():
                print(" \t".join(map(str, row)))
