# -*- coding: utf-8 -*-
"""
Encode the game state in binary representation.
"""
from typing import Tuple

from numpy import eye, int64, log2, ndarray, reshape

from simulate.envs import TwentyFortyEight


class EncodedTwentyFortyEight(TwentyFortyEight):
    """
    Encoded version of the 2048 game environment.

    This class extends the base `TwentyFortyEight` class to provide an encoded representation of the game state.
    The encoding uses a one-hot representation for each cell value, which can be more suitable for machine
    learning models.

    Methods
    -------
    reset() -> np.ndarray
        Resets the game and returns the encoded initial state.
    step(action: int) -> Tuple[np.ndarray, float, bool]
        Applies an action and returns the encoded next state, reward, and game status.
    observation : np.ndarray
        Property that returns the current encoded state of the game board.

    Notes
    -----
    The encoding process converts each cell value to its base-2 logarithm and then applies one-hot encoding.
    This results in a 3D array where each cell is represented by a one-hot vector.
    """

    def __init__(self, size: int = 4, block_size: int = 31):
        """
        Initialize the encoded 2048 game board.

        Parameters
        ----------
        size : int, optional
            The size of the game board (default is 4).
        block_size : int, optional
            The size of the one-hot encoding block for each cell (default is 31).
            This should be large enough to represent the largest possible tile value.
        """
        self._block_size = block_size
        super().__init__(size)

    def reset(self) -> ndarray:
        """
        Reset the environment and return an encoded observation.

        This method resets the game to its initial state and returns the encoded representation of the initial board.

        Returns
        -------
        ndarray
            The encoded initial state of the game board.
        """
        super().reset()
        return self.observation

    def step(self, action: int) -> Tuple[ndarray, float, bool]:
        """
        Apply the action, step the environment, and return an encoded observation.

        This method applies the given action to the game state and returns the encoded representation of the new
        state along with the reward and game status.

        Parameters
        ----------
        action : int
            The action to apply to the game board (0: left, 1: up, 2: right, 3: down).

        Returns
        -------
        Tuple[ndarray, float, bool]
            A tuple containing:
            - The encoded state of the game board after the action (np.ndarray)
            - The reward obtained from this action (float)
            - Whether the game has finished after this action (bool)
        """
        _, reward, terminated = super().step(action)
        return self.observation, reward, terminated

    @property
    def observation(self) -> ndarray:
        """
        Encode the game board state into a binary representation.

        This property provides an encoded version of the current game state, using a one-hot representation for
        each cell value.

        Returns
        -------
        ndarray
            The binary-encoded state of the game board.

        Notes
        -----
        The encoding process involves the following steps:
        1. Flatten the game board state.
        2. Replace zeros with ones to avoid issues with logarithm calculations.
        3. Take the base-2 logarithm of each cell value.
        4. Apply one-hot encoding to the logarithmic values.
        5. Reshape the result into a 3D array.

        The resulting array has shape (board_size, board_size, block_size),
        where each cell is represented by a one-hot vector of length block_size.
        """
        obs = super().observation.flatten()
        obs[obs == 0] = 1
        obs = log2(obs).astype(int64)
        encoded = eye(self._block_size, dtype=int64)[obs]
        return reshape(encoded, -1)
