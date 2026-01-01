"""2048 game environment for reinforcement learning agents."""

from numpy import int64, ndarray, zeros

from twentyfortyeight.core.gameboard import fill_cells, is_done, next_state
from twentyfortyeight.utils.binary import encode_flatten
from twentyfortyeight.utils.normalize import normalize_reward


class TwentyFortyEight:
    """
    2048 game environment.

    This class implements the core mechanics of the 2048 game, providing methods to initialize the game board,
    make moves, and check the game state.
    """

    # ##: Current game state.
    _current_state: ndarray | None = None
    _current_reward: float | None = None

    # ##: All Actions.
    ACTIONS = {'left': 0, 'up': 1, 'right': 2, 'down': 3}

    def __init__(self, size: int = 4, encoded: bool = False, normalize: bool = False):
        """
        Initialize the 2048 game board.

        Parameters
        ----------
        size : int, optional
            The size of the square grid (default is 4).
        encoded : bool, optional
            Whether to binary encode the board or not (default is False).
        normalize : bool, optional
            Whether to normalize the reward or not (default is False).
        """
        self.size = size
        self._encoded = encoded
        self._normalize = normalize

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
        return is_done(self._current_state)

    @property
    def observation(self) -> ndarray:
        """
        Get the current state of the game board.

        Returns
        -------
        ndarray
            The current state of the game board as a 2D numpy array.
        """
        if self._encoded:
            return encode_flatten(self._current_state, encodage_size=31)
        return self._current_state

    @property
    def reward(self) -> float:
        """
        Get the current reward of the game.

        Returns
        -------
        float
            The current reward of the game.
        """
        if self._normalize:
            return normalize_reward(self._current_reward)
        return self._current_reward

    def reset(self, seed: int | None = None) -> ndarray:
        """
        Initialize an empty board and add two random tiles.

        This method resets the game to its initial state, creating a new board with two randomly placed
        tiles (usually 2's, occasionally 4's).

        Returns
        -------
        ndarray
            The new game board as a 2D numpy array.

        Notes
        -----
        - The initial board will have two tiles, typically 2's, placed randomly.
        - There's a small chance (10%) that one of the initial tiles will be a 4.
        """
        self._current_state = zeros((self.size, self.size), dtype=int64)
        self._current_state = fill_cells(state=self._current_state, number_tile=2, seed=seed)
        self._current_reward = 0
        return self.observation

    def step(self, action: int) -> tuple[ndarray, float, bool]:
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
        tuple[ndarray, float, bool]
            A tuple containing:
            - The updated game board (ndarray)
            - The reward obtained from this action (float)
            - Whether the game has finished after this action (bool)

        Notes
        -----
        - The reward is the sum of merged tile values in this step.
        - The game is considered finished if no more moves are possible.
        - A new tile (2 or 4) is added to the board after each successful move.
        """
        self._current_state, self._current_reward = next_state(state=self._current_state, action=action)
        return self.observation, self.reward, self.is_finished

    def render(self) -> None:
        """
        Render the game board. This method prints the current state of the game board to the console.
        """
        for row in self._current_state.tolist():
            print(' \t'.join(map(str, row)))
