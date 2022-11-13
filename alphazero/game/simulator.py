# -*- coding:utf-8 -*-
"""
Simulator for helping during Monte Carlos Tree Search
"""
from typing import List, Sequence, Tuple

import numpy as np
from numpy import ndarray

from alphazero.addons.types import SimulatorOutput, StochasticState
from simulate_2048.envs.utils import slide_and_merge, compute_penalties


class Simulator:
    """
    Simulator class.
    Implement the rules of the 2048 Game.
    """

    @staticmethod
    def _stochastic_states(state: ndarray) -> List[StochasticState]:
        """
        Generate all possible states.

        Parameters
        ----------
        state: ndarray
            Transient state

        Returns
        -------
        list[StochasticState]
            List of possible state
        """
        # ##: If board is full, return it.
        if state.all():
            return [StochasticState(state=state.copy(), probability=1.0)]

        # ##: Store all possible states.
        all_possibilities = []
        for value, prob in [(2, 0.9), (4, 0.1)]:
            # ##: Get all available positions.
            available_cells = np.argwhere(state == 0)

            # ##: Generate possible states.
            for position in available_cells:
                # ##: New state
                board = state.copy()
                board[tuple(position)] = value
                prior = prob * 1 / len(available_cells)

                all_possibilities.append(StochasticState(state=board, probability=prior))

        return all_possibilities

    def _apply_action(self, state: ndarray, action: int) -> Tuple[List[StochasticState], int]:
        """
        Apply action to a state.

        Parameters
        ----------
        state: ndarray
            Current state
        action: Action
            Move to apply

        Returns
        -------
        list
            List of all possible state from a specific action and the reward.
        """
        reward = -10

        # ##: Applied action.
        rotated_board = np.rot90(state, k=action)
        score, updated_board = slide_and_merge(rotated_board)
        penalty = compute_penalties(rotated_board)

        # ##: If same board, return simulation output.
        if np.array_equal(rotated_board, updated_board):
            return [StochasticState(state=state.copy(), probability=1.0)], reward

        # ##: Board has evolved, compute reward.
        _board = np.rot90(updated_board, k=4 - action)
        reward = score  # - penalty

        # ##: Generate all stochastic states possibles.
        all_states = self._stochastic_states(_board)

        return all_states, reward

    def step(self, state: ndarray) -> SimulatorOutput:
        """
        Generate all possible next state.

        Parameters
        ----------
        state: ndarray
            Current state

        Returns
        -------
        SimulatorOutput
            All possible next state
        """
        # ##: Generate all possible state for all actions.
        outputs = {}
        for action in range(4):
            outputs[action] = self._apply_action(state, action)

        return SimulatorOutput(stochastic_states=outputs)

    @classmethod
    def legal_actions(cls, state: ndarray) -> Sequence[int]:
        """
        Returns the legal actions for the current state.

        Parameters
        ----------
        state: ndarray
            Current state

        Returns
        -------
        Sequence
            List of legal action
        """

        legal_moves = []

        # ##: Loop over all possible moves.
        for action in range(4):
            board = state.copy()

            # ##: Generate next board.
            rotated_board = np.rot90(board, k=action)
            _, updated_board = slide_and_merge(rotated_board)
            if not np.array_equal(rotated_board, updated_board):
                legal_moves.append(action)

        return legal_moves
