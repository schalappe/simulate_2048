# -*- coding:utf-8 -*-
"""
Simulator for helping during Monte Carlos Tree Search
"""
from typing import List, Sequence, Tuple

import numpy as np
from numba import jit, prange
from numpy import ndarray

from alphazero.addons.types import SimulatorOutput, StochasticState
from alphazero.models.network import NetworkOutput
from simulate_2048.envs.utils import slide_and_merge


@jit(nopython=True, cache=True)
def stochastic_states(state: ndarray) -> Sequence[Tuple[ndarray, float]]:
    """
    Generate all possible states.

    Parameters
    ----------
    state: ndarray
        Current state

    Returns
    -------
    List
        All possible state
    """
    all_possibilities = []
    for value, prob in [(2, 0.9), (4, 0.1)]:
        # ##: Get all available positions.
        available_cells = np.argwhere(state == 0)

        # ##: Generate possible states.
        for position in available_cells:
            # ##: New state
            board = state.copy()
            board[(position[0], position[1])] = value
            prior = prob * 1 / len(available_cells)

            all_possibilities.append((board, prior))
    return all_possibilities


@jit(nopython=True, cache=True)
def legal_actions(state: ndarray) -> Sequence[int]:
    """
    Returns the legal actions for the current state.

    Parameters
    ----------
    state: ndarray
        Current state

    Returns
    -------
    Sequence[int]
        List of legal action
    """

    legal_moves = np.zeros(4, dtype=np.int8)

    # ##: Loop over all possible moves.
    for action in prange(4):
        board = state.copy()

        # ##: Generate next board.
        rotated_board = np.rot90(board, k=action)
        _, updated_board = slide_and_merge(rotated_board)
        if not np.array_equal(rotated_board, updated_board):
            legal_moves[action] = action

    return np.extract(legal_moves > 0, legal_moves)


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
        all_possibilities = stochastic_states(state)
        all_possibilities = [StochasticState(state=np.reshape(board, -1), probability=prior) for board, prior in all_possibilities]

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

        # ##: Applied action.
        _state = np.reshape(state, (4, 4))
        rotated_board = np.rot90(_state, k=action)
        score, updated_board = slide_and_merge(rotated_board)

        # ##: Board has evolved, compute reward.
        _board = np.rot90(updated_board, k=4 - action)

        # ##: Generate all stochastic states possibles.
        all_states = self._stochastic_states(_board)

        return all_states, score

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
        outputs = {}

        # ##: Compute all legal moves.
        _state = np.reshape(state, (4, 4))
        legal_moves = legal_actions(_state)

        # ##: Generate all possible state for all actions.
        for action in range(4):
            if action in legal_moves:
                outputs[action] = self._apply_action(state, action)
            else:
                outputs[action] = []

        return SimulatorOutput(stochastic_states=outputs)

    @classmethod
    def mask_illegal_actions(cls, state: ndarray, outputs: NetworkOutput) -> NetworkOutput:
        """
        Masks any actions which are illegal at the root.

        Parameters
        ----------
        state: ndarray
            Current state
        outputs: NetworkOutput
            Previous network output

        Returns
        -------
        NetworkOutput
            New network output with legal action
        """

        # ##: We mask out and keep only the legal actions.
        masked_policy = {}
        _state = np.reshape(state, (4, 4))
        network_policy = outputs.probabilities
        norm = 0
        for action in legal_actions(_state):
            if action in network_policy:
                masked_policy[action] = network_policy[action]
            else:
                masked_policy[action] = 0.0
            norm += masked_policy[action]

        # ##: Re-normalize the masked policy.
        masked_policy = {a: v / norm for a, v in masked_policy.items()}
        return NetworkOutput(value=outputs.value, probabilities=masked_policy)
