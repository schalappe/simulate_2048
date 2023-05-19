# -*-  coding: utf-8 -*-
"""
Set of test for GameBoard
"""
from random import choice
from typing import Sequence
from unittest import TestCase, main

from numpy import argwhere, array, array_equal, extract, ndarray, rot90

from simulate_2048 import GameBoard
from simulate_2048.envs.utils import slide_and_merge
from simulate_2048.wrappers import EncodedObservation


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

    legal_moves = array([-1, -1, -1, -1])

    # ##: Loop over all possible moves.
    for action in range(4):
        board = state.copy()

        # ##: Generate next board.
        rotated_board = rot90(board, k=action)
        _, updated_board = slide_and_merge(rotated_board)
        if not array_equal(rotated_board, updated_board):
            legal_moves[action] = action

    return list(extract(legal_moves > -1, legal_moves))


class GameBoardTest(TestCase):
    # ##: Board and update board.
    BOARD = array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 4], [4, 0, 4, 2]])
    UP_BOARD = array([[4, 0, 4, 4], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]])
    DOWN_BOARD = array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 4], [4, 0, 4, 2]])
    LEFT_BOARD = array([[0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 0], [8, 2, 0, 0]])
    RIGHT_BOARD = array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 8, 2]])

    # ##: Actions possible.
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    def test_reset_game(self):
        # ##: Check many times
        for _ in range(100):
            # ##: Generate new board.
            game = GameBoard(size=4)
            board = game.board

            # ##: Get the occupied cells.
            occupied_cells = argwhere(board != 0)
            occupied_cells = tuple(map(tuple, occupied_cells))

            # ##: Check that only two cells is occupied.
            self.assertEqual(len(occupied_cells), 2)

            # ##: Check that the value of occupied cells is 2 or 4.
            good_values = [board[cell] in [2, 4] for cell in occupied_cells]
            self.assertTrue(all(good_values))

    def test_slide_merge(self):
        # ##: Actions and board.
        actions = [self.LEFT, self.UP, self.RIGHT, self.DOWN]
        boards = [self.LEFT_BOARD, self.UP_BOARD, self.RIGHT_BOARD, self.DOWN_BOARD]

        for action, expected_board in zip(actions, boards):
            # ##: Applied action.
            rotated_board = rot90(self.BOARD, k=action)
            _, updated_board = slide_and_merge(rotated_board)
            next_board = rot90(updated_board, k=4 - action)

            # ##: Compare boards.
            self.assertTrue(array_equal(expected_board, next_board))

    def test_step(self):
        # ##: Generate new board.
        game = GameBoard(size=4)

        # ##: Check that only two cell are filled.
        occupied_cells = argwhere(game.board != 0)
        occupied_cells = tuple(map(tuple, occupied_cells))
        self.assertEqual(len(occupied_cells), 2)

        # ##: Check that the value of occupied cells is 2 or 4.
        good_values = [game.board[cell] in [2, 4] for cell in occupied_cells]
        self.assertTrue(all(good_values))

        # ##: Perform an action.
        action = choice(legal_actions(game.board))
        _, _, done, _, _ = game.step(action)

        # ##: Check that reward is positive and one cell has been filled.
        occupied_cells = argwhere(game.board != 0)
        occupied_cells = tuple(map(tuple, occupied_cells))
        self.assertLessEqual(len(occupied_cells), 3)

        while not done:
            # ##: Perform an action.
            action = choice(legal_actions(game.board))
            _, _, done, _, _ = game.step(action)


class WrapperTest(TestCase):

    def test_encodage(self):
        # ##: Generate new board.
        game = GameBoard(size=4)

        # ##: add wrapper.
        encoded_game = EncodedObservation(game)

        # ##: Get observation from reset.
        board, _ = encoded_game.reset()
        self.assertEqual(len(board), 496)


if __name__ == "__main__":
    main()
