# -*-  coding: utf-8 -*-
"""
Set of test for GameBoard
"""
import unittest
import numpy as np
from simulate_2048 import GameBoard


class GameBoardTest(unittest.TestCase):
    # ## ----> Board and update board.
    BOARD = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 4],
        [2, 0, 4, 2]
    ])
    UP_BOARD = np.array([
        [2, 0, 4, 4],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    DOWN_BOARD = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 4],
        [2, 0, 4, 2]
    ])
    LEFT_BOARD = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [4, 0, 0, 0],
        [2, 4, 2, 0]
    ])
    RIGHT_BOARD = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 4],
        [0, 2, 4, 2]
    ])

    # ## ----> Actions possible.
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    def test_reset_game(self):
        # ## ----> Check many times
        for _ in range(100):
            # ## ----> Generate new board.
            game = GameBoard(size=4)
            board = game.board

            # ## ----> Get the occupied cells.
            occupied_cells = np.argwhere(board != 0)
            occupied_cells = tuple(map(tuple, occupied_cells))

            # ## ----> Check that only two cells is occupied.
            self.assertEqual(len(occupied_cells), 2)

            # ## ----> Check that the value of occupied cells is 2 or 4.
            good_values = [board[cell] in [2, 4] for cell in occupied_cells]
            self.assertTrue(all(good_values))

    def test_fill_cell(self):
        # ## ----> Check many times
        for _ in range(100):
            # ## ----> Generate new board.
            game = GameBoard(size=4)

            # ## ----> Fill only one cell
            game._fill_cells(number_tile=1)

            # ## ----> Get the occupied cells.
            occupied_cells = np.argwhere(game.board != 0)
            occupied_cells = tuple(map(tuple, occupied_cells))

            # ## ----> Check that only two cells is occupied.
            self.assertEqual(len(occupied_cells), 3)

            # ## ----> Check that the value of occupied cells is 2 or 4.
            good_values = [game.board[cell] in [2, 4] for cell in occupied_cells]
            self.assertTrue(all(good_values))

    def test_slide_merge(self):
        # ## ----> Generate new board.
        game = GameBoard(size=4)

        # ## ----> Actions and board.
        actions = [self.LEFT, self.UP, self.RIGHT, self.DOWN]
        boards = [self.LEFT_BOARD, self.UP_BOARD, self.RIGHT_BOARD, self.DOWN_BOARD]

        for action, expected_board in zip(actions, boards):
            # ## ----> Applied action.
            rotated_board = np.rot90(self.BOARD, k=action)
            _, updated_board = game._slide_and_merge(rotated_board)
            next_board = np.rot90(updated_board, k=4-action)

            # ## ----> Compare boards.
            self.assertTrue(np.array_equal(expected_board, next_board))


if __name__ == '__main__':
    unittest.main()
