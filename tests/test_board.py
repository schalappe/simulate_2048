"""
Set of test for GameBoard.
"""

from unittest import TestCase, main

import numpy as np

from twentyfortyeight.core.gameboard import (
    after_state,
    after_state_lazy,
    generate_outcome,
    is_done,
    merge_column,
    next_state,
    slide_and_merge,
)
from twentyfortyeight.core.gamemove import illegal_actions
from twentyfortyeight.envs.twentyfortyeight import TwentyFortyEight


class TestGameBoard(TestCase):
    """
    Test for the GameBoard class.
    This class tests the core functionality of the 2048 game logic.
    """

    def setUp(self):
        """Initialize a new game board before each test."""
        self.game = TwentyFortyEight(size=4)

    def test_reset(self):
        """Test if the game board is correctly reset with two initial tiles."""
        board = self.game.reset()
        self.assertEqual(np.count_nonzero(board), 2)
        self.assertTrue(np.all(np.isin(board[board != 0], [2, 4])))

    def test_step_valid_move(self):
        """Test if a valid move is correctly processed."""
        board = np.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        board, reward = next_state(board, TwentyFortyEight.ACTIONS['left'])
        self.assertEqual(board[0, 0], 4)
        self.assertGreater(reward, 0)

    def test_step_invalid_move(self):
        """Test if an invalid move is correctly handled."""
        board = np.array([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        print(illegal_actions(board))
        next_board, reward = next_state(board, TwentyFortyEight.ACTIONS['left'])
        self.assertTrue(np.array_equal(board, next_board))
        self.assertEqual(reward, 0)

    def test_is_finished(self):
        """Test if the game correctly identifies a finished state."""
        board = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [8192, 16384, 32768, 65536]])
        self.assertTrue(is_done(board))

    def test_not_finished(self):
        """Test if the game correctly identifies an unfinished state."""
        board = np.array([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertFalse(is_done(board))

    def test_merge_column(self):
        """Test if columns are correctly merged."""
        column = np.array([2, 2, 4, 4])
        score, result = merge_column(column)
        self.assertEqual(score, 12)
        np.testing.assert_array_equal(result, np.array([4, 8]))

    def test_slide_and_merge(self):
        """Test if the entire board is correctly slid and merged."""
        board = np.array([[2, 2, 4, 4], [0, 2, 2, 4], [2, 0, 0, 2], [2, 2, 2, 2]])
        score, result = slide_and_merge(board)
        expected = np.array([[4, 8, 0, 0], [4, 4, 0, 0], [4, 0, 0, 0], [4, 4, 0, 0]])
        self.assertEqual(score, 28)
        np.testing.assert_array_equal(result, expected)

    def test_after_state_lazy_equivalence(self):
        """Verify lazy generation produces same outcomes as eager."""
        board = np.array([[2, 4, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]])

        # ##>: Eager - generates all outcomes upfront.
        eager_outcomes = after_state(board)

        # ##>: Lazy - generates outcomes incrementally.
        base, cells, num_empty = after_state_lazy(board)
        lazy_outcomes = []
        for cell in cells:
            for value in [2, 4]:
                lazy_outcomes.append(generate_outcome(base, cell, value, num_empty))

        # ##>: Both should produce the same number of outcomes.
        self.assertEqual(len(eager_outcomes), len(lazy_outcomes))
        self.assertEqual(len(eager_outcomes), num_empty * 2)

        # ##>: Match each lazy outcome to an eager one (order may differ).
        for l_state, l_prob in lazy_outcomes:
            found = False
            for e_state, e_prob in eager_outcomes:
                if np.array_equal(e_state, l_state):
                    self.assertAlmostEqual(e_prob, l_prob, places=10)
                    found = True
                    break
            self.assertTrue(found, f'Lazy outcome not found in eager: {l_state}')

    def test_after_state_lazy_no_empty_cells(self):
        """Verify lazy generation handles full board correctly."""
        board = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [2, 4, 8, 16]])

        base, cells, num_empty = after_state_lazy(board)

        self.assertEqual(num_empty, 0)
        self.assertEqual(len(cells), 0)


if __name__ == '__main__':
    main()
