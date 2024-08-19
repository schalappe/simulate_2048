# -*-  coding: utf-8 -*-
"""
Set of test for GameBoard.
"""
from unittest import TestCase, main

import numpy as np

from simulate.envs.gameboard import GameBoard
from simulate.envs.utils import illegal_actions, merge_column, slide_and_merge
from simulate.wrappers import EncodedGameBoard


class TestGameBoard(TestCase):
    """
    Test for the GameBoard class.
    This class tests the core functionality of the 2048 game logic.
    """

    def setUp(self):
        """Initialize a new game board before each test."""
        self.game = GameBoard(size=4)

    def test_init(self):
        """Test if the game board is correctly initialized."""
        self.assertEqual(self.game.size, 4)
        self.assertIsNotNone(self.game.board)
        self.assertEqual(self.game.board.shape, (4, 4))

    def test_reset(self):
        """Test if the game board is correctly reset with two initial tiles."""
        board = self.game.reset(seed=42)
        self.assertEqual(np.count_nonzero(board), 2)
        self.assertTrue(np.all(np.isin(board[board != 0], [2, 4])))

    def test_step_valid_move(self):
        """Test if a valid move is correctly processed."""
        self.game._board = np.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        board, reward, done = self.game.step(GameBoard.ACTIONS["left"])
        self.assertEqual(board[0, 0], 4)
        self.assertGreater(reward, 0)
        self.assertFalse(done)

    def test_step_invalid_move(self):
        """Test if an invalid move is correctly handled."""
        self.game._board = np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        board, reward, done = self.game.step(GameBoard.ACTIONS["left"])
        self.assertTrue(np.array_equal(board, self.game._board))
        self.assertEqual(reward, -4)
        self.assertFalse(done)

    def test_is_finished(self):
        """Test if the game correctly identifies a finished state."""
        self.game._board = np.array(
            [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [8192, 16384, 32768, 65536]]
        )
        self.assertTrue(self.game.is_finished)

    def test_not_finished(self):
        """Test if the game correctly identifies an unfinished state."""
        self.game._board = np.array(
            [[2, 2, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [8192, 16384, 32768, 65536]]
        )
        self.assertFalse(self.game.is_finished)


class TestUtils(TestCase):
    """
    Test for utility functions.
    This class tests the helper functions used in the game logic.
    """

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

    def test_illegal_actions(self):
        """Test if illegal actions are correctly identified."""
        board = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [8192, 16384, 32768, 65536]])
        illegal = illegal_actions(board)
        self.assertEqual(set(illegal), set(GameBoard.ACTIONS.values()))


class TestEncodedGameBoard(TestCase):
    """
    Test for the EncodedGameBoard class.
    This class tests the binary encoding of the game state.
    """

    def setUp(self):
        """Initialize a new encoded game board before each test."""
        self.game = EncodedGameBoard(size=4, block_size=31)

    def test_init(self):
        """Test if the encoded game board is correctly initialized."""
        self.assertEqual(self.game.size, 4)
        self.assertEqual(self.game._block_size, 31)

    def test_reset(self):
        """Test if the encoded game board is correctly reset."""
        encoded_board = self.game.reset(seed=42)
        self.assertEqual(encoded_board.shape, (4 * 4 * 31,))

    def test_step(self):
        """Test if a step in the encoded game is correctly processed."""
        self.game.reset(seed=42)
        encoded_board, reward, done = self.game.step(GameBoard.ACTIONS["left"])
        self.assertEqual(encoded_board.shape, (4 * 4 * 31,))
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)

    def test_observation(self):
        """Test if the game state is correctly encoded."""
        board = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [8192, 16384, 32768, 65536]])
        encoded = self.game.observation(board)
        self.assertEqual(encoded.shape, (4 * 4 * 31,))
        self.assertTrue(np.all(np.isin(encoded, [0, 1])))


if __name__ == "__main__":
    main()
