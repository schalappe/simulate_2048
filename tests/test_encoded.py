# -*-  coding: utf-8 -*-
"""
Set of test for Encoded Game.
"""
from unittest import TestCase, main

import numpy as np

from simulate.envs import TwentyFortyEight
from simulate.wrappers import EncodedTwentyFortyEight


class TestEncodedGameBoard(TestCase):
    """
    Test for the EncodedGameBoard class.
    This class tests the binary encoding of the game state.
    """

    def setUp(self):
        """Initialize a new encoded game board before each test."""
        self.game = EncodedTwentyFortyEight(size=4, block_size=31)

    def test_reset(self):
        """Test if the encoded game board is correctly reset."""
        encoded_board = self.game.reset()
        self.assertEqual(encoded_board.shape, (4 * 4 * 31,))

    def test_step(self):
        """Test if a step in the encoded game is correctly processed."""
        self.game.reset()
        encoded_board, reward, done = self.game.step(TwentyFortyEight.ACTIONS["left"])
        self.assertEqual(encoded_board.shape, (4 * 4 * 31,))
        self.assertIsInstance(reward, (int, float))
        self.assertFalse(done)

    def test_observation(self):
        """Test if the game state is correctly encoded."""
        board = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [8192, 16384, 32768, 65536]])
        self.game._board = board
        encoded = self.game.observation
        self.assertEqual(encoded.shape, (4 * 4 * 31,))
        self.assertTrue(np.all(np.isin(encoded, [0, 1])))


if __name__ == "__main__":
    main()
