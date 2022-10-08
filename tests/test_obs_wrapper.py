# -*-  coding: utf-8 -*-
"""
Set of test for Observation Wrapper
"""
import unittest

import gym
import numpy as np

from simulate_2048 import FlattenLogObservation, FlattenObservation, LogObservation


class ObservationWrappersTest(unittest.TestCase):
    # ## ----> Game board
    BOARD = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 4], [2, 0, 4, 2]])

    def test_flatten_observation(self):
        for _ in range(100):
            # ## ----> Generate new board.
            game = gym.make("GameBoard")
            wrapper_game = FlattenObservation(game)

            # ## ----> test observation
            flatten_observation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 4, 2])
            good_observation = np.array_equal(flatten_observation, wrapper_game.observation(self.BOARD))
            self.assertTrue(good_observation)

    def test_flatten_log_observation(self):
        for _ in range(100):
            # ## ----> Generate new board.
            game = gym.make("GameBoard")
            wrapper_game = FlattenLogObservation(game)

            # ## ----> test observation
            flatten_observation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 2, 1])
            good_observation = np.array_equal(flatten_observation, wrapper_game.observation(self.BOARD))
            self.assertTrue(good_observation)

    def test_log_observation(self):
        for _ in range(100):
            # ## ----> Generate new board.
            game = gym.make("GameBoard")
            wrapper_game = LogObservation(game)

            # ## ----> test observation
            flatten_observation = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [1, 0, 2, 1]])
            good_observation = np.array_equal(flatten_observation, wrapper_game.observation(self.BOARD))
            self.assertTrue(good_observation)


if __name__ == "__main__":
    unittest.main()
