# -*-  coding: utf-8 -*-
"""
Set of test for Observation Wrapper
"""
import unittest

import gym
import numpy as np

from simulate_2048 import (
    FlattenLogObservation,
    FlattenObservation,
    FlattenOneHotObservation,
    LogObservation,
)


class ObservationWrappersTest(unittest.TestCase):
    # ##: Game board
    BOARD = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 4], [2, 0, 4, 2]])

    def test_flatten_observation(self):
        for _ in range(100):
            # ##: Generate new board.
            game = gym.make("GameBoard")
            wrapper_game = FlattenObservation(game)

            # ##: Test observation.
            flatten_observation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 4, 2])
            good_observation = np.array_equal(flatten_observation, wrapper_game.observation(self.BOARD))
            self.assertTrue(good_observation)

    def test_flatten_log_observation(self):
        for _ in range(100):
            # ##: Generate new board.
            game = gym.make("GameBoard")
            wrapper_game = FlattenLogObservation(game)

            # ##: Test observation.
            flatten_observation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 2, 1])
            good_observation = np.array_equal(flatten_observation, wrapper_game.observation(self.BOARD))
            self.assertTrue(good_observation)

    def test_log_observation(self):
        for _ in range(100):
            # ##: Generate new board.
            game = gym.make("GameBoard")
            wrapper_game = LogObservation(game)

            # ##: Test observation..
            flatten_observation = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [1, 0, 2, 1]])
            good_observation = np.array_equal(flatten_observation, wrapper_game.observation(self.BOARD))
            self.assertTrue(good_observation)

    def test_flatten_one_hot_observation(self):
        for _ in range(100):
            # ##: Generate new board.
            game = gym.make("GameBoard")
            wrapper_game = FlattenOneHotObservation(game)

            # ##: Prepare output.
            flatten_observation = np.reshape(self.BOARD, -1)
            flatten_observation[flatten_observation == 0] = 1
            flatten_observation = np.log2(flatten_observation)
            flatten_observation = flatten_observation.astype(int)
            flatten_observation = np.reshape(np.eye(31)[flatten_observation], -1)

            # ##: Test observation.
            good_observation = np.array_equal(flatten_observation, wrapper_game.observation(self.BOARD))
            self.assertTrue(good_observation)


if __name__ == "__main__":
    unittest.main()
