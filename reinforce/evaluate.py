# -*- coding: utf-8 -*-
"""
Script for evaluate an agent.
"""
from os.path import abspath, dirname, join

import gym
from module import Agent

from simulate_2048 import GameBoard

STORAGE_MODEL = join(dirname(dirname(abspath(__file__))), "models")

if __name__ == "__main__":
    import argparse
    import collections
    from itertools import count

    from numpy import max as max_array_values

    # ## ----> Get arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="Which model to warn-up", required=True, type=str)
    args = parser.parse_args()

    # ## ----> Evaluation with specific algorithm.
    if args.algo == "dqn":
        # ## ----> Create agent.
        model_path = join(STORAGE_MODEL, "dqn_model_dense_classic_affine")
        agent = Agent(model_path, 0.05)

        # ## ----> Create environment.
        game = gym.make("GameBoard", size=4, type_reward="affine")
        # game = GameBoard(size=4, type_reward="affine")

        # ## ----> Loop over 100 parties.
        score = []
        for i in range(100):
            # ## ----> Reset game.
            board, _ = game.reset()

            # ## ----> Play one party
            for t in count():
                # ## ----> Perform action.
                action = agent.select_action(board)
                next_board, _, done, _, _ = game.step(action)
                board = next_board

                # ## ----> store max element.
                print(f"Game: {i+1} - Score: {max_array_values(board)} - move: {t}", end="\r")
                if done:
                    score.append(max_array_values(game.board))
                    break

            print(f"Game: {i+1} is finished, score egal {score[-1]}.")

        # ## ----> Print score.
        print("Evaluation is finished.")
        frequency = collections.Counter(score)
        print(f"Result: {dict(frequency)}")
