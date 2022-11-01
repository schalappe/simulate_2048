# -*- coding: utf-8 -*-
"""
Script for evaluate an agent.
"""
from os.path import abspath, dirname, join

import gym
from module import AgentA2C, AgentDQN

from simulate_2048 import FlattenOneHotObservation

STORAGE_MODEL = join(dirname(dirname(abspath(__file__))), "zoo")

if __name__ == "__main__":
    import argparse
    import collections
    from itertools import count

    from numpy import max as max_array_values
    from numpy import sum as sum_array_values

    # ## ----> Get arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="Which algorithm to use", required=True, type=str)
    parser.add_argument("--model", help="Which type of model tu use", required=True, type=str)
    parser.add_argument("--style", required=False, type=str, default="simple")
    args = parser.parse_args()

    # ## ----> Evaluation with specific algorithm.
    if args.algo == "dqn":
        # ## ----> Create agent.
        model_path = join(STORAGE_MODEL, f"model_{args.style}-{args.algo}-{args.model}")
        agent = AgentDQN(model_path)
    elif args.algo == "a2c":
        model_path = join(STORAGE_MODEL, f"model_{args.algo}")
        agent = AgentA2C(model_path)

    # ## ----> Create environment.
    game = FlattenOneHotObservation(gym.make("GameBoard", size=4))

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
            print(f"Game: {i + 1} - Score: {sum_array_values(game.board)} - move: {t}", end="\r")
            if done or t > 5000:
                score.append(max_array_values(game.board))
                break

        print(f"Game: {i + 1} is finished, score egal {score[-1]}.")

        # ## ----> Print score.
    print("Evaluation is finished.")
    frequency = collections.Counter(score)
    print(f"Result: {dict(frequency)}")
