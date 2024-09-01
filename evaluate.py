# -*- coding: utf-8 -*-
"""
Evaluate a method of reinforcement learning.
"""
from collections import Counter
from typing import Dict

import numpy as np
from tqdm import trange

from monte_carlo.actor import MonteCarloAgent
from simulate.envs import TwentyFortyEight

ACTORS = {
    "mcts": MonteCarloAgent,
}


def evaluate(method: str, length: int = 10) -> Dict[int, int]:
    """
    Evaluate a method of reinforcement learning.

    Parameters
    ----------
    method : str
        The name of the method to evaluate.
    length : int, optional
        The number of games to play (default is 100).

    Returns
    -------
    Dict[int, int]
        The evaluation results.
    """
    actor = ACTORS[method]()
    score = []

    with trange(length) as period:
        for num in period:
            env = TwentyFortyEight()

            # ##: Play a game.
            while not env.is_finished:
                # ##: Interact with environment
                env.step(actor.choose_action(env.observation))

                # ##: Log.
                period.set_description(f"Evaluation: {num + 1}")
                period.set_postfix(score=np.sum(env.observation), max=np.max(env.observation))

            # ##: Save max cells.
            score.append(int(np.max(env.observation)))

    # ##: Final log.
    frequency = Counter(score)
    return dict(frequency)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--method", type=str, default="mcts")
    args = parser.parse_args()

    result = evaluate(method=args.method)
    print(f"Evalation de la methode {args.method}, score: {result}")
