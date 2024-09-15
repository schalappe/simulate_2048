# -*- coding: utf-8 -*-
"""
Evaluate a method of reinforcement learning.
"""
from collections import Counter
from typing import Dict

import numpy as np
from tqdm import trange

from monte_carlo.actor import MonteCarloAgent
from dqn.agent import DQNActor
from simulate.envs import TwentyFortyEight
from simulate.wrappers import EncodedTwentyFortyEight
from pathlib import Path

ACTORS = {
    "mcts": MonteCarloAgent,
    "deep_q": DQNActor,
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
    if method == "mcts":
        actor = ACTORS[method]()
        env = TwentyFortyEight()
    else:
        env = EncodedTwentyFortyEight()
        storage = Path.cwd() / "zoo"
        model_path = list(storage.glob(f"{method}*prioritized_identity*.keras"))
        actor = ACTORS[method].from_path(path=model_path[0], epsilon=0.05)
    score = []

    with trange(length) as period:
        for num in period:
            env.reset()
            rewards, done = 0, False

            # ##: Play a game.
            while not done:
                # ##: Interact with environment
                _, reward, done = env.step(actor.choose_action(env.observation))
                rewards += reward

                # ##: Log.
                period.set_description(f"Evaluation: {num + 1}")
                period.set_postfix(score=rewards, max=np.max(env.board))

            # ##: Save max cells.
            score.append(int(np.max(env.board)))

    # ##: Final log.
    frequency = Counter(score)
    return dict(frequency)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--method", type=str, default="mcts")
    args = parser.parse_args()

    result = evaluate(method=args.method)
    print(f"Evaluation de la methode {args.method}, score: {result}")
