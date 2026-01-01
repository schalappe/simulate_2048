"""
Evaluate reinforcement learning methods on the 2048 game.
"""

from collections import Counter

import numpy as np
from tqdm import trange

from reinforce.mcts import MonteCarloAgent
from twentyfortyeight.envs import TwentyFortyEight


def evaluate(method: str = 'mcts', length: int = 10) -> dict[int, int]:
    """
    Evaluate a reinforcement learning method.

    Parameters
    ----------
    method : str, optional
        The name of the method to evaluate (default is "mcts").
    length : int, optional
        The number of games to play (default is 10).

    Returns
    -------
    Dict[int, int]
        Frequency distribution of maximum tiles achieved across games.
    """
    if method != 'mcts':
        raise ValueError(f"Unknown method: {method}. Currently only 'mcts' is supported.")

    actor = MonteCarloAgent()
    env = TwentyFortyEight()
    score = []

    with trange(length) as period:
        for num in period:
            env.reset()
            rewards, done = 0, False

            # ##>: Play a game.
            while not done:
                # ##>: Interact with environment.
                _, reward, done = env.step(actor.choose_action(env.observation))
                rewards += reward

                # ##>: Log.
                period.set_description(f'Evaluation: {num + 1}')
                period.set_postfix(score=rewards, max=np.max(env.observation))

            # ##>: Save max cells.
            score.append(int(np.max(env.observation)))

    # ##>: Final log.
    frequency = Counter(score)
    return dict(frequency)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--method', type=str, default='mcts')
    parser.add_argument('--length', type=int, default=10)
    args = parser.parse_args()

    result = evaluate(method=args.method, length=args.length)
    print(f'Evaluation of method {args.method}, score: {result}')
