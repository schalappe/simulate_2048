"""
Evaluate reinforcement learning methods on the 2048 game.

Supports evaluation of:
- mcts: Basic MCTS without neural network guidance
- stochastic_muzero: Full Stochastic MuZero with learned model
"""

from collections import Counter
from typing import Any

import numpy as np
from tqdm import trange

from reinforce.mcts import MonteCarloAgent
from reinforce.mcts.stochastic_agent import StochasticMuZeroAgent
from twentyfortyeight.envs.twentyfortyeight import TwentyFortyEight


def evaluate(
    method: str = 'mcts',
    length: int = 10,
    checkpoint_path: str | None = None,
    num_simulations: int = 100,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a reinforcement learning method.

    Parameters
    ----------
    method : str
        The method to evaluate. Options: 'mcts', 'stochastic_muzero'.
    length : int
        The number of games to play.
    checkpoint_path : str | None
        Path to checkpoint directory (for stochastic_muzero).
        If None for stochastic_muzero, uses untrained network.
    num_simulations : int
        Number of MCTS simulations per move.
    verbose : bool
        Whether to show progress bar.

    Returns
    -------
    dict[str, any]
        Evaluation results including tile frequency, mean/max rewards.
    """
    # ##>: Create agent based on method.
    if method == 'mcts':
        agent = MonteCarloAgent(iterations=num_simulations)
        env = TwentyFortyEight()
    elif method == 'stochastic_muzero':
        if checkpoint_path is not None:
            agent = StochasticMuZeroAgent.from_checkpoint(
                checkpoint_path,
                num_simulations=num_simulations,
                temperature=0.0,  # Greedy for evaluation
                add_noise=False,
            )
        else:
            # ##>: Use untrained network for testing.
            agent = StochasticMuZeroAgent.create_untrained(
                num_simulations=num_simulations,
                temperature=0.0,
                add_noise=False,
            )
        env = TwentyFortyEight()
    else:
        raise ValueError(f"Unknown method: {method}. Options: 'mcts', 'stochastic_muzero'")

    max_tiles = []
    total_rewards = []
    episode_lengths = []

    progress_bar = trange(length) if verbose else None
    iterator = progress_bar if progress_bar is not None else range(length)

    for num in iterator:
        state = env.reset()
        rewards, done = 0.0, False
        steps = 0

        # ##>: Play a game.
        while not done:
            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            rewards += reward
            steps += 1

            if progress_bar is not None:
                progress_bar.set_description(f'Game {num + 1}/{length}')
                progress_bar.set_postfix(reward=int(rewards), max_tile=int(np.max(state)))

        max_tiles.append(int(np.max(state)))
        total_rewards.append(rewards)
        episode_lengths.append(steps)

    # ##>: Compile results.
    tile_frequency = dict(Counter(max_tiles))
    results = {
        'tile_frequency': tile_frequency,
        'mean_reward': float(np.mean(total_rewards)),
        'max_reward': float(np.max(total_rewards)),
        'mean_max_tile': float(np.mean(max_tiles)),
        'max_tile': int(np.max(max_tiles)),
        'mean_length': float(np.mean(episode_lengths)),
        'num_games': length,
        'method': method,
    }

    return results


def evaluate_checkpoint(
    checkpoint_path: str,
    num_games: int = 100,
    num_simulations: int = 100,
) -> dict[str, Any]:
    """
    Evaluate a trained Stochastic MuZero checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint directory.
    num_games : int
        Number of games to play.
    num_simulations : int
        MCTS simulations per move.

    Returns
    -------
    dict[str, any]
        Evaluation results.
    """
    return evaluate(
        method='stochastic_muzero',
        length=num_games,
        checkpoint_path=checkpoint_path,
        num_simulations=num_simulations,
    )


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Evaluate 2048 agents')
    parser.add_argument('--method', type=str, default='mcts', choices=['mcts', 'stochastic_muzero'])
    parser.add_argument('--length', type=int, default=10, help='Number of games')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path (for stochastic_muzero)')
    parser.add_argument('--simulations', type=int, default=100, help='MCTS simulations per move')
    args = parser.parse_args()

    result = evaluate(
        method=args.method,
        length=args.length,
        checkpoint_path=args.checkpoint,
        num_simulations=args.simulations,
    )

    print(f'\nEvaluation Results ({args.method}):')
    print(f'  Games played: {result["num_games"]}')
    print(f'  Mean reward: {result["mean_reward"]:.1f}')
    print(f'  Max reward: {result["max_reward"]:.1f}')
    print(f'  Mean max tile: {result["mean_max_tile"]:.1f}')
    print(f'  Best tile achieved: {result["max_tile"]}')
    print(f'  Mean game length: {result["mean_length"]:.1f}')
    print('\nTile frequency distribution:')
    for tile, count in sorted(result['tile_frequency'].items()):
        print(f'  {tile}: {count} games ({100 * count / result["num_games"]:.1f}%)')
