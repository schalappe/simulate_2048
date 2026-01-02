"""
Evaluate Stochastic MuZero on the 2048 game using JAX.

Usage:
    # Evaluate untrained model (random baseline)
    python -m reinforce.evaluate --games 10

    # Evaluate trained model from checkpoint
    python -m reinforce.evaluate --checkpoint checkpoints --games 20
"""

from collections import Counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from reinforce.game.core import create_initial_board, is_done, next_state
from reinforce.mcts.policy import select_action
from reinforce.mcts.stochastic_mctx import run_mcts
from reinforce.neural.network import create_network
from reinforce.training.config import TrainConfig, small_config


def evaluate(
    checkpoint_path: str | None = None,
    num_games: int = 10,
    num_simulations: int = 50,
    temperature: float = 0.0,
    config: TrainConfig | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Evaluate Stochastic MuZero agent.

    Parameters
    ----------
    checkpoint_path : str | None
        Path to checkpoint directory. If None, uses untrained network.
    num_games : int
        Number of games to play.
    num_simulations : int
        Number of MCTS simulations per move.
    temperature : float
        Temperature for action selection (0 = greedy).
    config : TrainConfig | None
        Training config. If None, uses small_config.
    verbose : bool
        Whether to show progress bar.

    Returns
    -------
    dict[str, Any]
        Evaluation results.
    """
    if config is None:
        config = small_config()

    # ##>: Create network.
    key = jax.random.PRNGKey(42)
    network, params = create_network(config, key)

    # ##>: Load checkpoint if provided.
    if checkpoint_path is not None:
        # ##@: Implement checkpoint loading once orbax integration is tested.
        print(f'Warning: Checkpoint loading from {checkpoint_path} not yet implemented')
        print('Using untrained network')

    max_tiles = []
    total_rewards = []
    episode_lengths = []

    progress_bar = trange(num_games) if verbose else None
    iterator = progress_bar if progress_bar is not None else range(num_games)

    for game_num in iterator:
        key, game_key = jax.random.split(key)
        state = create_initial_board(game_key)
        cumulative_reward = 0.0
        steps = 0

        # ##>: Play a game.
        while not is_done(state):
            key, action_key, step_key = jax.random.split(key, 3)

            # ##>: Run MCTS to get policy.
            mcts_output = run_mcts(
                params=params,
                network=network,
                state=state,
                rng_key=action_key,
                num_simulations=num_simulations,
                config=config,
            )

            # ##>: Select action.
            action = select_action(mcts_output.action_weights, action_key, temperature=temperature)
            action = int(action)

            # ##>: Step environment.
            new_state, reward = next_state(state, action, step_key)
            state = new_state
            cumulative_reward += float(reward)
            steps += 1

            if progress_bar is not None:
                max_tile = int(jnp.max(state))
                progress_bar.set_description(f'Game {game_num + 1}/{num_games}')
                progress_bar.set_postfix(reward=int(cumulative_reward), max_tile=max_tile)

        max_tiles.append(int(jnp.max(state)))
        total_rewards.append(cumulative_reward)
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
        'num_games': num_games,
    }

    return results


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Evaluate Stochastic MuZero for 2048')
    parser.add_argument('--games', type=int, default=10, help='Number of games')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--simulations', type=int, default=50, help='MCTS simulations per move')
    parser.add_argument('--temperature', type=float, default=0.0, help='Action selection temperature')
    args = parser.parse_args()

    result = evaluate(
        checkpoint_path=args.checkpoint,
        num_games=args.games,
        num_simulations=args.simulations,
        temperature=args.temperature,
    )

    print('\nEvaluation Results (Stochastic MuZero):')
    print(f'  Games played: {result["num_games"]}')
    print(f'  Mean reward: {result["mean_reward"]:.1f}')
    print(f'  Max reward: {result["max_reward"]:.1f}')
    print(f'  Mean max tile: {result["mean_max_tile"]:.1f}')
    print(f'  Best tile achieved: {result["max_tile"]}')
    print(f'  Mean game length: {result["mean_length"]:.1f}')
    print('\nTile frequency distribution:')
    for tile, count in sorted(result['tile_frequency'].items()):
        print(f'  {tile}: {count} games ({100 * count / result["num_games"]:.1f}%)')
