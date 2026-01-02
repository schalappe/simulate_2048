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
import numpy as np
from tqdm import trange

from reinforce.game.core import encode_observation, is_done, legal_actions_mask, max_tile, next_state
from reinforce.game.env import reset
from reinforce.mcts.policy import select_action
from reinforce.mcts.stochastic_mctx import run_mcts_jit
from reinforce.neural.network import create_network
from reinforce.training.config import TrainConfig, small_config


def evaluate(
    checkpoint_path: str | None = None,
    num_games: int = 10,
    num_simulations: int = 50,
    temperature: float = 0.001,  # ##>: Use small value instead of 0.0 to avoid JAX tracing division-by-zero.
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
        Temperature for action selection (< 0.01 = greedy).
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

    # ##>: Create network with correct API.
    key = jax.random.PRNGKey(42)
    network = create_network(
        key=key,
        observation_shape=config.observation_shape,
        hidden_size=config.hidden_size,
        num_blocks=config.num_residual_blocks,
        num_actions=config.action_size,
        codebook_size=config.codebook_size,
    )
    params = network.params
    apply_fns = network.apply_fns

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
        state = reset(game_key)
        cumulative_reward = 0.0
        steps = 0

        # ##>: Play a game.
        while not bool(state.done):
            key, mcts_key, action_key, step_key = jax.random.split(key, 4)

            # ##>: Get observation and legal actions.
            obs = encode_observation(state.board)
            legal_mask = legal_actions_mask(state.board)

            # ##>: Run MCTS to get policy.
            mcts_output = run_mcts_jit(
                observation=obs,
                params=params,
                apply_fns=apply_fns,
                key=mcts_key,
                num_simulations=num_simulations,
                num_actions=config.action_size,
                codebook_size=config.codebook_size,
                discount=config.discount,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_fraction=config.dirichlet_fraction,
                pb_c_init=config.pb_c_init,
                pb_c_base=config.pb_c_base,
            )

            # ##>: Select action using policy output.
            action = select_action(mcts_output, action_key, legal_mask, temperature)
            action = int(action)

            # ##>: Step environment.
            new_board, reward = next_state(state.board, action, step_key)
            done = is_done(new_board)
            cumulative_reward += float(reward)
            steps += 1

            # ##>: Update state.
            state = state._replace(board=new_board, done=done)

            if progress_bar is not None:
                current_max = int(max_tile(state.board))
                progress_bar.set_description(f'Game {game_num + 1}/{num_games}')
                progress_bar.set_postfix(reward=int(cumulative_reward), max_tile=current_max)

        max_tiles.append(int(max_tile(state.board)))
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
    parser.add_argument(
        '--temperature', type=float, default=0.001, help='Action selection temperature (0.001 = greedy)'
    )
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
