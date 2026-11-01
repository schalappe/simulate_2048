#!/usr/bin/env python
"""
Evaluate a Stochastic MuZero agent on 2048 with greedy search and no root noise.

With `--run-dir`, the network and configuration come from that training run (for example a run directory
copied back from a RunPod volume). Without it, the agent is an untrained network, useful only as a smoke
check of the search.

Usage:
    python -m reinforce.evaluate --run-dir runs/full --games 20
    python -m reinforce.evaluate --mode tiny --games 1 --simulations 8
"""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
from pathlib import Path

import jax

from reinforce.training.config import (
    TrainConfig,
    budget_config,
    default_config,
    load_config,
    small_config,
    tiny_config,
)
from reinforce.training.learner import CheckpointManager, create_train_state
from reinforce.training.self_play import evaluate_games, summarize_games

CONFIGS = {'tiny': tiny_config, 'small': small_config, 'budget': budget_config, 'full': default_config}


def evaluate(
    config: TrainConfig,
    num_games: int,
    run_dir: Path | None = None,
    step: int | None = None,
    seed: int = 0,
) -> dict:
    """
    Play greedy evaluation games.

    Parameters
    ----------
    config : TrainConfig
        Configuration; must match the run when `run_dir` is given.
    num_games : int
        Number of games, played side by side.
    run_dir : Path | None
        Training run to load. None evaluates an untrained network.
    step : int | None
        Checkpoint step to load; the latest if None.
    seed : int
        Random seed for the games.

    Returns
    -------
    dict
        Output of `summarize_games`, plus the loaded step and the tile frequencies.

    Raises
    ------
    FileNotFoundError
        If `run_dir` holds no checkpoint.
    """
    loaded_step = None
    if run_dir is None:
        params = create_train_state(config, jax.random.PRNGKey(seed)).params
    else:
        manager = CheckpointManager(run_dir / 'checkpoints')
        state = manager.restore(config, step)
        manager.close()
        if state is None:
            raise FileNotFoundError(f'No checkpoint in {run_dir / "checkpoints"}')
        params = state.params
        loaded_step = int(state.step)

    games = evaluate_games(params, jax.random.PRNGKey(seed), config, num_games)
    return {
        'step': loaded_step,
        **summarize_games(games),
        'tile_frequency': dict(sorted(Counter(game.max_tile for game in games).items())),
    }


def main() -> None:
    """Parse arguments, evaluate, and print the results."""
    parser = ArgumentParser(description='Evaluate Stochastic MuZero for 2048')
    parser.add_argument('--run-dir', type=Path, default=None, help='Training run directory to load')
    parser.add_argument('--step', type=int, default=None, help='Checkpoint step (default: latest)')
    parser.add_argument(
        '--mode', default='small', choices=sorted(CONFIGS), help='Configuration when no --run-dir is given'
    )
    parser.add_argument('--games', type=int, default=10, help='Number of games')
    parser.add_argument('--simulations', type=int, default=None, help='MCTS simulations per move (override)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    config = load_config(args.run_dir / 'config.json') if args.run_dir is not None else CONFIGS[args.mode]()
    if args.simulations is not None:
        config = replace(config, num_simulations=args.simulations)

    if args.run_dir is None:
        print('No --run-dir: evaluating an untrained network.')
    result = evaluate(config, args.games, args.run_dir, args.step, args.seed)

    print('\nEvaluation Results (Stochastic MuZero):')
    if result['step'] is not None:
        print(f'  Checkpoint step: {result["step"]}')
    print(f'  Games played: {result["games"]}')
    print(f'  Mean score: {result["mean_score"]:.1f}')
    print(f'  Max score: {result["max_score"]:.1f}')
    print(f'  Mean max tile: {result["mean_max_tile"]:.1f}')
    print(f'  Best tile achieved: {result["max_tile"]}')
    print(f'  Mean game length: {result["mean_length"]:.1f}')
    print(f'  Truncated games: {result["truncated"]}')
    print('\nTile frequency distribution:')
    for tile, count in result['tile_frequency'].items():
        print(f'  {tile}: {count} games ({100 * count / result["games"]:.1f}%)')


if __name__ == '__main__':
    main()
