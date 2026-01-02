#!/usr/bin/env python
"""
Profile MCTS evaluation to identify performance bottlenecks.

Usage:
    uv run python scripts/profile_mcts.py
    uv run python scripts/profile_mcts.py --games 5 --simulations 50

After running, analyze with:
    uv run python -c "import pstats; p = pstats.Stats('profile.out'); p.sort_stats('cumulative').print_stats(30)"

Or visualize with snakeviz:
    uv run pip install snakeviz
    uv run snakeviz profile.out
"""

import cProfile
import pstats
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description='Profile MCTS evaluation')
    parser.add_argument('--games', type=int, default=3, help='Number of games to play')
    parser.add_argument('--simulations', type=int, default=50, help='MCTS simulations per move')
    parser.add_argument('--output', type=str, default='profile.out', help='Output file for profile data')
    parser.add_argument('--top', type=int, default=30, help='Number of top functions to show')
    args = parser.parse_args()

    # ##>: Import inside main to profile module loading separately.
    from reinforce.evaluate import evaluate

    print(f'Profiling MCTS: {args.games} games, {args.simulations} simulations/move')
    print(f'Output: {args.output}\n')

    # ##>: Run profiler.
    profiler = cProfile.Profile()
    profiler.enable()

    evaluate(method='mcts', length=args.games, num_simulations=args.simulations, verbose=True)

    profiler.disable()
    profiler.dump_stats(args.output)

    # ##>: Print summary.
    print(f'\n{"=" * 60}')
    print(f'Top {args.top} functions by cumulative time:')
    print('=' * 60)

    stats = pstats.Stats(args.output)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(args.top)

    print('=' * 60)
    print(f'Top {args.top} functions by total time (self):')
    print('=' * 60)

    stats.sort_stats('tottime')
    stats.print_stats(args.top)

    print(f'\nProfile saved to: {args.output}')
    print('Visualize with: uv run snakeviz profile.out')


if __name__ == '__main__':
    main()
