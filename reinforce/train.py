#!/usr/bin/env python
"""
Train Stochastic MuZero to play 2048 using JAX.

Usage:
    # Quick test with tiny config (debugging)
    python train.py --mode tiny --steps 100

    # Small config for experimentation
    python train.py --mode small --steps 10000

    # Full training (paper configuration)
    python train.py --mode full --steps 1000000
"""

from argparse import ArgumentParser

from reinforce.training.config import default_config, small_config, tiny_config
from reinforce.training.trainer import Trainer


def main():
    """Run Stochastic MuZero training with JAX."""
    parser = ArgumentParser(description='Train Stochastic MuZero for 2048 (JAX)')
    parser.add_argument(
        '--mode',
        type=str,
        default='small',
        choices=['tiny', 'small', 'full'],
        help='Training mode: tiny (debugging), small (experimentation), or full (paper config)',
    )
    parser.add_argument('--steps', type=int, default=None, help='Number of training steps (overrides mode default)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    args = parser.parse_args()

    # ##>: Create configuration based on mode.
    if args.mode == 'tiny':
        config = tiny_config()
        print('Using tiny configuration (for debugging)')
    elif args.mode == 'small':
        config = small_config()
        print('Using small configuration (for experimentation)')
    else:
        config = default_config()
        print('Using full paper configuration')

    # ##>: Create trainer.
    trainer = Trainer(config=config, checkpoint_dir=args.checkpoint_dir)

    # ##>: Initialize training state.
    trainer.initialize(seed=args.seed)

    # ##>: Fill buffer with initial games.
    print('\nGenerating initial games for replay buffer...')
    trainer.fill_buffer(show_progress=not args.no_progress)

    # ##>: Determine number of steps.
    num_steps = args.steps if args.steps is not None else config.training_steps

    # ##>: Run training.
    print(f'\nStarting training for {num_steps} steps...')
    print(f'Checkpoints will be saved to: {args.checkpoint_dir}')
    print()

    results = trainer.train(num_steps=num_steps, show_progress=not args.no_progress)

    # ##>: Print training summary.
    print('\n' + '=' * 50)
    print('Training Complete')
    print('=' * 50)
    print(f'Total steps: {results["total_steps"]}')
    print(f'Total time: {results["total_time_seconds"]:.1f}s')
    print(f'Steps/second: {results["steps_per_second"]:.2f}')

    # ##>: Final evaluation.
    print('\n' + '=' * 50)
    print('Final Evaluation (10 games)')
    print('=' * 50)
    final_eval = trainer.evaluate(num_games=10)
    print(f'Mean Reward: {final_eval["mean_reward"]:.1f}')
    print(f'Max Reward: {final_eval["max_reward"]:.1f}')
    print(f'Mean Max Tile: {final_eval["mean_max_tile"]:.1f}')
    print(f'Best Tile Achieved: {final_eval["max_tile"]}')


if __name__ == '__main__':
    main()
