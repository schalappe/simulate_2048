#!/usr/bin/env python
"""
Train Stochastic MuZero to play 2048.

Usage:
    # Quick test with small config (debugging)
    python train.py --mode small --steps 1000

    # Full training (paper configuration)
    python train.py --mode full --steps 1000000

    # Resume from checkpoint
    python train.py --resume checkpoints/step_10000
"""

from argparse import ArgumentParser

from reinforce.training import (
    StochasticMuZeroTrainer,
    default_2048_config,
    small_2048_config,
)


def main():
    """Run Stochastic MuZero training."""
    parser = ArgumentParser(description='Train Stochastic MuZero for 2048')
    parser.add_argument(
        '--mode',
        type=str,
        default='small',
        choices=['small', 'full'],
        help='Training mode: small (for testing) or full (paper config)',
    )
    parser.add_argument('--steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    parser.add_argument('--log-interval', type=int, default=100, help='Steps between logging')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Steps between evaluation')
    parser.add_argument('--eval-games', type=int, default=5, help='Games per evaluation')
    args = parser.parse_args()

    # ##>: Create configuration.
    if args.mode == 'small':
        config = small_2048_config()
        print('Using small configuration (for testing)')
    else:
        config = default_2048_config()
        print('Using full paper configuration')

    # ##>: Create trainer.
    trainer = StochasticMuZeroTrainer(config=config, checkpoint_dir=args.checkpoint_dir)

    # ##>: Resume from checkpoint if specified.
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        trainer.load_checkpoint(args.resume)

    # ##>: Define evaluation callback.
    def eval_callback(step: int, _metrics: dict) -> None:
        if step > 0 and step % args.eval_interval == 0:
            eval_results = trainer.evaluate(num_games=args.eval_games)
            print(
                f'[Eval @ {step}] Mean Reward: {eval_results["mean_reward"]:.1f}, Max Tile: {eval_results["max_tile"]}'
            )

    # ##>: Run training.
    print(f'\nStarting training for {args.steps} steps...')
    print(f'Checkpoints will be saved to: {args.checkpoint_dir}')
    print()

    trainer.train(
        num_steps=args.steps,
        log_interval=args.log_interval,
        callback=eval_callback,
    )

    # ##>: Final evaluation.
    print('\n' + '=' * 50)
    print('Final Evaluation (20 games)')
    print('=' * 50)
    final_eval = trainer.evaluate(num_games=20)
    print(f'Mean Reward: {final_eval["mean_reward"]:.1f}')
    print(f'Max Reward: {final_eval["max_reward"]:.1f}')
    print(f'Mean Max Tile: {final_eval["mean_max_tile"]:.1f}')
    print(f'Best Tile Achieved: {final_eval["max_tile"]}')


if __name__ == '__main__':
    main()
