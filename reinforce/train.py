#!/usr/bin/env python
"""
Train Stochastic MuZero to play 2048 using JAX.

Training requires a GPU and stops at startup when JAX cannot see one, because JAX otherwise falls back
to the CPU silently and a paid GPU pod would sit idle. Pass `--allow-cpu` for local runs.

Rerunning with the same `--run-dir` resumes the run: network, optimizer, step, and replay buffer.

Usage:
    # Quick test with tiny config on a laptop CPU (debugging)
    python -m reinforce.train --mode tiny --steps 100 --run-dir /tmp/run --allow-cpu

    # Full training (paper configuration) on a GPU pod
    python -m reinforce.train --mode full --run-dir /workspace/runs/full
"""

import sys
from argparse import ArgumentParser

import jax

from reinforce.neural.network import count_parameters
from reinforce.training.config import default_config, small_config, tiny_config
from reinforce.training.trainer import Trainer

CONFIGS = {'tiny': tiny_config, 'small': small_config, 'full': default_config}


def require_gpu(allow_cpu: bool) -> None:
    """
    Stop the process unless JAX runs on a GPU.

    Parameters
    ----------
    allow_cpu : bool
        If True, accept the CPU backend (local debugging runs).
    """
    backend = jax.default_backend()
    print(f'JAX backend: {backend}, devices: {jax.devices()}')
    if backend != 'gpu' and not allow_cpu:
        sys.exit(
            f'No GPU visible to JAX (backend: {backend}). Install the CUDA wheels with `uv sync --extra gpu`. '
            'On images that set LD_LIBRARY_PATH to their own CUDA (RunPod PyTorch images), run '
            '`unset LD_LIBRARY_PATH` first. Pass --allow-cpu for a local run.'
        )


def main() -> None:
    """Run or resume Stochastic MuZero training."""
    parser = ArgumentParser(description='Train Stochastic MuZero for 2048 (JAX)')
    parser.add_argument(
        '--mode',
        default='small',
        choices=sorted(CONFIGS),
        help='Configuration: tiny (debugging), small (experiments), or full (paper)',
    )
    parser.add_argument('--steps', type=int, default=None, help='Training steps to run (default: up to the preset)')
    parser.add_argument('--run-dir', default='runs/default', help='Run directory: checkpoints, buffer, metrics')
    parser.add_argument('--allow-cpu', action='store_true', help='Allow training without a GPU (local runs)')
    args = parser.parse_args()

    require_gpu(args.allow_cpu)
    config = CONFIGS[args.mode]()
    print(f'Configuration: {args.mode}')

    trainer = Trainer(config=config, run_dir=args.run_dir)
    trainer.initialize()
    print(f'Network parameters: {count_parameters(trainer.state.params):,}')

    num_steps = args.steps if args.steps is not None else max(0, config.training_steps - trainer.step)
    print(f'Training for {num_steps} steps from step {trainer.step}; run directory: {args.run_dir}')
    results = trainer.train(num_steps)
    print(
        f'Training done: {results["steps"]} steps in {results["seconds"]:.1f}s '
        f'({results["steps_per_second"]:.2f} steps/s)'
    )

    # ##>: train() already evaluated if it stopped on an evaluation step.
    if trainer.step % config.eval_interval != 0:
        trainer.evaluate()
    trainer.close()


if __name__ == '__main__':
    main()
