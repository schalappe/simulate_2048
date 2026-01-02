"""
Benchmark script for training performance measurement.

Measures:
1. Batch sampling time
2. Training step time (with JIT warmup)
3. Full iteration time (sample + train)
4. Steps per second throughput

Run before and after optimization to measure improvements.

Usage:
    uv run python benchmark_training.py --baseline          # Run baseline only
    uv run python benchmark_training.py --optimized        # Run optimized only
    uv run python benchmark_training.py --compare          # Run both and compare
"""

import argparse
import time
from dataclasses import dataclass

import jax
import numpy as np
from tqdm import tqdm

from reinforce.training.config import tiny_config
from reinforce.training.learner import create_train_state
from reinforce.training.replay_buffer import ReplayBuffer, Trajectory


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""

    sample_times_ms: list[float]
    train_times_ms: list[float]
    iteration_times_ms: list[float]
    warmup_time_ms: float
    jit_compilation_time_ms: float

    @property
    def mean_sample_ms(self) -> float:
        return np.mean(self.sample_times_ms)

    @property
    def mean_train_ms(self) -> float:
        return np.mean(self.train_times_ms)

    @property
    def mean_iteration_ms(self) -> float:
        return np.mean(self.iteration_times_ms)

    @property
    def steps_per_second(self) -> float:
        return 1000.0 / self.mean_iteration_ms

    @property
    def std_sample_ms(self) -> float:
        return np.std(self.sample_times_ms)

    @property
    def std_train_ms(self) -> float:
        return np.std(self.train_times_ms)

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            '=' * 60,
            'TRAINING BENCHMARK RESULTS',
            '=' * 60,
            f'JIT Compilation Time:  {self.jit_compilation_time_ms:>8.1f} ms',
            f'Warmup Time:           {self.warmup_time_ms:>8.1f} ms',
            '-' * 60,
            'Per-Step Timing (after warmup):',
            f'  Batch Sampling:      {self.mean_sample_ms:>8.2f} ± {self.std_sample_ms:.2f} ms',
            f'  Training Step:       {self.mean_train_ms:>8.2f} ± {self.std_train_ms:.2f} ms',
            f'  Total Iteration:     {self.mean_iteration_ms:>8.2f} ms',
            '-' * 60,
            f'Throughput:            {self.steps_per_second:>8.1f} steps/second',
            '=' * 60,
        ]
        return '\n'.join(lines)


def generate_dummy_trajectories(buffer: ReplayBuffer, num_games: int, config) -> None:
    """Fill buffer with dummy trajectories for benchmarking."""
    for _ in range(num_games):
        traj_len = np.random.randint(50, 150)
        trajectory = Trajectory(
            observations=np.random.randn(traj_len, config.observation_shape[0]).astype(np.float32),
            actions=np.random.randint(0, config.action_size, size=traj_len),
            rewards=np.random.randn(traj_len).astype(np.float32) * 100,
            policies=np.random.dirichlet(np.ones(config.action_size), size=traj_len).astype(np.float32),
            values=np.random.randn(traj_len).astype(np.float32) * 1000,
            done=True,
            total_reward=float(np.random.randn() * 1000),
            max_tile=2 ** np.random.randint(4, 12),
        )
        buffer.add(trajectory)


def run_benchmark(
    num_warmup_steps: int = 10,
    num_benchmark_steps: int = 100,
    use_optimized: bool = False,
    batch_size: int | None = None,
) -> BenchmarkResults:
    """
    Run training benchmark.

    Parameters
    ----------
    num_warmup_steps : int
        Number of warmup steps (for JIT compilation).
    num_benchmark_steps : int
        Number of steps to benchmark.
    use_optimized : bool
        Whether to use optimized training functions.
    batch_size : int | None
        Override batch size. If None, uses config default.

    Returns
    -------
    BenchmarkResults
        Benchmark results.
    """
    # ##>: Use tiny config for faster benchmarking.
    config = tiny_config()

    if batch_size is not None:
        # ##>: Create new config with custom batch size.
        config = type(config)(**{**config.__dict__, 'batch_size': batch_size})

    print(f'Benchmark Configuration:')
    print(f'  Batch size:       {config.batch_size}')
    print(f'  Hidden size:      {config.hidden_size}')
    print(f'  Num blocks:       {config.num_residual_blocks}')
    print(f'  Unroll steps:     {config.num_unroll_steps}')
    print(f'  Warmup steps:     {num_warmup_steps}')
    print(f'  Benchmark steps:  {num_benchmark_steps}')
    print(f'  Using optimized:  {use_optimized}')
    print()

    # ##>: Initialize state and buffer.
    key = jax.random.PRNGKey(42)
    state = create_train_state(config, key)

    buffer = ReplayBuffer(config)
    print('Generating dummy trajectories...')
    generate_dummy_trajectories(buffer, config.min_buffer_size + 100, config)
    print(f'Buffer filled with {len(buffer)} trajectories')
    print()

    # ##>: Select training and sampling functions.
    if use_optimized:
        from reinforce.training.learner import train_step_optimized

        train_fn = train_step_optimized
        sample_fn = buffer.sample_batch_vectorized
        print('Using OPTIMIZED training function')
        print('Using VECTORIZED batch sampling')
    else:
        from reinforce.training.learner import train_step

        train_fn = train_step
        sample_fn = buffer.sample_batch
        print('Using BASELINE training function')
        print('Using LEGACY batch sampling')

    # ##>: JIT compilation / warmup phase.
    print(f'\nRunning {num_warmup_steps} warmup steps (JIT compilation)...')
    jit_start = time.perf_counter()

    batch, _ = sample_fn(config.batch_size)
    state, _ = train_fn(state, batch, config)
    jax.block_until_ready(state.network.params)

    jit_time_ms = (time.perf_counter() - jit_start) * 1000

    # ##>: Additional warmup steps.
    warmup_start = time.perf_counter()
    for _ in range(num_warmup_steps - 1):
        batch, _ = sample_fn(config.batch_size)
        state, _ = train_fn(state, batch, config)
    jax.block_until_ready(state.network.params)
    warmup_time_ms = (time.perf_counter() - warmup_start) * 1000

    print(f'JIT compilation: {jit_time_ms:.1f} ms')
    print(f'Warmup ({num_warmup_steps - 1} steps): {warmup_time_ms:.1f} ms')

    # ##>: Benchmark phase.
    print(f'\nRunning {num_benchmark_steps} benchmark steps...')
    sample_times = []
    train_times = []
    iteration_times = []

    for _ in tqdm(range(num_benchmark_steps), desc='Benchmarking', unit='step'):
        iter_start = time.perf_counter()

        # ##>: Measure sampling time.
        sample_start = time.perf_counter()
        batch, _ = sample_fn(config.batch_size)
        sample_end = time.perf_counter()

        # ##>: Measure training time.
        train_start = time.perf_counter()
        state, _ = train_fn(state, batch, config)
        jax.block_until_ready(state.network.params)
        train_end = time.perf_counter()

        iter_end = time.perf_counter()

        sample_times.append((sample_end - sample_start) * 1000)
        train_times.append((train_end - train_start) * 1000)
        iteration_times.append((iter_end - iter_start) * 1000)

    return BenchmarkResults(
        sample_times_ms=sample_times,
        train_times_ms=train_times,
        iteration_times_ms=iteration_times,
        warmup_time_ms=warmup_time_ms,
        jit_compilation_time_ms=jit_time_ms,
    )


def compare_benchmarks(baseline: BenchmarkResults, optimized: BenchmarkResults) -> str:
    """Generate comparison report between baseline and optimized."""
    speedup_sample = baseline.mean_sample_ms / optimized.mean_sample_ms
    speedup_train = baseline.mean_train_ms / optimized.mean_train_ms
    speedup_iter = baseline.mean_iteration_ms / optimized.mean_iteration_ms
    speedup_throughput = optimized.steps_per_second / baseline.steps_per_second

    lines = [
        '=' * 60,
        'OPTIMIZATION COMPARISON',
        '=' * 60,
        '',
        'Metric                  Baseline    Optimized   Speedup',
        '-' * 60,
        f'Batch Sampling (ms)    {baseline.mean_sample_ms:>8.2f}    {optimized.mean_sample_ms:>8.2f}    {speedup_sample:>5.2f}x',
        f'Training Step (ms)     {baseline.mean_train_ms:>8.2f}    {optimized.mean_train_ms:>8.2f}    {speedup_train:>5.2f}x',
        f'Total Iteration (ms)   {baseline.mean_iteration_ms:>8.2f}    {optimized.mean_iteration_ms:>8.2f}    {speedup_iter:>5.2f}x',
        f'Throughput (steps/s)   {baseline.steps_per_second:>8.1f}    {optimized.steps_per_second:>8.1f}    {speedup_throughput:>5.2f}x',
        '-' * 60,
        '',
        f'JIT Compile (ms)       {baseline.jit_compilation_time_ms:>8.1f}    {optimized.jit_compilation_time_ms:>8.1f}',
        '',
        '=' * 60,
        f'OVERALL SPEEDUP: {speedup_iter:.2f}x',
        '=' * 60,
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Benchmark training performance')
    parser.add_argument('--warmup-steps', type=int, default=10, help='Number of warmup steps')
    parser.add_argument('--benchmark-steps', type=int, default=100, help='Number of benchmark steps')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--optimized', action='store_true', help='Run optimized version only')
    parser.add_argument('--baseline', action='store_true', help='Run baseline version only')
    parser.add_argument('--compare', action='store_true', help='Run both and compare')

    args = parser.parse_args()

    if args.compare or (not args.optimized and not args.baseline):
        print('\n' + '=' * 60)
        print('BASELINE BENCHMARK')
        print('=' * 60 + '\n')
        baseline = run_benchmark(
            num_warmup_steps=args.warmup_steps,
            num_benchmark_steps=args.benchmark_steps,
            use_optimized=False,
            batch_size=args.batch_size,
        )
        print('\n' + baseline.summary())

        print('\n' + '=' * 60)
        print('OPTIMIZED BENCHMARK')
        print('=' * 60 + '\n')
        optimized = run_benchmark(
            num_warmup_steps=args.warmup_steps,
            num_benchmark_steps=args.benchmark_steps,
            use_optimized=True,
            batch_size=args.batch_size,
        )
        print('\n' + optimized.summary())

        print('\n' + compare_benchmarks(baseline, optimized))

    elif args.optimized:
        results = run_benchmark(
            num_warmup_steps=args.warmup_steps,
            num_benchmark_steps=args.benchmark_steps,
            use_optimized=True,
            batch_size=args.batch_size,
        )
        print('\n' + results.summary())

    else:
        results = run_benchmark(
            num_warmup_steps=args.warmup_steps,
            num_benchmark_steps=args.benchmark_steps,
            use_optimized=False,
            batch_size=args.batch_size,
        )
        print('\n' + results.summary())


if __name__ == '__main__':
    main()
