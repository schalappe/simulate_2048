import timeit
import numpy as np
from typing import Callable, List
from simulate.utils import slide_and_merge, illegal_actions


generator = np.random.default_rng(42)


def generate_random_board(size: int = 4) -> np.ndarray:
    """Generate a random 2048 game board."""
    board = np.zeros((size, size), dtype=np.int64)
    num_tiles = generator.integers(1, size * size + 1)
    tile_values = generator.choice([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], size=num_tiles)
    indices = generator.choice(size * size, size=num_tiles, replace=False)
    board.flat[indices] = tile_values
    return board


def time_function(func: Callable, args: List, number: int = 1000, repeat: int = 5) -> float:
    """Time the execution of a function."""
    timer = timeit.Timer(lambda: func(*args))
    times = timer.repeat(repeat=repeat, number=number)
    return min(times) / number * 1000


def benchmark_slide_and_merge(board_size: int = 4, num_boards: int = 100):
    """Benchmark the slide_and_merge function."""
    boards = [generate_random_board(board_size) for _ in range(num_boards)]
    total_time = sum(time_function(slide_and_merge, [board]) for board in boards)
    avg_time = total_time / num_boards
    print(f"Average time for slide_and_merge (board size {board_size}x{board_size}): {avg_time:.3f} ms")


def benchmark_illegal_actions(board_size: int = 4, num_boards: int = 100):
    """Benchmark the illegal_actions function."""
    boards = [generate_random_board(board_size) for _ in range(num_boards)]
    total_time = sum(time_function(illegal_actions, [board]) for board in boards)
    avg_time = total_time / num_boards
    print(f"Average time for illegal_actions (board size {board_size}x{board_size}): {avg_time:.3f} ms")


if __name__ == "__main__":
    print("Benchmarking slide_and_merge:")
    for size in [4, 6, 8]:
        benchmark_slide_and_merge(board_size=size)

    print("\nBenchmarking illegal_actions:")
    for size in [4, 6, 8]:
        benchmark_illegal_actions(board_size=size)
