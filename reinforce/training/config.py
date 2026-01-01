"""
Configuration for Stochastic MuZero training.

Hyperparameters follow the paper "Planning in Stochastic Environments with a Learned Model"
(ICLR 2022), specifically Appendix C for 2048 experiments.
"""

from dataclasses import dataclass, field
from enum import Enum


class MCTSMode(str, Enum):
    """
    MCTS execution mode for self-play.

    SEQUENTIAL: Original MCTS with batched afterstate_dynamics (~4x speedup).
    BATCHED: Leaf batching for improved throughput (~10x speedup).
    THREADED: Parallel workers with virtual loss (~25-50x speedup).
    """

    SEQUENTIAL = 'sequential'
    BATCHED = 'batched'
    THREADED = 'threaded'


@dataclass
class StochasticMuZeroConfig:
    """
    Configuration for Stochastic MuZero training.

    Attributes are organized into sections following the paper's pseudocode.
    """

    # ##>: Environment parameters.
    observation_shape: tuple[int, ...] = (16,)  # Flattened 4x4 board
    action_size: int = 4  # left, up, right, down
    codebook_size: int = 32  # Number of chance codes (paper default)

    # ##>: Network architecture.
    hidden_size: int = 256  # Hidden state dimension
    num_residual_blocks: int = 10  # ResNet blocks per network

    # ##>: Self-play parameters.
    num_simulations: int = 100  # MCTS simulations per move
    discount: float = 0.999  # Reward discount factor

    # ##>: MCTS execution mode (for batched inference).
    mcts_mode: MCTSMode = MCTSMode.BATCHED  # Default to batched for ~10x speedup
    mcts_batch_size: int = 8  # Batch size for leaf/threaded modes
    mcts_num_workers: int = 4  # Number of workers for threaded mode

    # ##>: Exploration parameters.
    root_dirichlet_alpha: float = 0.25  # Dirichlet noise alpha (paper: 0.25)
    root_dirichlet_fraction: float = 0.1  # Noise fraction (paper: 0.1)
    exploration_weight: float = 1.25  # PUCT c_puct constant

    # ##>: Temperature schedule for action selection.
    # ##>: Format: [(step, temperature), ...]
    temperature_schedule: list[tuple[int, float]] = field(
        default_factory=lambda: [
            (0, 1.0),  # Steps 0-100k: temperature 1.0
            (100_000, 0.5),  # Steps 100k-200k: temperature 0.5
            (200_000, 0.1),  # Steps 200k-300k: temperature 0.1
            (300_000, 0.0),  # Steps 300k+: greedy
        ]
    )

    # ##>: Replay buffer parameters.
    replay_buffer_size: int = 125_000  # Max trajectories in buffer (paper)
    max_trajectory_length: int = 200  # Max steps per game

    # ##>: Training parameters.
    batch_size: int = 1024  # Training batch size (paper)
    num_unroll_steps: int = 5  # K steps to unroll model
    td_steps: int = 10  # n-step returns (paper)
    td_lambda: float = 0.5  # TD(λ) parameter (paper)

    # ##>: Prioritized replay (paper: α=1, β=1).
    priority_alpha: float = 1.0  # Priority exponent
    priority_beta: float = 1.0  # Importance sampling exponent

    # ##>: Optimization parameters.
    learning_rate: float = 3e-4  # Adam learning rate (paper)
    weight_decay: float = 0.0  # No weight decay for 2048 (paper)
    training_steps: int = 20_000_000  # Total training steps (paper)
    export_network_every: int = 1000  # Save checkpoint interval

    # ##>: Value scaling (paper: invertible transform for large rewards).
    # ##>: h(x) = sign(x) * (sqrt(|x| + 1) - 1 + ε*x), ε = 0.001
    value_epsilon: float = 0.001

    # ##>: Value/reward support for categorical representation.
    # ##>: 2048 scores can reach ~600k; paper uses 601 bins [0, 600].
    value_support_size: int = 601
    value_support_min: float = 0.0
    value_support_max: float = 600.0

    # ##>: VQ-VAE commitment loss weight (paper: β).
    commitment_loss_weight: float = 0.25

    def get_temperature(self, training_step: int) -> float:
        """
        Get the temperature for action selection at given training step.

        Parameters
        ----------
        training_step : int
            Current training step.

        Returns
        -------
        float
            Temperature value.
        """
        temperature = self.temperature_schedule[0][1]
        for step, temp in self.temperature_schedule:
            if training_step >= step:
                temperature = temp
        return temperature


def default_2048_config() -> StochasticMuZeroConfig:
    """
    Create default configuration for 2048.

    Returns
    -------
    StochasticMuZeroConfig
        Configuration matching paper's Appendix C.
    """
    return StochasticMuZeroConfig()


def small_2048_config() -> StochasticMuZeroConfig:
    """
    Create a smaller configuration for faster experimentation.

    Reduces network size and training parameters for quick testing.

    Returns
    -------
    StochasticMuZeroConfig
        Reduced configuration.
    """
    return StochasticMuZeroConfig(
        hidden_size=128,
        num_residual_blocks=5,
        num_simulations=50,
        replay_buffer_size=10_000,
        batch_size=256,
        training_steps=100_000,
    )
