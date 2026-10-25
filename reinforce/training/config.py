"""
Configuration for JAX-based Stochastic MuZero training.

Hyperparameters follow the paper "Planning in Stochastic Environments with a Learned Model"
(ICLR 2022), specifically Appendix C for 2048 experiments.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """
    Immutable configuration for Stochastic MuZero training.

    Frozen dataclass ensures configuration cannot be accidentally modified
    during training, which is important for reproducibility.

    Attributes are organized into sections following the paper's pseudocode.
    """

    # ##>: Environment parameters.
    action_size: int = 4  # left, up, right, down
    codebook_size: int = 32  # Number of chance codes (paper default)

    # ##>: Network architecture.
    hidden_size: int = 256  # Hidden state dimension
    num_residual_blocks: int = 10  # ResNet blocks per network

    # ##>: MCTS parameters.
    num_simulations: int = 100  # MCTS simulations per move
    discount: float = 0.999  # Reward discount factor
    dirichlet_alpha: float = 0.25  # Dirichlet noise alpha
    dirichlet_fraction: float = 0.1  # Noise fraction at root
    pb_c_init: float = 1.25  # PUCT exploration constant
    pb_c_base: float = 19652.0  # PUCT base

    # ##>: Temperature schedule for action selection, as (training_step, temperature) pairs.
    temperature_schedule: tuple[tuple[int, float], ...] = (
        (0, 1.0),
        (100_000, 0.5),
        (200_000, 0.1),
        (300_000, 0.0),  # Greedy afterwards
    )

    # ##>: Self-play. Games run to their end; `max_game_moves` only guards against runaway games.
    num_parallel_games: int = 256  # Games played side by side on the device
    moves_per_generation: int = 50  # Moves per game in one compiled self-play chunk
    generation_interval: int = 10  # Training steps between self-play chunks
    max_game_moves: int = 100_000  # Safety limit; a game stopped here is truncated, not terminal

    # ##>: Replay buffer (paper: 125k sequences of up to 200 steps).
    replay_buffer_size: int = 125_000  # Max sequences in buffer
    sequence_length: int = 200  # Steps per stored sequence
    min_buffer_size: int = 1000  # Sequences required before training starts

    # ##>: Training parameters.
    batch_size: int = 1024  # Training batch size
    num_unroll_steps: int = 5  # K steps to unroll model
    td_steps: int = 10  # n-step returns
    td_lambda: float = 0.5  # TD(λ) parameter

    # ##>: Prioritized replay (paper: priority |search value - value target|, α=1, β=1).
    priority_alpha: float = 1.0  # Priority exponent
    priority_beta: float = 1.0  # Importance sampling exponent

    # ##>: Optimization parameters.
    learning_rate: float = 3e-4  # Adam learning rate
    max_grad_norm: float = 5.0  # Gradient clipping
    warmup_steps: int = 1000  # Learning rate warmup

    # ##>: Training schedule.
    training_steps: int = 20_000_000  # Total training steps
    checkpoint_interval: int = 1000  # Save checkpoint and replay buffer every N steps
    log_interval: int = 100  # Log metrics every N steps
    eval_interval: int = 10_000  # Evaluate every N steps
    eval_games: int = 16  # Number of games for evaluation

    # ##>: Loss weights.
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.25  # Also weights the afterstate value loss
    reward_loss_weight: float = 1.0
    chance_loss_weight: float = 1.0
    commitment_loss_weight: float = 0.25  # VQ-VAE commitment cost β

    # ##>: Random seed.
    seed: int = 42

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


def save_config(config: TrainConfig, path: Path) -> None:
    """
    Write a configuration as JSON, so evaluation can rebuild the network a run was trained with.

    Parameters
    ----------
    config : TrainConfig
        Configuration to save.
    path : Path
        Destination file.
    """
    path.write_text(json.dumps(asdict(config), indent=2))


def load_config(path: Path) -> TrainConfig:
    """
    Read a configuration written by `save_config`.

    Parameters
    ----------
    path : Path
        JSON file.

    Returns
    -------
    TrainConfig
        The saved configuration. Unknown fields raise a TypeError.
    """
    data = json.loads(path.read_text())
    data['temperature_schedule'] = tuple(tuple(pair) for pair in data['temperature_schedule'])
    return TrainConfig(**data)


def default_config() -> TrainConfig:
    """
    Create default configuration for 2048.

    Returns
    -------
    TrainConfig
        Configuration matching paper's Appendix C.
    """
    return TrainConfig()


def small_config() -> TrainConfig:
    """
    Create a smaller configuration for experiments on one GPU.

    Returns
    -------
    TrainConfig
        Reduced configuration.
    """
    return TrainConfig(
        hidden_size=128,
        num_residual_blocks=5,
        num_simulations=50,
        temperature_schedule=((0, 1.0), (10_000, 0.5), (20_000, 0.1), (30_000, 0.0)),
        num_parallel_games=64,
        replay_buffer_size=10_000,
        min_buffer_size=100,
        batch_size=256,
        training_steps=100_000,
        checkpoint_interval=1000,
        log_interval=50,
        eval_interval=2000,
        eval_games=8,
    )


def tiny_config() -> TrainConfig:
    """
    Create a tiny configuration for local debugging on a CPU.

    Returns
    -------
    TrainConfig
        Minimal configuration for testing.
    """
    return TrainConfig(
        hidden_size=32,
        num_residual_blocks=1,
        num_simulations=8,
        temperature_schedule=((0, 1.0), (100, 0.5), (200, 0.1), (300, 0.0)),
        num_parallel_games=4,
        moves_per_generation=20,
        generation_interval=5,
        replay_buffer_size=1_000,
        sequence_length=50,
        min_buffer_size=2,
        batch_size=32,
        warmup_steps=10,
        training_steps=1_000,
        checkpoint_interval=50,
        log_interval=10,
        eval_interval=500,
        eval_games=2,
    )
