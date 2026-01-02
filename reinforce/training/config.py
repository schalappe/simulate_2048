"""
Configuration for JAX-based Stochastic MuZero training.

Hyperparameters follow the paper "Planning in Stochastic Environments with a Learned Model"
(ICLR 2022), specifically Appendix C for 2048 experiments.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """
    Immutable configuration for Stochastic MuZero training.

    Frozen dataclass ensures configuration cannot be accidentally modified
    during training, which is important for reproducibility.

    Attributes are organized into sections following the paper's pseudocode.
    """

    # ##>: Environment parameters.
    observation_shape: tuple[int, ...] = (16,)  # Flattened 4x4 board
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

    # ##>: Temperature schedule for action selection.
    # ##>: Format: [(step, temperature), ...]
    temperature_schedule: tuple[tuple[int, float], ...] = (
        (0, 1.0),  # Steps 0-100k: temperature 1.0
        (100_000, 0.5),  # Steps 100k-200k: temperature 0.5
        (200_000, 0.1),  # Steps 200k-300k: temperature 0.1
        (300_000, 0.0),  # Steps 300k+: greedy
    )

    # ##>: Replay buffer parameters.
    replay_buffer_size: int = 125_000  # Max trajectories in buffer
    min_buffer_size: int = 1000  # Minimum before training starts
    max_trajectory_length: int = 200  # Max steps per game

    # ##>: Training parameters.
    batch_size: int = 1024  # Training batch size
    num_unroll_steps: int = 5  # K steps to unroll model
    td_steps: int = 10  # n-step returns
    td_lambda: float = 0.5  # TD(λ) parameter

    # ##>: Prioritized replay (paper: α=1, β=1).
    priority_alpha: float = 1.0  # Priority exponent
    priority_beta: float = 1.0  # Importance sampling exponent

    # ##>: Optimization parameters.
    learning_rate: float = 3e-4  # Adam learning rate
    weight_decay: float = 0.0  # L2 regularization
    max_grad_norm: float = 5.0  # Gradient clipping
    warmup_steps: int = 1000  # Learning rate warmup

    # ##>: Training schedule.
    training_steps: int = 20_000_000  # Total training steps
    games_per_iteration: int = 1  # Games to generate per training iteration
    train_steps_per_iteration: int = 1  # Training steps per iteration
    checkpoint_interval: int = 1000  # Save checkpoint every N steps
    log_interval: int = 100  # Log metrics every N steps
    eval_interval: int = 1000  # Evaluate every N steps
    eval_games: int = 10  # Number of games for evaluation

    # ##>: Parallel self-play parameters.
    num_parallel_games: int = 8  # Number of games to run in parallel
    generation_interval: int = 100  # Generate games every N training steps

    # ##>: Value scaling (paper: invertible transform for large rewards).
    # ##>: h(x) = sign(x) * (sqrt(|x| + 1) - 1 + ε*x), ε = 0.001
    value_epsilon: float = 0.001

    # ##>: Loss weights.
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.25
    reward_loss_weight: float = 1.0
    chance_loss_weight: float = 1.0
    commitment_loss_weight: float = 0.25  # VQ-VAE commitment loss

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
    Create a smaller configuration for faster experimentation.

    Reduces network size and training parameters for quick testing.

    Returns
    -------
    TrainConfig
        Reduced configuration.
    """
    return TrainConfig(
        hidden_size=128,
        num_residual_blocks=5,
        num_simulations=50,
        replay_buffer_size=10_000,
        min_buffer_size=100,
        batch_size=256,
        training_steps=100_000,
        checkpoint_interval=100,
        log_interval=10,
        eval_interval=100,
        num_parallel_games=4,
        generation_interval=50,
    )


def tiny_config() -> TrainConfig:
    """
    Create a tiny configuration for debugging.

    Returns
    -------
    TrainConfig
        Minimal configuration for testing.
    """
    return TrainConfig(
        hidden_size=64,
        num_residual_blocks=2,
        num_simulations=10,
        replay_buffer_size=1_000,
        min_buffer_size=10,
        batch_size=32,
        training_steps=1_000,
        checkpoint_interval=100,
        log_interval=1,
        eval_interval=50,
        eval_games=2,
        num_parallel_games=2,
        generation_interval=20,
    )
