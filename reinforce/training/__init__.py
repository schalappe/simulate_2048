"""
Training infrastructure for Stochastic MuZero.

This package provides the complete training system:
- Config: Hyperparameters for 2048 training
- ReplayBuffer: Experience storage with prioritized sampling
- Losses: Policy, value, reward, and chance losses
- Targets: TD(λ) n-step return computation
- SelfPlay: Data generation using MCTS
- Learner: Network optimization with model unrolling
- Trainer: Main training loop orchestrator
"""

from .config import StochasticMuZeroConfig, default_2048_config, small_2048_config
from .learner import StochasticMuZeroLearner
from .losses import (
    compute_chance_loss,
    compute_policy_loss,
    compute_reward_loss,
    compute_total_loss,
    compute_value_loss,
)
from .replay_buffer import ReplayBuffer, Trajectory, TransitionData
from .self_play import SelfPlayActor, generate_games
from .targets import compute_td_lambda_targets
from .trainer import StochasticMuZeroTrainer, train_stochastic_muzero

__all__ = [
    # Config
    'StochasticMuZeroConfig',
    'default_2048_config',
    'small_2048_config',
    # Replay Buffer
    'ReplayBuffer',
    'Trajectory',
    'TransitionData',
    # Targets
    'compute_td_lambda_targets',
    # Losses
    'compute_policy_loss',
    'compute_value_loss',
    'compute_reward_loss',
    'compute_chance_loss',
    'compute_total_loss',
    # Self-Play
    'SelfPlayActor',
    'generate_games',
    # Training
    'StochasticMuZeroLearner',
    'StochasticMuZeroTrainer',
    'train_stochastic_muzero',
]
