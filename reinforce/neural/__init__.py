"""
Neural network module for MuZero-style reinforcement learning.

This module provides:
- Model builders for representation, dynamics, and prediction networks
- A unified Network class that wraps all three models
- Utility functions for tensor conversion
"""

from .models import (
    build_dynamics_model,
    build_prediction_model,
    build_representation_model,
    identity_block_dense,
)
from .network import Network, ndarray_to_tensor

__all__ = [
    'identity_block_dense',
    'build_representation_model',
    'build_dynamics_model',
    'build_prediction_model',
    'Network',
    'ndarray_to_tensor',
]
