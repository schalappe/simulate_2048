"""
Reinforcement learning module for 2048.

This module provides:
- MCTS (Monte Carlo Tree Search) agent with PUCT selection
- Neural network models (MuZero-style architecture)
- Evaluation utilities

Submodules
----------
mcts : Monte Carlo Tree Search implementation
    - MonteCarloAgent: MCTS-based agent for action selection
neural : Neural network models
    - Network: Unified wrapper for representation, dynamics, and prediction models
evaluate : Evaluation script for testing agents
"""

from .mcts import MonteCarloAgent
from .neural import Network

__all__ = ['MonteCarloAgent', 'Network']
