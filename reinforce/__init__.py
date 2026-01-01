"""
Reinforcement learning module for 2048.

This module provides:
- MCTS (Monte Carlo Tree Search) agent with PUCT selection
- Neural network models (Stochastic MuZero architecture)
- Evaluation utilities

Submodules
----------
mcts : Monte Carlo Tree Search implementation
    - MonteCarloAgent: MCTS-based agent for action selection (no network)
    - StochasticMuZeroAgent: Full agent with neural network integration
neural : Neural network models
    - Network: Original 3-model wrapper for backward compatibility
    - StochasticNetwork: Full 6-model wrapper for Stochastic MuZero
evaluate : Evaluation script for testing agents
"""

from .mcts import MonteCarloAgent, StochasticMuZeroAgent
from .neural import Network, StochasticNetwork

__all__ = ['MonteCarloAgent', 'StochasticMuZeroAgent', 'Network', 'StochasticNetwork']
