"""
Reinforcement learning module for 2048.

This module provides a complete implementation of Stochastic MuZero:
- MCTS (Monte Carlo Tree Search) agent with PUCT selection
- Neural network models (Stochastic MuZero architecture)
- Training infrastructure (self-play, replay buffer, learner)
- Evaluation utilities

Submodules
----------
mcts : Monte Carlo Tree Search implementation
    - MonteCarloAgent: MCTS-based agent for action selection (no network)
    - StochasticMuZeroAgent: Full agent with neural network integration
neural : Neural network models
    - Network: Original 3-model wrapper for backward compatibility
    - StochasticNetwork: Full 6-model wrapper for Stochastic MuZero
training : Training infrastructure
    - StochasticMuZeroConfig: Training configuration
    - StochasticMuZeroTrainer: Main training orchestrator
    - ReplayBuffer: Experience storage with prioritization
evaluate : Evaluation script for testing agents

Reference: "Planning in Stochastic Environments with a Learned Model" (ICLR 2022)
"""
