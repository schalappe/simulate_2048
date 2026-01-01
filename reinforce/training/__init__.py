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
