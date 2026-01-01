"""
Monte Carlo Tree Search module for reinforcement learning in stochastic environments.

This module provides a complete MCTS implementation with:
- Decision and Chance nodes for stochastic environments
- PUCT selection strategy (AlphaZero-style)
- Progressive widening for managing branching factor
- Adaptive simulation count based on tree depth

For Stochastic MuZero (network-guided MCTS):
- StochasticMuZeroAgent: Full agent with neural network integration
- network_search: Low-level MCTS with network predictions

For high-performance batched MCTS (AlphaZero-style optimizations):
- BatchedMuZeroAgent: Agent with configurable batching levels
- batched_search: MCTS with virtual loss and threaded evaluation
- MCTSMode: SEQUENTIAL (4x), BATCHED (10x), THREADED (25-50x speedup)
"""
