"""
Reinforce: JAX-based Stochastic MuZero implementation for 2048.

This package provides GPU-accelerated training and evaluation of
Stochastic MuZero agents using JAX, Flax, and mctx.

Submodules
----------
game : JAX-native 2048 game logic
    Vectorizable game operations with JIT compilation.
mcts : MCTS with DeepMind's mctx library
    GPU-accelerated tree search for Stochastic MuZero.
neural : Flax neural network models
    Stochastic MuZero architecture with 6 networks.
training : Training infrastructure
    Self-play, replay buffer, losses, and training loop.

Reference: "Planning in Stochastic Environments with a Learned Model" (ICLR 2022)
"""
