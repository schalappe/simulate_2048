# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search module for reinforcement learning in stochastic environments.

This module provides a complete MCTS implementation with:
- Decision and Chance nodes for stochastic environments
- PUCT selection strategy (AlphaZero-style)
- Progressive widening for managing branching factor
- Adaptive simulation count based on tree depth
"""
from .actor import MonteCarloAgent

__all__ = ["MonteCarloAgent"]
