# 2048 AI Agent

AI agent that plays 2048 using Monte Carlo Tree Search guided by neural networks, inspired by AlphaZero and Stochastic MuZero. Implements strategies from "Planning in Stochastic Environments with a Learned Model."

## Overview

**Goal**: Train an agent to play 2048 effectively using learned models and MCTS planning.

**Key Components**:
- Game environment (Gymnasium interface)
- MCTS planning algorithm with stochastic branching
- Neural networks for state evaluation and transition modeling

## Architecture

### Data Flow

```text
Observation (4x4 board) → Encoder → Hidden State → Predictor → (Policy, Value)
                                         ↓
                         Action + State → Dynamic → (Next State, Reward)
```

### Module Structure

**`twentyfortyeight/`** - Game implementation
- `core/gameboard.py`: Board state and operations (numpy rotations for movement)
- `core/gamemove.py`: Move validation and application
- `envs/twentyfortyeight.py`: Gymnasium environment interface

**`monte_carlo/`** - MCTS planning
- `actor.py`: Action selection from search results
- `node.py`: Decision and Chance nodes with progressive widening
- `search.py`: PUCT selection, adaptive simulation, backpropagation

**`neurals/`** - Neural network models
- `models.py`: Representation, Dynamics, and Prediction networks
- `network.py`: Unified interface for all three models

**`notebooks/`** - Analysis and visualization
- `test_network.ipynb`: Network debugging
- `visualize_model.ipynb`: State representation visualization
- `visualize_trees.ipynb`: MCTS tree exploration

**`tests/`** - Unit tests
- `test_board.py`: Game board logic
- `test_encoded.py`: State encoding
- `test_move.py`: Move application
- `test_perf_utils.py`: Utility performance

## Key Concepts

### Stochastic State Handling

The codebase distinguishes between deterministic and stochastic transitions:

- `latent_state()`: Deterministic - applies action without adding new tile
- `after_state()`: Stochastic - generates all possible states after tile spawn with probabilities
- `next_state()`: Full transition including random tile placement

### Critical Implementation Patterns

**Board Rotation**: All movement operations (left/up/right/down) are implemented by rotating the board, applying a left-slide, then rotating back. See `gameboard.py:119`.

**Reward Normalization**: Log-scale normalization compresses the reward range for stable learning. Maximum theoretical tile is 2^16.

**Progressive Widening**: Chance nodes in MCTS use progressive widening to manage the stochastic branching factor efficiently.

## Quick Start

```bash
# Install dependencies
uv sync

# Play manually
uv run python manuals_control.py

# Evaluate MCTS agent
uv run python evaluate.py --method mcts

# Run tests
uv run pytest tests/
```
