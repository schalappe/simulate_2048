# 2048 AI Agent

MuZero-inspired reinforcement learning agent that learns to play 2048 through self-play. Uses Monte Carlo Tree Search with neural networks for planning in stochastic environments.

![2048 Game](docs/imgs/game_short.gif)

## Features

- **JAX-based training**: Fast, GPU-accelerated self-play and learning
- **Stochastic MCTS**: Handles random tile spawns with chance nodes and progressive widening
- **Three-network architecture**: Representation, Dynamics, and Prediction networks (MuZero-style)
- **Vectorized game engine**: High-performance 2048 implementation with batch operations

## Quick Start

```bash
# Install dependencies (Python 3.12 required)
uv sync

# Play manually
uv run python manuals_control.py

# Train agent
uv run python -m reinforce.train

# Evaluate trained agent
uv run python -m reinforce.evaluate --checkpoint checkpoints/latest.ckpt
```

## Installation

**Prerequisites**: Python 3.12, [uv](https://docs.astral.sh/uv/) package manager

```bash
# Clone repository
git clone <repository-url>
cd simulate_2048

# Install dependencies
uv sync

# Install with development tools
uv sync --group dev
```

**GPU Support (Optional)**: For faster training, install JAX with CUDA support following [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Project Structure

```text
twentyfortyeight/   # Game engine (NumPy-based)
reinforce/          # RL components (JAX-based)
  ├── game/         # JAX game logic
  ├── mcts/         # Stochastic MCTS implementation
  ├── neural/       # Three-network model
  └── training/     # Self-play, replay buffer, losses
tests/              # Unit tests
notebooks/          # Visualization and analysis
docs/               # Architecture documentation
```

See [docs/project.md](docs/project.md) for detailed architecture documentation.

## Development

```bash
# Run tests
uv run pytest tests/

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run pyrefly check .
```

## Architecture

The agent uses three neural networks:
- **Representation**: Encodes 4×4 board to latent state
- **Dynamics**: Predicts next state and reward from action
- **Prediction**: Outputs policy and value estimates

MCTS explores the game tree using these networks, handling stochastic tile spawns through chance nodes. Training uses self-play data with policy and value targets from MCTS.

See [docs/project.md](docs/project.md) for implementation details.

## References

- [Stochastic MuZero](https://openreview.net/pdf?id=X6D9bAHhBQ1) - Planning in stochastic environments
- [AlphaZero](https://arxiv.org/pdf/1712.01815v1.pdf) - MCTS with neural networks
- [MuZero](https://arxiv.org/abs/1911.08265) - Model-based RL without environment model

## License

MIT License - see [LICENSE](LICENSE) for details.
