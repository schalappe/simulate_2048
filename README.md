# 2048 AI Agent

MuZero-inspired reinforcement learning agent that learns to play 2048 through self-play. Uses Monte Carlo Tree Search with neural networks for planning in stochastic environments.

![2048 Game](docs/imgs/game_short.gif)

## Features

- **Stochastic MuZero, as in the paper**: learned chance codes for tile spawns, categorical values, TD(λ) targets, and prioritized replay (Appendix C)
- **Self-play compiled on the GPU**: games, search, and action selection run in one JAX program
- **Resumable runs**: a run directory holds checkpoints, the replay buffer, and metrics, so a restarted GPU pod continues where it stopped
- **Vectorized game engine**: 2048 rules in JAX, batched with `vmap`

## Quick Start

```bash
# Install dependencies (Python 3.12 required)
uv sync

# Play manually
uv run python manuals_control.py

# Train a tiny agent on the local CPU (smoke run)
uv run python -m reinforce.train --mode tiny --steps 100 --run-dir runs/tiny --allow-cpu

# Evaluate that run
uv run python -m reinforce.evaluate --run-dir runs/tiny --games 10
```

Real training (`--mode small` or `--mode full`) needs a GPU: see [docs/runpod.md](docs/runpod.md). Without a
GPU, training stops at startup unless you pass `--allow-cpu`.

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

**GPU support**: on a Linux machine with an NVIDIA GPU, install the CUDA 12 wheels with `uv sync --extra gpu`.

## Project Structure

```text
twentyfortyeight/   # Game engine (NumPy-based)
reinforce/          # RL components (JAX-based)
  ├── game/         # JAX game logic
  ├── mcts/         # Stochastic MCTS on top of mctx
  ├── neural/       # Six Flax models and the categorical value support
  └── training/     # Config, self-play, replay buffer, losses, learner, trainer
tests/              # Unit tests
notebooks/          # Visualization and analysis
docs/               # Architecture documentation
```

See [docs/project.md](docs/project.md) for detailed architecture documentation.

## Development

```bash
make test     # full test suite
make check    # ruff lint, then pyrefly type check (scoped to reinforce/, tests/, twentyfortyeight/)
make format   # ruff format, then ruff check --fix
```

## Architecture

The agent learns six models, as in Stochastic MuZero:
- **Representation**: encodes the board (16 tiles × 31 bits) into a latent state
- **Prediction**: policy and value from a latent state
- **Afterstate dynamics and prediction**: the state after a move, before the tile spawn, with its value and the
  distribution of chance codes
- **Dynamics**: next latent state and reward from an afterstate and a chance code
- **Encoder**: the chance code of a real transition, learned without supervision (VQ-VAE style)

MCTS plans with these models, sampling tile spawns at chance nodes. Training unrolls the models along real
games toward search policies, TD(λ) value targets, rewards, and the encoder's chance codes.

See [docs/project.md](docs/project.md) for implementation details.

## References

- [Stochastic MuZero](https://openreview.net/pdf?id=X6D9bAHhBQ1) - Planning in stochastic environments
- [AlphaZero](https://arxiv.org/pdf/1712.01815v1.pdf) - MCTS with neural networks
- [MuZero](https://arxiv.org/abs/1911.08265) - Model-based RL without environment model

## License

MIT License - see [LICENSE](LICENSE) for details.
