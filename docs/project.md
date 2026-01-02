# 2048 AI Agent

MuZero-inspired reinforcement learning agent for 2048. Uses Monte Carlo Tree Search with neural networks for planning in stochastic environments.

## Architecture

### Data Flow

```text
Board (4×4) → Representation → Hidden State → Prediction → (Policy, Value)
                                     ↓
                   Hidden State + Action → Dynamics → (Next State, Reward)
```

**Three-network architecture:**
- **Representation**: Encodes board observation to latent state
- **Dynamics**: Predicts next latent state and reward from action
- **Prediction**: Outputs policy (action preferences) and value (expected return)

### Modules

**`twentyfortyeight/`** - Game engine (NumPy-based)
- `core/gameboard.py` - Board operations via rotation transforms
- `core/gamemove.py` - Move validation
- `envs/twentyfortyeight.py` - Gymnasium environment

**`reinforce/`** - RL components (JAX-based)
- `game/core.py` - JAX game logic with `latent_state()` and `after_state()`
- `mcts/stochastic_mctx.py` - MCTS with chance nodes and progressive widening
- `neural/network.py` - Three-network model interface
- `training/` - Self-play, replay buffer, loss functions

**`tests/`** - Unit tests for board logic, environment, JAX game

**`notebooks/`** - Visualization and debugging

**`scripts/`** - Benchmarking and profiling tools

## Key Concepts

### Stochastic Transitions

2048 has deterministic player actions (slide) and stochastic environment responses (tile spawn). The code models this explicitly:

- `latent_state(state, action)` - Deterministic: slide without new tile
- `after_state(state)` - Stochastic: all possible tile spawns with probabilities
- `next_state(state, action)` - Full transition: slide + random tile

MCTS uses **chance nodes** for stochastic branching with **progressive widening** to limit tree growth.

### Implementation Details

**Board Rotation** (`gameboard.py:119`): Movement in any direction is rotation + left-slide + inverse rotation. This reduces all four moves to a single primitive.

**Reward Normalization** (`utils/normalize.py`): Log-scale normalization compresses exponential tile values (2→2^16) to stable learning range.

**JAX Optimization**: Training uses JIT compilation (`@jax.jit`) for fast vectorized game execution and gradient computation.
