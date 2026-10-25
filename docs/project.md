# 2048 AI Agent

Stochastic MuZero agent for 2048, following "Planning in Stochastic Environments with a Learned Model"
(ICLR 2022, `docs/4498_planning_in_stochastic_environ.pdf`, Appendix C for 2048). The agent plans with Monte
Carlo Tree Search over learned models, including chance nodes for tile spawns.

## Architecture

### Data flow

```text
Board → Representation → state ─► Prediction → (policy, value)
                           │
             state + action ─► Afterstate dynamics → afterstate ─► Afterstate prediction → (afterstate value, chance distribution)
                                                       │
                              afterstate + chance code ─► Dynamics → (next state, reward)

Training only: (observation t, observation t+1) ─► Encoder → chance code of the real transition
```

Six models, each a ResNet v2 tower of fully connected layers with layer normalization:
- **Representation**: binary observation (16 tiles × 31 bits) to latent state.
- **Prediction**: policy logits and categorical value.
- **Afterstate dynamics**: latent state and action to afterstate (after the slide, before the spawn).
- **Afterstate prediction**: categorical afterstate value and logits over 32 chance codes.
- **Dynamics**: afterstate and chance code to next latent state and categorical reward.
- **Encoder**: two consecutive observations to a chance code (one-hot forward, straight-through gradient).

Values, rewards, and afterstate values are categorical over 601 bins of h(x) = sign(x)(√(|x|+1) − 1) + 0.001x
(`reinforce/neural/support.py`). The search (`reinforce/mcts/stochastic_mctx.py`) reads every head as a raw scalar.

### Modules

**`twentyfortyeight/`**: NumPy game engine, used by manual play and its tests only.
- `core/gameboard.py`: board operations via rotation transforms; `after_state()` lists every spawn with its probability
- `core/gamemove.py`: move validation
- `envs/twentyfortyeight.py`: Gymnasium-style environment with `normalize_reward()`

**`reinforce/`**: Stochastic MuZero in JAX.
- `game/core.py`: JAX game rules, binary observation encoding, uint8 board storage
- `mcts/stochastic_mctx.py`: bridge to `mctx.stochastic_muzero_policy`, legal-move mask included
- `mcts/policy.py`: action selection from visit counts, temperature as a traced value
- `neural/`: the six Flax models, parameter containers, categorical value support
- `training/config.py`: presets (`tiny` for local CPU, `small`, `full` for GPU)
- `training/self_play.py`: self-play compiled in one `lax.scan`, game collection, evaluation games
- `training/replay_buffer.py`: TD(λ) targets, sequences of up to 200 steps, priorities, atomic save
- `training/losses.py`: MuZero loss plus chance losses (paper eq. 6–7)
- `training/learner.py`: train step and orbax checkpoints
- `training/trainer.py`: training loop over a run directory

## Training loop

1. Self-play plays `num_parallel_games` games side by side, `moves_per_generation` moves per call, every
   `generation_interval` training steps. Games run to their real end (`max_game_moves` only stops runaway
   games, marked truncated).
2. Each finished game gets TD(λ) value targets over the whole game (n = 10, λ = 0.5, γ = 0.999), then is split
   into sequences of up to 200 steps. Each sequence also stores the next K steps, so unrolls read real data.
3. Training samples positions with priority |search value − value target| (α = β = 1) and unrolls the models
   K = 5 steps.
4. Every `checkpoint_interval` steps the run directory receives the replay buffer and a checkpoint.

## Run directory

```text
run_dir/
    config.json          configuration the run was started with
    checkpoints/         parameters, optimizer state, step, random key
    replay_buffer.npz    replay buffer
    metrics.jsonl        "train" and "eval" records
```

Rerunning `reinforce.train` with the same `--run-dir` resumes the run. Full training runs on RunPod: see
`docs/runpod.md`.

## Implementation details

**Board rotation** (`gameboard.py`, `core.py`): a move in any direction is rotate, slide left, rotate back.

**Board storage**: the replay buffer stores boards as 16 uint8 exponents (16 bytes per position) and encodes
observations on the device. 125,000 sequences of 200 steps as float32 observations would need about 50 GB.

**JIT**: self-play, the train step, and the losses compile with `jax.jit`. Use `lax.cond`, `lax.scan`, and
`jnp.where` instead of Python control flow on traced values.
