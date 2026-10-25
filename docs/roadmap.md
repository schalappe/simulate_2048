# Roadmap: make Stochastic MuZero training correct and RunPod-ready

The training loop runs but cannot learn what it claims: the chance branch is a stub, value targets are
self-referential and scaled twice, and games stop at 200 moves. This plan fixes the algorithm first, then
makes full training run efficiently on a RunPod GPU pod. Small-scale proof runs locally on CPU; full
training runs only on RunPod.

Tick a box only when its acceptance criterion has evidence (test name, command output, or run log).
The paper is `docs/4498_planning_in_stochastic_environ.pdf`; "paper C" means its Appendix C (2048).

## Decisions

- **Keep JAX.** `mctx` (JAX-only) already provides Stochastic MuZero search with chance nodes, and the
  game engine is in JAX, so the whole self-play step can compile into one GPU program. The bottleneck is
  the per-move Python loop, not the framework. Flax stays on the Linen API.
- **Persist the replay buffer** next to the checkpoints, so a restarted or preempted pod resumes training
  without replaying about 1000 games first.
- **Local vs RunPod.** Local runs prove correctness at small scale (`--mode tiny --allow-cpu`).
  `--mode full` runs only on RunPod. No pod is created without an explicit go and a quoted hourly price.
- **Follow paper C** for representation and targets: binary observation (16 tiles × 31 bits), categorical
  value, reward, and afterstate value with 601 bins over `h(x)` in [0, 600], TD(λ) targets with n = 10,
  λ = 0.5, γ = 0.999, a buffer of 125,000 sequences of up to 200 steps, and priorities `|v − z|`.
- **Store boards, not observations.** 125,000 × 200 positions as float32 observations would take about
  50 GB. The buffer stores each board as 16 `uint8` exponents and encodes observations inside jitted code.
- **Old checkpoints are incompatible** with this branch (new observation size and categorical heads).

## 1. Dependencies and GPU guard

- [x] Bump every dependency to its latest release (jax 0.12, flax 0.12.10, mctx 0.0.71, orbax 0.12.8,
      pyrefly 1.4, ruff 0.17). Drop unused direct dependencies (`jaxlib`, `chex`, `vulture`).
      Evidence: `make test` 72 passed, `make check` clean, tiny smoke-train passes.
- [x] Add a `gpu` extra (`jax[cuda12]`, Linux only) for RunPod: `uv sync --extra gpu`.
- [x] Training stops at startup when JAX sees no GPU, because JAX otherwise falls back to the CPU with only
      a warning and a paid GPU pod would sit idle. `--allow-cpu` opts in for local runs; the Makefile tiny
      and small targets, AGENTS.md, and the verify skill pass it. Evidence: without the flag the CLI prints
      the error and exits 1; with it the tiny run completes.
- [ ] Prove `jax.devices()` lists the GPU on a RunPod pod, and watch the startup line `JAX backend: gpu`
      in every pod log (step 4).

## 2. Model, targets, and replay buffer (paper C)

- [x] Binary observation: each tile becomes 31 bits (bit k set for tile 2^k), 496 inputs in total.
      Evidence: `TestEncodeObservation` in `tests/test_jax_game.py`.
- [x] Categorical value, reward, and afterstate value heads (601 bins, two-hot of `h(x)` clipped to
      [0, 600]) in `reinforce/neural/support.py`. Evidence: `TestValueSupport`.
- [x] The search sees raw values only: `run_mcts` reads every head through `logits_to_scalar` before
      handing it to `mctx`. Stored search values are raw.
- [x] TD(λ) value targets computed once per finished game, vectorized, with no bootstrap after a terminal
      state and a bootstrap from the last search value at the safety limit. Evidence: `TestValueTargets`
      (hand-computed 3-step game).
- [x] Each game is split into sequences of up to 200 steps, each carrying a K-step tail; absorbing targets
      after a terminal state, masked targets after a truncation. Evidence: `TestReplayBuffer`.
- [x] Priorities `|v − z|` per position at insertion, α = β = 1, two-level sampling, importance weights
      over all stored positions. Sampling 1024 positions from a full buffer (125,000 sequences) takes about
      5 ms on an Apple Silicon CPU.
- [x] Chance codes: the encoder reads two consecutive observations, its straight-through code feeds
      `dynamics`, `afterstate_prediction` learns it, the commitment loss pulls the encoder toward it, and
      the afterstate value learns the value target (paper eq. 7). Evidence: `TestLosses` (every model
      receives gradient; the encoder changes after training steps).

## 3. Self-play compiled on the device

- [x] One jitted `lax.scan` (`play_moves`) plays a chunk of moves for a batch of games, restarting
      finished games inside the scan. Temperature is traced.
- [x] Legal moves reach `mctx` as `invalid_actions`; games run to their end, with `max_game_moves`
      (100,000) as a safety limit that marks games truncated. Evidence: `TestSearch`, `TestSelfPlay`.
- [x] Evaluation reuses `play_moves` with temperature 0 and no Dirichlet noise (`evaluate_games`).

## 4. Checkpoints, resume, and RunPod

- [x] Restore keeps the optimizer state through a `jax.eval_shape` template. Evidence:
      `test_restore_keeps_optimizer_state` (every leaf equal after save and restore).
- [x] The replay buffer is saved as one file in the run directory, written to a temporary file then
      renamed. Resume restores network, optimizer, step, RNG key, and buffer; a run directory refuses another
      configuration. Evidence: `test_trainer_resumes_run`; CLI resume restored step 60, 31 sequences, and
      1163 positions.
- [x] Pending checkpoint saves finish before the process exits (`CheckpointManager.wait` and `close`).
- [x] Metrics go to `metrics.jsonl` in the run directory (`"kind": "train"` and `"kind": "eval"`).
- [x] `tiny` and `small` presets have their own temperature thresholds.
- [x] Pod setup documented in `docs/runpod.md`.
- [ ] Prove on a pod: `JAX backend: gpu`, a short run, then resume after a restart. Measure steps per
      second and self-play moves per second, then tune `num_parallel_games` and `generation_interval`.

## 5. Evaluation

- [x] `reinforce.evaluate --run-dir` loads the run's `config.json` and latest checkpoint.
      Evidence: `test_evaluate_loads_the_run`; CLI printed `Checkpoint step: 80` for an 80-step run.

## 6. Cleanup and docs

- [x] Deleted duplicate and dead paths: legacy sampling and train steps, sequential self-play,
      `run_mcts_jit`, `batched_run_mcts`, `policy.temperature_schedule`, `core.sample_action`,
      `core.max_tile`, `core.count_empty`, `game/env.py`, the background batch loader, the two scripts in
      `scripts/` (`profile_mcts.py` already called a nonexistent `evaluate` signature), and dead config fields.
- [x] Removed the background batch loader: JAX dispatches each train step asynchronously, so host sampling
      already overlaps with device compute.
- [x] Updated `README.md`, `docs/project.md`, `AGENTS.md`, and the `verify-simulate-2048` skill.
- [x] `make check` and `make test` pass (86 tests); tiny smoke-train, resume, and evaluation pass.
- [x] Rewrote the four notebooks for the new API (network walkthrough, Flax model tables, `mctx` search-tree
      plot, Kaggle/local training with `Trainer`). Evidence: all four execute end to end (training patched to
      a 20-step tiny run); ruff and pyrefly report no error in `notebooks/`.

## 7. Independent review follow-ups

- [x] Truncated games bootstrap from the last searched state instead of reusing its value one step later.
      Evidence: `test_truncated_game_bootstraps_from_last_searched_state`.
- [x] An interrupt during self-play no longer leaves a gap inside games in progress (actors advance only
      after the collector holds the chunk).
- [x] `evaluate_games` rejects fewer than one game; training no longer evaluates twice when it stops on an
      evaluation step.
- [x] Chance loss target keeps its straight-through gradient and the 0.5 gradient scale sits on the state
      carried to the next unroll step, as in the paper's pseudocode (Appendix H) and MuZero.
- [ ] Untrained value heads read about 57,000 (uniform logits put the expected bin at 300 of 600), and early
      TD targets bootstrap from these values. Watch the first RunPod run; if early learning stalls, initialize
      the value and reward heads toward bin 0.
