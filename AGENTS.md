# AGENTS.md

## Project overview

A Stochastic MuZero agent that learns 2048 through self-play: MCTS with chance nodes for tile spawns, guided by representation, dynamics, and prediction networks written in JAX and Flax. Most changes land in `reinforce/training/`, `reinforce/train.py`, and `reinforce/evaluate.py`. Read `docs/project.md` for the architecture, `docs/4498_planning_in_stochastic_environ.pdf` for the Stochastic MuZero paper (Appendix C holds the 2048 settings), `docs/roadmap.md` for the work in progress, and `docs/runpod.md` for GPU training. To prove behavior beyond unit tests, follow the `verify-simulate-2048` skill in `.agents/skills/verify-simulate-2048/SKILL.md` (`.claude/skills` is a symlink to `.agents/skills`).

## Structure

| Path | Responsibility |
|---|---|
| `twentyfortyeight/` | NumPy game engine. Only `manuals_control.py` and `tests/` use it. |
| `reinforce/game/core.py` | JAX game engine used by training and evaluation. |
| `reinforce/mcts/` | Stochastic MCTS on top of `mctx`, plus policy targets and action selection. |
| `reinforce/neural/` | The six Flax models, the `NetworkParams` container, and the categorical value support (`support.py`). |
| `reinforce/training/` | Config presets, compiled self-play, replay buffer and value targets, losses, learner, trainer, orbax checkpoints. |
| `reinforce/train.py`, `reinforce/evaluate.py` | Training and evaluation CLIs. |
| `manuals_control.py` | Manual play in a matplotlib window, driven by arrow keys. |
| `notebooks/` | Network walkthrough, model tables, search-tree plot, and a Kaggle/local training notebook. `make check` does not cover them; run `uv run ruff check notebooks` and `uv run pyrefly check notebooks` after editing one. |

## Commands

Run every command from the repository root.

```bash
uv sync --group dev                                                  # install runtime and dev dependencies
uv run pytest tests/test_board.py::TestGameBoard::test_merge_column  # run a single test
make test                                                            # full suite: 86 tests, about 25 seconds
make check                                                           # ruff lint, then pyrefly type check
make format                                                          # ruff format, then ruff check --fix (rewrites files)
```

A task is done when `make check` and `make test` both pass.

Do not use bare `uv run ruff check .` or `uv run pyrefly check .` as a gate. On a clean checkout pyrefly reports 4 errors in the ignored `zoo/`. The Makefile scopes both tools to `reinforce/`, `tests/`, and `twentyfortyeight/`.

```bash
uv run python -m reinforce.train --mode tiny --steps 20 --run-dir "$TMPDIR/run" --allow-cpu  # smoke-train, about 10 seconds on CPU
uv run python -m reinforce.evaluate --run-dir "$TMPDIR/run" --games 1                       # evaluate that run, about 5 seconds
```

`make train-small` and `make train-full` need a GPU and run for hours to days, on RunPod (`docs/runpod.md`). Do not use either as a check. Training stops at startup without a GPU unless you pass `--allow-cpu`. `uv run python manuals_control.py` opens a window and never exits on its own; drive it with `.agents/skills/verify-simulate-2048/scripts/drive_manual_play.py` instead.

## Architecture

- The two game engines are independent: `reinforce/` never imports `twentyfortyeight/`. A change to the game rules must land in both `twentyfortyeight/core/gameboard.py` and `reinforce/game/core.py`, each with its own tests (`tests/test_board.py`, `tests/test_jax_game.py`).
- Actions 0 to 3 mean left, up, right, down in both engines. Each move rotates the board, slides left, and rotates back (`latent_state()` in both engines).
- Only the NumPy engine has `after_state()`, which lists every tile spawn with its probability. The JAX path models spawns as MCTS chance nodes in `reinforce/mcts/stochastic_mctx.py`.
- Rewards are normalized in two different places. The NumPy environment applies `normalize_reward()` from `twentyfortyeight/utils/normalize.py`. Training represents values, rewards, and afterstate values as 601-bin categoricals over `scale_value()` (h) in `reinforce/neural/support.py`.
- Training follows Appendix C of the paper: binary observation (16 × 31 bits), TD(λ) targets, sequences of up to 200 steps, priorities |v − z|. Games themselves run to their end. The replay buffer stores boards as uint8 exponents and encodes observations on the device.
- Code in `reinforce/` runs under `jax.jit`. Use `lax.cond`, `lax.scan`, and `jnp.where` instead of Python control flow on traced values (see the module docstring of `reinforce/game/core.py`).

## Conventions

- Docstrings follow the NumPy style. Inline comments use the prefixes from the `caveman-comments` skill, mostly `# ##>:`.
- Every `__init__.py` holds a docstring and re-exports nothing. Import from the defining module, for example `from reinforce.training.trainer import Trainer`.
- `docs/agents/issue-tracker.md` explains how to manage GitHub issues with `gh`. `docs/agents/triage-labels.md` lists the triage labels, and `docs/agents/domain.md` describes the glossary and ADR layout.

## Git and PR workflow

- Write commit subjects as English Conventional Commits with a scope, for example `fix(learner): restore checkpoints with proper type reconstruction`.
- Most commits end with the trailer `Co-Authored-By: Claude <noreply@anthropic.com>`.
- PRs target `main`. The repository has no CI workflows, so run `make check` and `make test` locally before pushing.

## Boundaries

- Always pass `--run-dir` outside the repository when you run `reinforce.train`. The default is `runs/default/`, and `Trainer.initialize()` resumes from whatever the run directory contains. `checkpoints/` holds the user's pre-rewrite state, which the current code cannot load; never write to it.
- `reinforce.evaluate` without `--run-dir` plays with an untrained network. Only `--run-dir` evaluates a trained model.
- Never commit `checkpoints/`, `runs/`, or `zoo/`. All three are gitignored model output.

## Failure log

- `lax.cond` traces both branches, even the one that never runs. Guard every division in both branches against edge values such as `temperature=0` (732d538, `reinforce/mcts/policy.py`).
- A masked policy can sum to zero. Pass legal moves to `mctx` as `invalid_actions` and keep zero-weight actions at `-inf` when sampling, so an illegal move is never chosen (bd385d2, `select_action` in `reinforce/mcts/policy.py`).
- Orbax restores NamedTuples as plain dicts unless you pass a target. Restore with `StandardRestore` of a `jax.eval_shape` template; rebuilding by hand lost the optimizer state and restarted the warmup on every resume (`CheckpointManager.restore` in `reinforce/training/learner.py`).
- Keep every trainable parameter inside `NetworkParams`. `update_params` silently skipped encoder parameters that lived in a config dict (cd85334).
- Apply the value transform h once. Heads learn h-space categoricals; the search must read raw values through `logits_to_scalar`. Re-scaling search values that were already in h-space compounded h at every generation.
- A placeholder in the loss trains silently. The chance target was a constant code 0, so every chance node expanded one branch; test that each model receives gradient (`tests/test_training.py`).
- JAX falls back to the CPU with only a warning when CUDA does not load. `reinforce.train` stops at startup without a GPU so a paid pod never trains at CPU speed.
- Never block on device values inside the training loop (`int(state.step)`, `float(loss)`) except for logging and saving: JAX dispatch is asynchronous, and the host samples the next batch while the device trains.
