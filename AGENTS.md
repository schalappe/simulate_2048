# AGENTS.md

## Project overview

A Stochastic MuZero agent that learns 2048 through self-play: MCTS with chance nodes for tile spawns, guided by representation, dynamics, and prediction networks written in JAX and Flax. Most changes land in `reinforce/training/`, `reinforce/train.py`, and `reinforce/evaluate.py`. Read `docs/project.md` for the architecture and `docs/4498_planning_in_stochastic_environ.pdf` for the Stochastic MuZero paper. To prove behavior beyond unit tests, follow the `verify-simulate-2048` skill in `.agents/skills/verify-simulate-2048/SKILL.md` (`.claude/skills` is a symlink to `.agents/skills`).

## Structure

| Path | Responsibility |
|---|---|
| `twentyfortyeight/` | NumPy game engine. Only `manuals_control.py` and `tests/` use it. |
| `reinforce/game/core.py` | JAX game engine used by training and evaluation. |
| `reinforce/mcts/` | Stochastic MCTS on top of `mctx`, plus policy targets and action selection. |
| `reinforce/neural/` | The three Flax networks and the `NetworkParams` container. |
| `reinforce/training/` | Config presets, self-play, replay buffer, losses, learner, trainer, orbax checkpoints. |
| `reinforce/train.py`, `reinforce/evaluate.py` | Training and evaluation CLIs. |
| `manuals_control.py` | Manual play in a matplotlib window, driven by arrow keys. |
| `scripts/`, `notebooks/` | Benchmarks, profiling, and analysis. Lint and type check do not cover them. |

## Commands

Run every command from the repository root.

```bash
uv sync --group dev                                                  # install runtime and dev dependencies
uv run pytest tests/test_board.py::TestGameBoard::test_merge_column  # run a single test
make test                                                            # full suite: 72 tests, about 3 seconds
make check                                                           # ruff lint, then pyrefly type check
make format                                                          # ruff format, then ruff check --fix (rewrites files)
```

A task is done when `make check` and `make test` both pass.

Do not use bare `uv run ruff check .` or `uv run pyrefly check .` as a gate. On a clean checkout they report 10 ruff errors and 23 pyrefly errors in `notebooks/`, `scripts/`, and the ignored `zoo/`. The Makefile scopes both tools to `reinforce/`, `tests/`, and `twentyfortyeight/`.

```bash
uv run python -m reinforce.train --mode tiny --steps 20 --checkpoint-dir "$TMPDIR/ckpt" --no-progress  # smoke-train, about 15 seconds on CPU
uv run python -m reinforce.evaluate --games 1 --simulations 8                                          # smoke-evaluate, about 5 seconds
```

`make train-small` runs for about 10 minutes and `make train-full` for hours. Do not use either as a check. `uv run python manuals_control.py` opens a window and never exits on its own; drive it with `.agents/skills/verify-simulate-2048/scripts/drive_manual_play.py` instead.

## Architecture

- The two game engines are independent: `reinforce/` never imports `twentyfortyeight/`. A change to the game rules must land in both `twentyfortyeight/core/gameboard.py` and `reinforce/game/core.py`, each with its own tests (`tests/test_board.py`, `tests/test_jax_game.py`).
- Actions 0 to 3 mean left, up, right, down in both engines. Each move rotates the board, slides left, and rotates back (`latent_state()` in both engines).
- Only the NumPy engine has `after_state()`, which lists every tile spawn with its probability. The JAX path models spawns as MCTS chance nodes in `reinforce/mcts/stochastic_mctx.py`.
- Rewards are normalized in two different places. The NumPy environment applies `normalize_reward()` from `twentyfortyeight/utils/normalize.py`. Training applies `scale_value()` from `reinforce/training/losses.py`.
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

- Always pass `--checkpoint-dir` outside the repository when you run `reinforce.train`. The default is `checkpoints/`, which holds the user's own training state, and `Trainer.initialize()` resumes from whatever that directory contains.
- Do not treat `reinforce.evaluate --checkpoint` as an evaluation of a trained model. It prints `Checkpoint loading ... not yet implemented` and plays with an untrained network (`reinforce/evaluate.py:77`).
- Never commit `checkpoints/` or `zoo/`. Both are gitignored model output.

## Failure log

- `lax.cond` traces both branches, even the one that never runs. Guard every division in both branches against edge values such as `temperature=0` (732d538, `reinforce/mcts/policy.py`).
- A masked policy can sum to zero. Fall back to a uniform policy over legal actions before you normalize (bd385d2, `reinforce/game/core.py`).
- Orbax restores NamedTuples as plain dicts. Rebuild `NetworkParams` explicitly after a restore (51a512b, `reinforce/training/learner.py`).
- Keep every trainable parameter inside `NetworkParams`. `update_params` silently skipped encoder parameters that lived in a config dict (cd85334).
- `queue.Full` in the replay-buffer prefetch loop is normal backpressure. Do not log it as an error (2b2bd04, `reinforce/training/replay_buffer.py`).
