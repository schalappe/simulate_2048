---
name: verify-simulate-2048
description: Prove behavior of the simulate_2048 app (Stochastic MuZero for 2048) through its real entry points - the training and evaluation CLIs (`python -m reinforce.train`, `python -m reinforce.evaluate`) and the manual-play matplotlib window (`manuals_control.py`). Use when a change touches training, evaluation, checkpoints, MCTS, the game engine, or manual play and you need evidence beyond unit tests.
---

# Verify simulate_2048

The app has no server. Its surfaces are:

- **Primary: CLI.** `reinforce.train` (self-play + learning + checkpoints, buffer, and metrics in a run directory) and `reinforce.evaluate` (greedy MCTS games from a run directory or an untrained network, score report).
- **Secondary: desktop GUI.** `manuals_control.py`, a matplotlib window driven by arrow keys. Drive it headlessly with `scripts/drive_manual_play.py`.

Every command runs from the repo root. Feature recipes live in `features/`; read the one matching your change.

## Run directory

Each verification owns one directory outside the repo. Put every log, training run, and screenshot in it:

```bash
RUN="${TMPDIR:-/tmp}/verify-2048-$(date +%Y%m%d-%H%M%S)"   # or your scratchpad dir
mkdir -p "$RUN"
```

**Always pass `--run-dir "$RUN/run"` to training.** `Trainer.initialize()` resumes from whatever its run directory already holds. The default is the repo's `runs/default/`, which may hold the user's own training state, so a launch without the flag silently resumes it and writes into it. The older `checkpoints/` directory holds the user's pre-rewrite state; never write to it.

Isolation: any number of runs can coexist if each one has its own `--run-dir`. Evaluation writes nothing to disk.

## Doctor

Read-only. Takes about 5 seconds.

```bash
uv run python -c "import sys, jax, mctx, orbax.checkpoint, matplotlib; print(sys.version.split()[0], jax.__version__, jax.devices())"
ls checkpoints/ runs/ 2>/dev/null | head      # user state: note it, never write to it
pgrep -fl "reinforce.train|reinforce.evaluate|manuals_control" || echo "no app process running"
```

Healthy: Python 3.12.x, JAX imports, at least one device (`CpuDevice` on a Mac without jax-metal). If imports fail, run `uv sync --group dev` and rerun Doctor. A process already running is not yours: leave it alone.

## Launch

A fresh tiny training run is the launch. It goes from an empty run directory to a usable checkpoint in about 10 s on CPU.

```bash
uv run python -m reinforce.train --mode tiny --steps 20 --run-dir "$RUN/run" --allow-cpu > "$RUN/train.log" 2>&1 &
TRAIN_PID=$!; echo "$TRAIN_PID" >> "$RUN/pids"
wait "$TRAIN_PID"; echo "exit=$?"
```

Ready signal, all three required:

- `exit=0`
- `$RUN/train.log` starts with `JAX backend: cpu` and contains `Initialized new training state at step 0`, `Training done: 20 steps`, and an `[eval] step 20:` line
- `$RUN/run/checkpoints/20/` exists (the step directory is named after `--steps`)

`Restored training state from step N` instead of `Initialized ...` means the run directory was not empty: the run is not fresh. Start over with a new `$RUN`.

## Drive

| Surface | Harness | Recipe |
|---|---|---|
| Training | shell, background + `wait` | `features/train-from-scratch.md`, `features/resume-training.md` |
| Evaluation | shell, background + `wait` | `features/evaluate.md` |
| Manual play | `scripts/drive_manual_play.py` (Agg backend, real key handler) | `features/manual-play.md` |

Run long commands with `&`, append `$!` to `$RUN/pids`, then `wait` on that PID. Both CLIs print plain lines (no progress bars); read results with `grep`.

Make targets (`make train-tiny`, `make eval`, `make play`) call the same modules, but `make train-*` writes to the repo's `runs/`. Prefer the explicit commands above.

## Evidence

Capture, all under `$RUN`:

- the exact command line and exit code of each step
- `train.log` / `eval.log` (full stdout+stderr)
- `ls -R "$RUN/run" | head -20` for run-directory side effects (`config.json`, `checkpoints/`, `replay_buffer.npz`, `metrics.jsonl`)
- for manual play: `play/NN-<key>.png` and `play/boards.txt`; open at least one PNG and check it matches the board text
- `git status --short > "$RUN/git-before.txt"` and `ls -la checkpoints/ runs/ > "$RUN/ckpt-before.txt"` before the first step, the same into `*-after.txt` at the end, then `diff` each pair: both must be empty, proving the run left the repo and the user's checkpoints untouched

Report the observed numbers (steps, steps/second, moves/second, max tile, tile frequency) verbatim. Untrained or tiny-trained agents score randomly; a score is evidence the loop runs, not that learning works.

## Cleanup

Kill only PIDs this run recorded; evidence stays.

```bash
[ -f "$RUN/pids" ] && while read -r pid; do kill "$pid" 2>/dev/null && echo "killed $pid"; done < "$RUN/pids"
while read -r pid; do ps -p "$pid" > /dev/null && echo "STILL ALIVE $pid"; done < "$RUN/pids"
ls "$RUN"
```

Expected: no `STILL ALIVE` line, and the evidence files still listed. A run that finished under `wait` is already dead, so the `kill` lines print nothing; the loop matters when a step was interrupted or never awaited. Do not delete `$RUN`; do not touch `checkpoints/` or `runs/`.

## Maintenance

1. Run Doctor first on every use. Launch fails → fix the recipe before testing anything.
2. A drive fails on a feature the code still has → stale recipe: fix it in the same PR. The code lost the feature → regression: report it, never edit the recipe to pass.
3. A PR that changes launch, routes, or seeds updates the matching feature file in the same PR.
4. A scenario blocked because you did not know how to create its data → once you find the recipe, add it to the matching feature file (or a new one) in the same PR, so the next agent is not blocked.
