# Train from scratch

**Reach**: `uv run python -m reinforce.train` (also `make train-tiny|train-small|train-full`). Flags: `--mode tiny|small|full`, `--steps N`, `--run-dir DIR`, `--allow-cpu` (required without a GPU: training otherwise stops at startup with `No GPU visible to JAX` and exit code 1).

**Set up**: an empty run directory. `mkdir -p "$RUN"` and let the trainer create `$RUN/run`. Nothing else: the replay buffer fills itself with self-play games before training starts.

**Drive**:

```bash
uv run python -m reinforce.train --mode tiny --steps 20 --run-dir "$RUN/run" --allow-cpu > "$RUN/train.log" 2>&1 &
TRAIN_PID=$!; echo "$TRAIN_PID" >> "$RUN/pids"; wait "$TRAIN_PID"; echo "exit=$?"
```

**Proof**:

- `train.log` shows, in order: `JAX backend: cpu`, `Configuration: tiny`, `Initialized new training state at step 0`, `Filling buffer: ... 2/2 sequences`, `step 10: loss ...` and `step 20: loss ...` lines, `Training done: 20 steps`, then `[eval] step 20: mean score <n>, best tile <tile>, mean length <n>`.
- `$RUN/run/` holds `config.json`, `checkpoints/20/`, `replay_buffer.npz`, and `metrics.jsonl` with two `"kind": "train"` records and one `"kind": "eval"` record.

**Gotchas**:

- Without `--run-dir`, the run resumes from and writes into the repo's `runs/default/`.
- Without `--allow-cpu` on a machine with no GPU, the CLI exits 1 before creating anything. That is the intended guard, not a failure of your change.
- Measured on an Apple Silicon CPU: tiny + 20 steps ≈ 10 s wall clock, mostly JIT compilation; `small` and `full` need a GPU and run for hours (see `docs/runpod.md`). Use `tiny` unless the change is config-specific.
- Games run to their real end, so buffer filling time depends on game length, not on a fixed game count.
- The final evaluation plays `eval_games` greedy games inside the same process; it is part of the proof, not a separate step.
