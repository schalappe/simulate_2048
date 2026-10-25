# Train from scratch

**Reach**: `uv run python -m reinforce.train` (also `make train-tiny|train-small|train-full`). Flags: `--mode tiny|small|full`, `--steps N`, `--checkpoint-dir DIR`, `--seed N`, `--no-progress`.

**Set up**: an empty checkpoint dir. `mkdir -p "$RUN"` and let the trainer create `$RUN/ckpt`. Nothing else: the replay buffer fills itself with self-play games before training starts.

**Drive**:

```bash
uv run python -m reinforce.train --mode tiny --steps 20 --checkpoint-dir "$RUN/ckpt" --no-progress > "$RUN/train.log" 2>&1 &
TRAIN_PID=$!; echo "$TRAIN_PID" >> "$RUN/pids"; wait "$TRAIN_PID"; echo "exit=$?"
```

**Proof**:

- `train.log` shows, in order: `Using tiny configuration`, `Initialized new training state at step 0`, `Buffer filled: 10 games`, `Starting training for 20 steps...`, `Training Complete`, `Total steps: 20`, then a `Final Evaluation (10 games)` block ending in `Best Tile Achieved: <tile>`.
- `$RUN/ckpt/20/default/` exists with `_METADATA` and `manifest.ocdbt`.

**Gotchas**:

- Without `--checkpoint-dir`, the run resumes from and writes into the repo's `checkpoints/`.
- Measured on an Apple Silicon CPU: tiny + 20 steps ≈ 15 s wall clock; `small` (`make train-small`, 10 000 steps) is about 10 min; `full` takes hours to days. Use `tiny` unless the change is config-specific.
- `--mode small` and `full` need far longer buffer fills (`min_buffer_size` 100 and 1000 games).
- The final evaluation runs 10 games inside the same process; it is part of the proof, not a separate step.
- Checkpoints are kept every `checkpoint_interval` (100 in tiny) plus the final step; with `--steps 20` only `20/` is written.
