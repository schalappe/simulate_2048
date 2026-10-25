# Resume training from a checkpoint

**Reach**: rerun `uv run python -m reinforce.train` with a `--checkpoint-dir` that already holds a checkpoint. There is no `--resume` flag: resuming is automatic.

**Set up**: a finished tiny run in `$RUN/ckpt` (see `train-from-scratch.md`; it leaves `$RUN/ckpt/20/`).

**Drive**:

```bash
uv run python -m reinforce.train --mode tiny --steps 5 --checkpoint-dir "$RUN/ckpt" --no-progress > "$RUN/resume.log" 2>&1 &
RESUME_PID=$!; echo "$RESUME_PID" >> "$RUN/pids"; wait "$RESUME_PID"; echo "exit=$?"
ls "$RUN/ckpt"
```

**Proof**:

- `resume.log` contains `Restored training state from step 20` (not `Initialized new training state`).
- `ls "$RUN/ckpt"` lists both `20` and `25`: `--steps` is added to the restored step.

**Gotchas**:

- Restore prints absl warnings: `Item "default" was found in the checkpoint, but could not be restored` and `StandardCheckpointHandler expects a target tree ... UNSAFE`. They are current, expected noise; the state is restored anyway. Treat them as a failure only if the `Restored ... from step` line is missing.
- The replay buffer is not checkpointed: a resumed run refills it with fresh self-play games.
- The config mode must match the checkpoint (network sizes differ between tiny, small, and full). Restoring a tiny checkpoint with `--mode small` is not a supported journey.
