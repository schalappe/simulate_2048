# Resume training from a run directory

**Reach**: rerun `uv run python -m reinforce.train` with a `--run-dir` that already holds a run. There is no `--resume` flag: resuming is automatic.

**Set up**: a finished tiny run in `$RUN/run` (see `train-from-scratch.md`; it leaves `$RUN/run/checkpoints/20/`).

**Drive**:

```bash
uv run python -m reinforce.train --mode tiny --steps 5 --run-dir "$RUN/run" --allow-cpu > "$RUN/resume.log" 2>&1 &
RESUME_PID=$!; echo "$RESUME_PID" >> "$RUN/pids"; wait "$RESUME_PID"; echo "exit=$?"
ls "$RUN/run/checkpoints"
```

**Proof**:

- `resume.log` contains `Restored training state from step 20` and `Restored replay buffer: <n> sequences, <m> positions` (not `Initialized new training state`).
- `ls "$RUN/run/checkpoints"` lists both `20` and `25`: `--steps` is added to the restored step.

**Gotchas**:

- The restore keeps the optimizer state (Adam moments and learning-rate schedule), so the warmup does not restart.
- Games in progress at the last save are not stored; a resumed run starts new games.
- The run directory pins its configuration in `config.json`. Resuming with another `--mode` raises `ValueError: ... holds a run started with a different configuration` before any training.
