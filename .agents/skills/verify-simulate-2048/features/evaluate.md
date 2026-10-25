# Evaluate the agent

**Reach**: `uv run python -m reinforce.evaluate` (also `make eval`). Flags: `--games N`, `--simulations N`, `--temperature T`, `--checkpoint PATH`.

**Set up**: none. Evaluation builds a fresh network from `small_config()` with a fixed seed (42) and writes nothing to disk.

**Drive**:

```bash
uv run python -m reinforce.evaluate --games 2 --simulations 8 > "$RUN/eval.log" 2>&1 &
EVAL_PID=$!; echo "$EVAL_PID" >> "$RUN/pids"; wait "$EVAL_PID"; echo "exit=$?"
grep -A20 'Evaluation Results' "$RUN/eval.log"
```

**Proof**: an `Evaluation Results (Stochastic MuZero):` block with `Games played: 2`, mean/max reward, mean max tile, best tile, mean game length, and a `Tile frequency distribution:` whose counts sum to the games played.

**Gotchas**:

- **`--checkpoint` is a no-op.** `reinforce/evaluate.py` prints `Warning: Checkpoint loading from ... not yet implemented` and `Using untrained network`, then evaluates the untrained net. README and `make eval-checkpoint` advertise it anyway. You cannot prove a trained model's strength through this CLI today. If your change claims to, report that the journey is missing; do not swap in a different code path and call it evaluation. To score trained weights, use the final evaluation printed by `reinforce.train` (see `train-from-scratch.md`).
- Measured: 1 game at 4 simulations ≈ 5 s wall clock, mostly JIT compile. Time grows with `--simulations` and game length.
- The log is tqdm carriage-return output on stderr; read it with `grep` or `tr '\r' '\n'`.
