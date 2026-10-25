# Evaluate the agent

**Reach**: `uv run python -m reinforce.evaluate` (also `make eval`, `make eval-run RUN_DIR=...`). Flags: `--run-dir DIR`, `--step N`, `--mode tiny|small|full` (only without `--run-dir`), `--games N`, `--simulations N`, `--seed N`.

**Set up**: none for an untrained network. To evaluate trained weights, a run directory from `train-from-scratch.md`. Evaluation writes nothing to disk.

**Drive**:

```bash
# Untrained network (smoke check of the search).
uv run python -m reinforce.evaluate --mode tiny --games 2 --simulations 8 > "$RUN/eval.log" 2>&1 &
EVAL_PID=$!; echo "$EVAL_PID" >> "$RUN/pids"; wait "$EVAL_PID"; echo "exit=$?"
grep -A20 'Evaluation Results' "$RUN/eval.log"

# Trained run.
uv run python -m reinforce.evaluate --run-dir "$RUN/run" --games 2 > "$RUN/eval-run.log" 2>&1 &
EVAL_PID=$!; echo "$EVAL_PID" >> "$RUN/pids"; wait "$EVAL_PID"; echo "exit=$?"
grep -A20 'Evaluation Results' "$RUN/eval-run.log"
```

**Proof**:

- Untrained: `No --run-dir: evaluating an untrained network.`, then an `Evaluation Results (Stochastic MuZero):` block with `Games played: 2`, mean and max score, mean max tile, best tile, mean game length, `Truncated games:`, and a `Tile frequency distribution:` whose counts sum to the games played.
- Trained run: the same block, starting with `Checkpoint step: <step>` equal to the run's latest checkpoint.

**Gotchas**:

- Evaluation is greedy (temperature 0) with no root noise; two runs with the same `--seed` play the same games.
- Measured: 2 untrained tiny games at 8 simulations ≈ 5 s wall clock, mostly JIT compilation.
- `--run-dir` reads the configuration from the run's `config.json`; `--mode` is ignored there.
