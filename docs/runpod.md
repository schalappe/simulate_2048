# Training on RunPod

Full training (`--mode full`) runs on a RunPod GPU pod. Local machines only run the tiny preset on CPU to prove
the code works. This page lists what the pod needs, how to start and resume a run, and what to watch.

RunPod facts below come from the official RunPod guides (`pod-deploy`, network volume concept), read on
2026-10-25. Prices change: read them in the console or the catalog before creating anything.

## Shape of the pod

| Choice | Value | Why |
|---|---|---|
| Cloud | Secure Cloud | Network volumes attach only to Secure Cloud pods. |
| GPU | One NVIDIA GPU | Self-play and training run on one device. |
| CUDA floor | `minCudaVersion` 12.8 | The `gpu` extra installs CUDA 12 wheels; the floor keeps the pod off hosts whose driver is too old. |
| Image | A `runpod/pytorch` image on CUDA 12, exact tag from the catalog | Only the NVIDIA driver and Python tooling are needed; JAX brings its own CUDA libraries. RunPod publishes no `:latest` tag. |
| Storage | Network volume mounted at `/workspace`, 10 GB or more | It lives in the data center and survives stopping or deleting the pod. Create it in the same data center as the pod. |

What a full run stores on the volume:

- `replay_buffer.npz`: about 1.4 GB, whatever its fill level (the arrays are preallocated).
- `checkpoints/`: the 3 most recent checkpoints. Each holds the parameters and Adam's two moment estimates.
- `metrics.jsonl` and `train.log`: small.

A 10 GB volume (the minimum size) leaves room for several runs. A volume can grow but never shrink, and it bills
per GB from creation until deletion, attached or not.

## First start

From a shell on the pod (web terminal or SSH):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
cd /workspace
git clone https://github.com/schalappe/simulate_2048.git
cd simulate_2048
git checkout fix/stochastic-muzero-training   # until it is merged into main

# ##>: Keep the virtual environment on the container disk: imports from a network volume are slow.
export UV_PROJECT_ENVIRONMENT=/root/venv
# ##!: RunPod PyTorch images set LD_LIBRARY_PATH=/usr/local/cuda/lib64. JAX's pip cuSPARSE then links the
# ##!: image's older libnvJitLink and fails to load, and JAX falls back to the CPU. Unset it in every shell.
unset LD_LIBRARY_PATH
uv sync --extra gpu
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"   # must print gpu
```

Then start training in the background, so closing the terminal does not stop it:

```bash
mkdir -p /workspace/runs/full
nohup uv run python -m reinforce.train --mode full --run-dir /workspace/runs/full \
    > /workspace/runs/full/train.log 2>&1 &
```

## What to watch

- **The first line of `train.log` must read `JAX backend: gpu`.** Without a GPU, training stops at once with
  `No GPU visible to JAX`. That guard exists because JAX otherwise falls back to the CPU with only a warning, and
  the pod would bill GPU time for CPU speed. If it fires, check that `uv sync --extra gpu` ran and that the pod
  has the CUDA floor above. On RunPod PyTorch images the usual cause is `LD_LIBRARY_PATH`: run
  `unset LD_LIBRARY_PATH` in the shell that starts training (see First start). Observed on
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`: `Unable to load cuSPARSE`, then `BACKEND cpu`.
- `tail -f /workspace/runs/full/train.log`: one line per `log_interval` steps with the loss, training steps per
  second, self-play moves per second, buffer size, and games played. Evaluation lines start with `[eval]`.
- `/workspace/runs/full/metrics.jsonl`: the same numbers as JSON, one record per line (`"kind": "train"` or
  `"kind": "eval"`), for plots.
- `nvidia-smi`: the GPU should stay busy. Low use while `moves/s` is low means self-play is the bottleneck; tune
  `num_parallel_games` and `generation_interval` in `reinforce/training/config.py`.

## Resume after a stop or restart

Run the same command with the same `--run-dir`. The run restores the network, optimizer state, step, random
key, and replay buffer, then continues. Games in progress at the last save are lost; new games start.
A run directory refuses a different `--mode`, so a resume cannot silently change the configuration.

Saves happen every `checkpoint_interval` steps and on Ctrl-C. The replay buffer is written to a temporary file
and then renamed, and orbax writes checkpoints the same way, so a pod that dies in the middle of a save keeps
the previous save.

## Measured speed and cost

Smoke test on 2026-10-25: one NVIDIA A40 (48 GB), Secure Cloud, EU-SE-1, image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, host CUDA 13.2, catalog price $0.59/h.

| Measurement (`--mode full`) | Value |
|---|---|
| Tests (`pytest`, on GPU) | 86 passed in 2 min 25 s |
| Self-play chunk (256 games × 50 moves, 100 simulations) | about 7 s, about 1,800 moves/s |
| Buffer fill to 1,000 sequences | 71 s, after about 35 s of compilation |
| Training, steady state (self-play included) | 0.44 steps/s (100 steps in 229 s) |
| Checkpoint + replay buffer save | about 7 s; checkpoint 112 MB, buffer 1.3 GB |
| Evaluation, 16 greedy games | about 2.5 min |
| Resume after a pod restart | restored step 300 and 3,862 sequences, then trained on |

Cost of a run on that A40 at 0.44 steps/s (1,584 steps per hour), GPU only; the volume bills separately:

| Training steps | GPU time | Cost |
|---|---|---|
| 100,000 | 63 h (2.6 days) | about $37 |
| 300,000 (end of the temperature schedule) | 189 h (7.9 days) | about $112 |
| 1,000,000 | 631 h (26 days) | about $372 |
| 20,000,000 (paper) | 12,626 h (526 days) | about $7,450 |

The paper's 20M steps are out of reach on one GPU: the authors used 1 TPU for training and 4 for acting
during 80 hours (Appendix G). The measured step is far slower than the arithmetic of the network suggests,
so profile one train step and one self-play chunk before committing to a long run.

## Budget run: prove the model reaches 2048

Goal: show that a Stochastic MuZero model reaches tile 2048 within a fixed budget, not reproduce the paper.
`--mode budget` uses 128 × 4 models, 50 simulations, and a search depth capped at 10.

`scripts/runpod_budget_run.sh` is the pod's start command. It trains with `--stop-at-tile 2048` and a time limit,
then stops the pod itself with `runpodctl stop pod $RUNPOD_POD_ID`, whether the target was reached, the time ran
out, or training crashed. The time budget (`BUDGET_HOURS`) counts from the first boot, stored in
`/workspace/start_epoch`, so a restarted pod only gets the time that is left. Create the pod with a persistent
`/workspace` and this start command, pinned to a commit:

```bash
bash -c "curl -fsSL https://raw.githubusercontent.com/schalappe/simulate_2048/<commit>/scripts/runpod_budget_run.sh | bash"
```

Reading the result: the run first evaluates its untrained network (`[baseline]`), because search alone already
plays better than the network. The proof is an `[eval]` line, after training, with `games reaching 2048` above 0
while the baseline had none. Everything is in `/workspace/run.log` and `runs/budget/metrics.jsonl`.

Measured on the A40 (2026-10-25) with the loop instrumented phase by phase: self-play is about 95% of the loop.
Search depth grows as the chance distribution sharpens (mean node depth 2.3 untrained, 10.9 after 400 tiny steps),
which made a self-play chunk 2.5 to 3 times slower than with an untrained network; `max_search_depth` caps it.

## Bring a run back and evaluate it locally

Copy the run directory from the volume (for example with `runpodctl send` / `runpodctl receive`, or the S3
API in data centers that offer it), then:

```bash
uv run python -m reinforce.evaluate --run-dir path/to/full --games 20
```

The replay buffer is not needed for evaluation; `config.json` and `checkpoints/` are enough.

## Billing reminders

- A running pod bills its GPU by the hour, used or not.
- A stopped pod stops GPU billing. The network volume keeps billing until you delete it.
- Deleting a network volume is permanent. Copy the run directory off first.
- If the account balance reaches $0, pods stop; data on a network volume stays, but storage keeps accruing.
