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
  has the CUDA floor above.
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
