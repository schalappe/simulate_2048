"""
Training state, training step, and orbax checkpoints for Stochastic MuZero.

The optimizer and the network apply functions are rebuilt from the configuration (and cached), so a
checkpoint only holds arrays: parameters, optimizer state, step, and random key.
"""

from functools import cache, partial
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import optax
import orbax.checkpoint as ocp

from reinforce.neural.network import NetworkApplyFns, NetworkParams, init_params, make_apply_fns
from reinforce.training.config import TrainConfig
from reinforce.training.losses import LossOutput, TrainingBatch, compute_loss

# ##>: Type aliases.
Array = jax.Array
PRNGKey = jax.Array


class TrainState(NamedTuple):
    """
    Everything a checkpoint must hold to resume training.

    Attributes
    ----------
    params : NetworkParams
        Network parameters.
    opt_state : Any
        Optimizer state, including Adam moments and the learning-rate schedule count.
    step : Array
        Number of training steps done, as a scalar int32 array.
    key : PRNGKey
        Random key for self-play and evaluation.
    """

    params: NetworkParams
    opt_state: Any
    step: Array
    key: PRNGKey


def apply_fns_for(config: TrainConfig) -> NetworkApplyFns:
    """
    Get the network apply functions for a configuration.

    Parameters
    ----------
    config : TrainConfig
        Training configuration.

    Returns
    -------
    NetworkApplyFns
        Cached apply functions for the configured architecture.
    """
    return make_apply_fns(config.hidden_size, config.num_residual_blocks, config.action_size, config.codebook_size)


@cache
def create_optimizer(config: TrainConfig) -> optax.GradientTransformation:
    """
    Create the optimizer: gradient clipping, then Adam with linear warmup to a constant learning rate.

    Parameters
    ----------
    config : TrainConfig
        Training configuration.

    Returns
    -------
    optax.GradientTransformation
        The same object for the same configuration, safe to pass as a static jit argument.
    """
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=0.0, end_value=config.learning_rate, transition_steps=config.warmup_steps),
            optax.constant_schedule(config.learning_rate),
        ],
        boundaries=[config.warmup_steps],
    )
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=schedule),
    )


def create_train_state(config: TrainConfig, key: PRNGKey) -> TrainState:
    """
    Initialize a fresh training state.

    Parameters
    ----------
    config : TrainConfig
        Training configuration.
    key : PRNGKey
        Random key for initialization.

    Returns
    -------
    TrainState
        State at step 0.
    """
    key, net_key = jax.random.split(key)
    params = init_params(
        net_key, config.hidden_size, config.num_residual_blocks, config.action_size, config.codebook_size
    )
    opt_state = create_optimizer(config).init(params)
    return TrainState(params=params, opt_state=opt_state, step=jax.numpy.int32(0), key=key)


@partial(jax.jit, static_argnums=(2,), donate_argnums=(0,))
def train_step(state: TrainState, batch: TrainingBatch, config: TrainConfig) -> tuple[TrainState, LossOutput]:
    """
    Apply one gradient step.

    Parameters
    ----------
    state : TrainState
        Current state. Its buffers are donated: do not reuse it after the call.
    batch : TrainingBatch
        Batch from the replay buffer.
    config : TrainConfig
        Training configuration (static).

    Returns
    -------
    tuple[TrainState, LossOutput]
        - Updated state
        - Batch losses
    """
    apply_fns = apply_fns_for(config)
    optimizer = create_optimizer(config)

    def loss_fn(params: NetworkParams) -> tuple[Array, LossOutput]:
        return compute_loss(params, apply_fns, batch, config)

    (_, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    params = cast(NetworkParams, optax.apply_updates(state.params, updates))
    return state._replace(params=params, opt_state=opt_state, step=state.step + 1), losses


class CheckpointManager:
    """
    Save and restore training states with orbax.

    Saves run asynchronously; call `wait()` before the process exits.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory for checkpoints.
    max_to_keep : int
        Number of checkpoints to retain.
    """

    def __init__(self, checkpoint_dir: str | Path, max_to_keep: int = 3):
        # ##>: Orbax requires absolute paths for async TensorStore operations.
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep),
        )

    @property
    def latest_step(self) -> int | None:
        """Return the latest saved step, or None."""
        return self.manager.latest_step()

    def save(self, state: TrainState) -> None:
        """
        Start saving a training state under its step number.

        Saving a step that already exists does nothing, so a resumed run never overwrites a checkpoint.

        Parameters
        ----------
        state : TrainState
            State to save.
        """
        step = int(state.step)
        if step in self.manager.all_steps():
            return
        self.manager.save(step, args=ocp.args.StandardSave(state._asdict()))

    def restore(self, config: TrainConfig, step: int | None = None) -> TrainState | None:
        """
        Restore a training state with its full structure, including optimizer state.

        Parameters
        ----------
        config : TrainConfig
            Configuration the checkpoint was trained with.
        step : int | None
            Step to restore; the latest if None.

        Returns
        -------
        TrainState | None
            The restored state, or None if the directory holds no checkpoint.
        """
        if step is None:
            step = self.manager.latest_step()
        if step is None:
            return None
        template = jax.eval_shape(lambda: create_train_state(config, jax.random.PRNGKey(0)))
        restored = self.manager.restore(step, args=ocp.args.StandardRestore(template._asdict()))
        return TrainState(**restored)

    def wait(self) -> None:
        """Block until pending saves are written."""
        self.manager.wait_until_finished()

    def close(self) -> None:
        """Wait for pending saves and release resources."""
        self.manager.close()
