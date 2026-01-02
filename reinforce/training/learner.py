"""
Training step and state management for Stochastic MuZero.

This module provides the core training infrastructure:
- TrainState: Container for all training state
- Gradient computation and optimization step
- Checkpointing via orbax
"""

from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from reinforce.mcts.stochastic_mctx import NetworkParams
from reinforce.neural.network import StochasticMuZeroNetwork, create_network, update_params
from reinforce.training.config import TrainConfig
from reinforce.training.losses import LossOutput, TrainingTargets, compute_loss

# ##>: Type aliases.
Array = jax.Array
PRNGKey = jax.Array


class TrainState(NamedTuple):
    """
    Container for all training state.

    Attributes
    ----------
    network : StochasticMuZeroNetwork
        The neural network with parameters and apply functions.
    opt_state : optax.OptState
        Optimizer state.
    step : int
        Current training step.
    key : PRNGKey
        Current random key.
    optimizer : optax.GradientTransformation
        The optimizer (stored to avoid recreation each step).
    """

    network: StochasticMuZeroNetwork
    opt_state: Any
    step: int
    key: PRNGKey
    optimizer: optax.GradientTransformation = None  # type: ignore[assignment]


def create_optimizer(config: TrainConfig) -> optax.GradientTransformation:
    """
    Create the optimizer with learning rate schedule.

    Parameters
    ----------
    config : TrainConfig
        Training configuration.

    Returns
    -------
    optax.GradientTransformation
        Configured optimizer.
    """
    # ##>: Learning rate schedule with warmup.
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )

    # ##>: Constant after warmup.
    constant_fn = optax.constant_schedule(config.learning_rate)

    schedule = optax.join_schedules(
        schedules=[warmup_fn, constant_fn],
        boundaries=[config.warmup_steps],
    )

    # ##>: Optimizer chain: gradient clipping + Adam.
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=schedule),
    )

    return optimizer


def create_train_state(config: TrainConfig, key: PRNGKey) -> TrainState:
    """
    Initialize training state.

    Parameters
    ----------
    config : TrainConfig
        Training configuration.
    key : PRNGKey
        Random key for initialization.

    Returns
    -------
    TrainState
        Initialized training state.
    """
    key, net_key = jax.random.split(key)

    # ##>: Create network.
    network = create_network(
        key=net_key,
        observation_shape=config.observation_shape,
        hidden_size=config.hidden_size,
        num_blocks=config.num_residual_blocks,
        num_actions=config.action_size,
        codebook_size=config.codebook_size,
    )

    # ##>: Create optimizer and store it in state to avoid recreation each step.
    optimizer = create_optimizer(config)
    opt_state = optimizer.init(network.params)

    return TrainState(
        network=network,
        opt_state=opt_state,
        step=0,
        key=key,
        optimizer=optimizer,
    )


def train_step(
    state: TrainState,
    batch: TrainingTargets,
    config: TrainConfig,
    weights: Array | None = None,
) -> tuple[TrainState, LossOutput]:
    """
    Perform a single training step (legacy version, recreates optimizer).

    Parameters
    ----------
    state : TrainState
        Current training state.
    batch : TrainingTargets
        Batch of training data.
    config : TrainConfig
        Training configuration.
    weights : Array | None
        Importance-sampling weights for prioritized experience replay.
        Shape (batch_size,). If None, uniform weighting is used.

    Returns
    -------
    tuple[TrainState, LossOutput]
        - Updated training state
        - Loss metrics

    Notes
    -----
    This is the legacy version that recreates the optimizer each step.
    Use train_step_optimized for better performance.
    """
    optimizer = create_optimizer(config)

    def loss_fn(params: NetworkParams) -> tuple[Array, LossOutput]:
        return compute_loss(
            params=params,
            apply_fns=state.network.apply_fns,
            batch=batch,
            config=config,
            weights=weights,
        )

    # ##>: Compute gradients.
    (loss, loss_output), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.network.params)

    # ##>: Apply gradients.
    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.network.params)
    new_params = optax.apply_updates(state.network.params, updates)

    # ##>: Update network.
    # ##>: Cast needed because optax.apply_updates returns ArrayTree.
    new_network = update_params(state.network, cast(NetworkParams, new_params))

    # ##>: Update state.
    new_state = TrainState(
        network=new_network,
        opt_state=new_opt_state,
        step=state.step + 1,
        key=state.key,
        optimizer=state.optimizer,
    )

    return new_state, loss_output


@partial(jax.jit, static_argnums=(2,))
def train_step_jit(
    state: TrainState,
    batch: TrainingTargets,
    config: TrainConfig,
) -> tuple[TrainState, LossOutput]:
    """
    JIT-compiled training step (legacy version).

    Same as train_step but with JIT compilation for maximum performance.
    Note: config must be static (hashable) for this to work.
    """
    return train_step(state, batch, config)


def _train_step_core(
    params: NetworkParams,
    apply_fns: Any,
    opt_state: Any,
    batch: TrainingTargets,
    config: TrainConfig,
    optimizer: optax.GradientTransformation,
    weights: Array | None = None,
) -> tuple[NetworkParams, Any, LossOutput]:
    """
    Core training computation (JIT-friendly).

    Separated from state management for cleaner JIT compilation.
    """

    def loss_fn(p: NetworkParams) -> tuple[Array, LossOutput]:
        return compute_loss(params=p, apply_fns=apply_fns, batch=batch, config=config, weights=weights)

    # ##>: Compute gradients.
    (loss, loss_output), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # ##>: Apply gradients using stored optimizer.
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return cast(NetworkParams, new_params), new_opt_state, loss_output


# ##>: JIT-compile the core training computation.
# ##>: static_argnums=(1, 4, 5) for apply_fns, config and optimizer which are static/unchanging.
_train_step_core_jit = jax.jit(_train_step_core, static_argnums=(1, 4, 5))


def train_step_optimized(
    state: TrainState,
    batch: TrainingTargets,
    config: TrainConfig,
    weights: Array | None = None,
) -> tuple[TrainState, LossOutput]:
    """
    Optimized training step with JIT compilation and optimizer reuse.

    This version:
    1. Uses JIT-compiled core computation
    2. Reuses the optimizer stored in TrainState
    3. Minimizes Python overhead

    Parameters
    ----------
    state : TrainState
        Current training state (must have optimizer set).
    batch : TrainingTargets
        Batch of training data.
    config : TrainConfig
        Training configuration.
    weights : Array | None
        Importance-sampling weights for prioritized experience replay.
        Shape (batch_size,). If None, uniform weighting is used.

    Returns
    -------
    tuple[TrainState, LossOutput]
        - Updated training state
        - Loss metrics
    """
    # ##>: Run JIT-compiled core computation.
    new_params, new_opt_state, loss_output = _train_step_core_jit(
        state.network.params,
        state.network.apply_fns,
        state.opt_state,
        batch,
        config,
        state.optimizer,
        weights,
    )

    # ##>: Update network with new params.
    new_network = update_params(state.network, new_params)

    # ##>: Create new state (lightweight Python operation).
    new_state = TrainState(
        network=new_network,
        opt_state=new_opt_state,
        step=state.step + 1,
        key=state.key,
        optimizer=state.optimizer,
    )

    return new_state, loss_output


def compute_gradient_stats(grads: Any) -> dict:
    """
    Compute gradient statistics for logging.

    Parameters
    ----------
    grads : Any
        Gradient pytree.

    Returns
    -------
    dict
        Gradient statistics.
    """
    flat_grads = jax.tree.leaves(grads)
    all_grads = jnp.concatenate([g.flatten() for g in flat_grads])

    return {
        'grad_norm': float(jnp.linalg.norm(all_grads)),
        'grad_mean': float(jnp.mean(all_grads)),
        'grad_std': float(jnp.std(all_grads)),
        'grad_max': float(jnp.max(jnp.abs(all_grads))),
    }


class CheckpointManager:
    """
    Manager for saving and loading checkpoints.

    Uses orbax for efficient checkpoint management with async saving.

    Attributes
    ----------
    checkpoint_dir : Path
        Directory for checkpoints.
    max_to_keep : int
        Maximum number of checkpoints to retain.
    """

    def __init__(self, checkpoint_dir: str | Path, max_to_keep: int = 5):
        """
        Initialize checkpoint manager.

        Parameters
        ----------
        checkpoint_dir : str | Path
            Directory for checkpoints.
        max_to_keep : int
            Maximum checkpoints to keep.
        """
        # ##>: Orbax requires absolute paths for async TensorStore operations.
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep

        # ##>: Create orbax checkpoint manager.
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=1,
        )
        self.manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            options=options,
        )

    def save(self, state: TrainState, step: int) -> None:
        """
        Save a checkpoint.

        Parameters
        ----------
        state : TrainState
            Training state to save.
        step : int
            Training step number.
        """
        # ##>: Extract saveable parts.
        checkpoint = {
            'params': state.network.params,
            'opt_state': state.opt_state,
            'step': state.step,
            'key': state.key,
            'config': state.network.config,
        }

        self.manager.save(
            step,
            args=ocp.args.StandardSave(checkpoint),
        )

    def load(self, step: int | None = None) -> dict | None:
        """
        Load a checkpoint.

        Parameters
        ----------
        step : int | None
            Step to load. If None, loads latest.

        Returns
        -------
        dict | None
            Checkpoint data, or None if not found.
        """
        if step is None:
            step = self.manager.latest_step()

        if step is None:
            return None

        # ##>: Cast needed for orbax restore return type.
        return cast(dict | None, self.manager.restore(step))

    def restore_train_state(
        self,
        config: TrainConfig,
        step: int | None = None,
    ) -> TrainState | None:
        """
        Restore a TrainState from checkpoint.

        Parameters
        ----------
        config : TrainConfig
            Training configuration.
        step : int | None
            Step to restore. If None, restores latest.

        Returns
        -------
        TrainState | None
            Restored training state, or None if no checkpoint found.
        """
        checkpoint = self.load(step)
        if checkpoint is None:
            return None

        # ##>: Reconstruct network.
        network = create_network(
            key=checkpoint['key'],
            observation_shape=config.observation_shape,
            hidden_size=config.hidden_size,
            num_blocks=config.num_residual_blocks,
            num_actions=config.action_size,
            codebook_size=config.codebook_size,
        )
        network = update_params(network, checkpoint['params'])

        # ##>: Recreate optimizer for restored state.
        optimizer = create_optimizer(config)

        return TrainState(
            network=network,
            opt_state=checkpoint['opt_state'],
            step=checkpoint['step'],
            key=checkpoint['key'],
            optimizer=optimizer,
        )

    @property
    def latest_step(self) -> int | None:
        """Get the latest checkpoint step."""
        return self.manager.latest_step()
