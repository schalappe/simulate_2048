"""
Stochastic MuZero network container: parameters and apply functions for the six models.

`make_apply_fns` is cached per architecture, so every caller gets the same function objects and
jitted code that takes them as static arguments compiles once.
"""

from functools import cache
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from reinforce.game.core import OBSERVATION_SIZE
from reinforce.neural.models import (
    AfterstateDynamics,
    AfterstatePrediction,
    Dynamics,
    Encoder,
    Prediction,
    Representation,
)

# ##>: Type aliases.
Array = jax.Array
PRNGKey = jax.Array


class NetworkParams(NamedTuple):
    """
    Parameters of the six models.

    Every trainable parameter must live here: the optimizer only updates this tree.
    """

    representation: Any
    prediction: Any
    afterstate_dynamics: Any
    afterstate_prediction: Any
    dynamics: Any
    encoder: Any


class NetworkApplyFns(NamedTuple):
    """
    Apply functions of the six models. Each takes the model parameters first.

    Attributes
    ----------
    representation : (params, observation) -> hidden_state
    prediction : (params, hidden_state) -> (policy_logits, value_logits)
    afterstate_dynamics : (params, hidden_state, action_onehot) -> afterstate
    afterstate_prediction : (params, afterstate) -> (q_logits, chance_logits)
    dynamics : (params, afterstate, chance_code) -> (next_hidden_state, reward_logits)
    encoder : (params, observation_pair) -> chance_logits
    """

    representation: Any
    prediction: Any
    afterstate_dynamics: Any
    afterstate_prediction: Any
    dynamics: Any
    encoder: Any


class _Models(NamedTuple):
    """Flax module instances for one architecture."""

    representation: Representation
    prediction: Prediction
    afterstate_dynamics: AfterstateDynamics
    afterstate_prediction: AfterstatePrediction
    dynamics: Dynamics
    encoder: Encoder


@cache
def _make_models(hidden_size: int, num_blocks: int, num_actions: int, codebook_size: int) -> _Models:
    """Build the Flax modules for one architecture."""
    return _Models(
        representation=Representation(hidden_size=hidden_size, num_blocks=num_blocks),
        prediction=Prediction(action_size=num_actions, hidden_size=hidden_size, num_blocks=num_blocks),
        afterstate_dynamics=AfterstateDynamics(hidden_size=hidden_size, action_size=num_actions, num_blocks=num_blocks),
        afterstate_prediction=AfterstatePrediction(
            codebook_size=codebook_size, hidden_size=hidden_size, num_blocks=num_blocks
        ),
        dynamics=Dynamics(hidden_size=hidden_size, codebook_size=codebook_size, num_blocks=num_blocks),
        encoder=Encoder(codebook_size=codebook_size, hidden_size=hidden_size, num_blocks=num_blocks),
    )


@cache
def make_apply_fns(hidden_size: int, num_blocks: int, num_actions: int, codebook_size: int) -> NetworkApplyFns:
    """
    Get the apply functions for one architecture.

    Parameters
    ----------
    hidden_size : int
        Size of hidden states and layers.
    num_blocks : int
        Residual blocks per model.
    num_actions : int
        Number of actions.
    codebook_size : int
        Number of chance codes.

    Returns
    -------
    NetworkApplyFns
        The same object for the same architecture, safe to pass as a static jit argument.
    """
    models = _make_models(hidden_size, num_blocks, num_actions, codebook_size)
    return NetworkApplyFns(
        representation=models.representation.apply,
        prediction=models.prediction.apply,
        afterstate_dynamics=models.afterstate_dynamics.apply,
        afterstate_prediction=models.afterstate_prediction.apply,
        dynamics=models.dynamics.apply,
        encoder=models.encoder.apply,
    )


def init_params(
    key: PRNGKey,
    hidden_size: int,
    num_blocks: int,
    num_actions: int,
    codebook_size: int,
) -> NetworkParams:
    """
    Initialize the parameters of the six models.

    Parameters
    ----------
    key : PRNGKey
        Random key for initialization.
    hidden_size : int
        Size of hidden states and layers.
    num_blocks : int
        Residual blocks per model.
    num_actions : int
        Number of actions.
    codebook_size : int
        Number of chance codes.

    Returns
    -------
    NetworkParams
        Fresh parameters.
    """
    models = _make_models(hidden_size, num_blocks, num_actions, codebook_size)
    keys = jax.random.split(key, 6)

    dummy_obs = jnp.zeros((1, OBSERVATION_SIZE))
    dummy_pair = jnp.zeros((1, 2 * OBSERVATION_SIZE))
    dummy_state = jnp.zeros((1, hidden_size))
    dummy_action = jnp.zeros((1, num_actions))
    dummy_chance = jnp.zeros((1, codebook_size))

    return NetworkParams(
        representation=models.representation.init(keys[0], dummy_obs),
        prediction=models.prediction.init(keys[1], dummy_state),
        afterstate_dynamics=models.afterstate_dynamics.init(keys[2], dummy_state, dummy_action),
        afterstate_prediction=models.afterstate_prediction.init(keys[3], dummy_state),
        dynamics=models.dynamics.init(keys[4], dummy_state, dummy_chance),
        encoder=models.encoder.init(keys[5], dummy_pair),
    )


def count_parameters(params: NetworkParams) -> int:
    """
    Count trainable parameters.

    Parameters
    ----------
    params : NetworkParams
        Network parameters.

    Returns
    -------
    int
        Total number of scalars.
    """
    return sum(leaf.size for leaf in jax.tree.leaves(params))
