"""
Unified network wrapper for Stochastic MuZero models.

This module provides a high-level interface for working with all six Stochastic MuZero networks,
handling initialization, parameter management, and inference.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from reinforce.mcts.stochastic_mctx import NetworkApplyFns, NetworkParams
from reinforce.neural.models import (
    HIDDEN_UNITS,
    NUM_RESIDUAL_BLOCKS,
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

# ##>: Default configuration.
NUM_ACTIONS = 4
DEFAULT_CODEBOOK_SIZE = 32


class StochasticMuZeroNetwork(NamedTuple):
    """
    Container for all Stochastic MuZero network components.

    Attributes
    ----------
    params : NetworkParams
        Parameters for all six models.
    apply_fns : NetworkApplyFns
        Functions to apply each model.
    config : dict
        Network configuration (hidden_size, codebook_size, etc.).
    """

    params: NetworkParams
    apply_fns: NetworkApplyFns
    config: dict


def create_network(
    key: PRNGKey,
    observation_shape: tuple[int, ...] = (16,),
    hidden_size: int = HIDDEN_UNITS,
    num_blocks: int = NUM_RESIDUAL_BLOCKS,
    num_actions: int = NUM_ACTIONS,
    codebook_size: int = DEFAULT_CODEBOOK_SIZE,
) -> StochasticMuZeroNetwork:
    """
    Create and initialize a complete Stochastic MuZero network.

    Parameters
    ----------
    key : PRNGKey
        JAX random key for parameter initialization.
    observation_shape : tuple[int, ...]
        Shape of the input observation.
    hidden_size : int
        Size of hidden state representations.
    num_blocks : int
        Number of residual blocks in each network.
    num_actions : int
        Number of possible actions.
    codebook_size : int
        Number of possible chance codes.

    Returns
    -------
    StochasticMuZeroNetwork
        Initialized network with parameters and apply functions.
    """
    # ##>: Split key for each model.
    keys = jax.random.split(key, 6)

    # ##>: Create model instances.
    representation = Representation(hidden_size=hidden_size, num_blocks=num_blocks)
    prediction = Prediction(action_size=num_actions, hidden_size=hidden_size, num_blocks=num_blocks)
    afterstate_dynamics = AfterstateDynamics(hidden_size=hidden_size, action_size=num_actions, num_blocks=num_blocks)
    afterstate_prediction = AfterstatePrediction(
        codebook_size=codebook_size, hidden_size=hidden_size, num_blocks=num_blocks
    )
    dynamics = Dynamics(hidden_size=hidden_size, codebook_size=codebook_size, num_blocks=num_blocks)
    encoder = Encoder(codebook_size=codebook_size, hidden_size=hidden_size, num_blocks=num_blocks)

    # ##>: Create dummy inputs for initialization.
    dummy_obs = jnp.zeros((1, *observation_shape))
    dummy_state = jnp.zeros((1, hidden_size))
    dummy_action = jnp.zeros((1, num_actions))
    dummy_chance = jnp.zeros((1, codebook_size))

    # ##>: Initialize parameters.
    representation_params = representation.init(keys[0], dummy_obs)
    prediction_params = prediction.init(keys[1], dummy_state)
    afterstate_dynamics_params = afterstate_dynamics.init(keys[2], dummy_state, dummy_action)
    afterstate_prediction_params = afterstate_prediction.init(keys[3], dummy_state)
    dynamics_params = dynamics.init(keys[4], dummy_state, dummy_chance)
    encoder_params = encoder.init(keys[5], dummy_obs)

    # ##>: Package parameters.
    params = NetworkParams(
        representation=representation_params,
        prediction=prediction_params,
        afterstate_dynamics=afterstate_dynamics_params,
        afterstate_prediction=afterstate_prediction_params,
        dynamics=dynamics_params,
    )

    # ##>: Create apply functions.
    apply_fns = NetworkApplyFns(
        representation=lambda p, x: representation.apply(p, x),
        prediction=lambda p, x: prediction.apply(p, x),
        afterstate_dynamics=lambda p, s, a: afterstate_dynamics.apply(p, s, a),
        afterstate_prediction=lambda p, x: afterstate_prediction.apply(p, x),
        dynamics=lambda p, s, c: dynamics.apply(p, s, c),
    )

    # ##>: Configuration dict.
    config = {
        'observation_shape': observation_shape,
        'hidden_size': hidden_size,
        'num_blocks': num_blocks,
        'num_actions': num_actions,
        'codebook_size': codebook_size,
        'encoder_params': encoder_params,
    }

    return StochasticMuZeroNetwork(params=params, apply_fns=apply_fns, config=config)


@jax.jit
def representation_forward(network: StochasticMuZeroNetwork, observation: Array) -> Array:
    """
    Encode observation to hidden state.

    Parameters
    ----------
    network : StochasticMuZeroNetwork
        The network wrapper.
    observation : Array
        Input observation.

    Returns
    -------
    Array
        Hidden state representation.
    """
    return network.apply_fns.representation(network.params.representation, observation)


@jax.jit
def prediction_forward(network: StochasticMuZeroNetwork, state: Array) -> tuple[Array, Array]:
    """
    Predict policy and value from hidden state.

    Parameters
    ----------
    network : StochasticMuZeroNetwork
        The network wrapper.
    state : Array
        Hidden state.

    Returns
    -------
    tuple[Array, Array]
        (policy_logits, value)
    """
    return network.apply_fns.prediction(network.params.prediction, state)


@jax.jit
def afterstate_dynamics_forward(network: StochasticMuZeroNetwork, state: Array, action: Array) -> Array:
    """
    Predict afterstate from state and action.

    Parameters
    ----------
    network : StochasticMuZeroNetwork
        The network wrapper.
    state : Array
        Hidden state.
    action : Array
        One-hot encoded action.

    Returns
    -------
    Array
        Afterstate.
    """
    return network.apply_fns.afterstate_dynamics(network.params.afterstate_dynamics, state, action)


@jax.jit
def afterstate_prediction_forward(network: StochasticMuZeroNetwork, afterstate: Array) -> tuple[Array, Array]:
    """
    Predict Q-value and chance distribution from afterstate.

    Parameters
    ----------
    network : StochasticMuZeroNetwork
        The network wrapper.
    afterstate : Array
        Afterstate.

    Returns
    -------
    tuple[Array, Array]
        (q_value, chance_logits)
    """
    return network.apply_fns.afterstate_prediction(network.params.afterstate_prediction, afterstate)


@jax.jit
def dynamics_forward(network: StochasticMuZeroNetwork, afterstate: Array, chance_code: Array) -> tuple[Array, Array]:
    """
    Predict next state and reward from afterstate and chance code.

    Parameters
    ----------
    network : StochasticMuZeroNetwork
        The network wrapper.
    afterstate : Array
        Afterstate.
    chance_code : Array
        One-hot encoded chance outcome.

    Returns
    -------
    tuple[Array, Array]
        (next_state, reward)
    """
    return network.apply_fns.dynamics(network.params.dynamics, afterstate, chance_code)


def encoder_forward(network: StochasticMuZeroNetwork, observation: Array) -> Array:
    """
    Encode observation to chance code.

    Parameters
    ----------
    network : StochasticMuZeroNetwork
        The network wrapper.
    observation : Array
        Input observation.

    Returns
    -------
    Array
        One-hot chance code.
    """
    encoder = Encoder(
        codebook_size=network.config['codebook_size'],
        hidden_size=network.config['hidden_size'],
        num_blocks=network.config['num_blocks'],
    )
    return encoder.apply(network.config['encoder_params'], observation)


def get_all_params(network: StochasticMuZeroNetwork) -> dict:
    """
    Get all network parameters as a flat dictionary.

    Parameters
    ----------
    network : StochasticMuZeroNetwork
        The network wrapper.

    Returns
    -------
    dict
        Dictionary containing all model parameters.
    """
    return {
        'representation': network.params.representation,
        'prediction': network.params.prediction,
        'afterstate_dynamics': network.params.afterstate_dynamics,
        'afterstate_prediction': network.params.afterstate_prediction,
        'dynamics': network.params.dynamics,
        'encoder': network.config['encoder_params'],
    }


def count_parameters(network: StochasticMuZeroNetwork) -> int:
    """
    Count total number of trainable parameters.

    Parameters
    ----------
    network : StochasticMuZeroNetwork
        The network wrapper.

    Returns
    -------
    int
        Total parameter count.
    """
    all_params = get_all_params(network)

    def count_leaves(pytree):
        leaves = jax.tree.leaves(pytree)
        return sum(leaf.size for leaf in leaves)

    return sum(count_leaves(params) for params in all_params.values())


def update_params(network: StochasticMuZeroNetwork, new_params: NetworkParams) -> StochasticMuZeroNetwork:
    """
    Create a new network with updated parameters.

    Parameters
    ----------
    network : StochasticMuZeroNetwork
        Original network.
    new_params : NetworkParams
        New parameters.

    Returns
    -------
    StochasticMuZeroNetwork
        Network with updated parameters.
    """
    return StochasticMuZeroNetwork(
        params=new_params,
        apply_fns=network.apply_fns,
        config=network.config,
    )
