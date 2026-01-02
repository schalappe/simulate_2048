"""
Action selection utilities for MCTS policy outputs.

This module provides functions to convert mctx policy outputs into action selections
with temperature-based sampling.
"""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import mctx
from jax import lax

# ##>: Type aliases.
Array = jax.Array
PRNGKey = jax.Array


@partial(jax.jit, static_argnums=(2,))
def get_policy_target(policy_output: mctx.PolicyOutput, legal_mask: Array, temperature: float = 1.0) -> Array:
    """
    Convert MCTS visit counts to a policy target for training.

    Parameters
    ----------
    policy_output : mctx.PolicyOutput
        Output from mctx search containing action_weights.
    legal_mask : Array
        Boolean mask of legal actions, shape (num_actions,).
    temperature : float
        Temperature for softening the policy.
        - 1.0: proportional to visit counts
        - 0.0: deterministic (argmax)
        - >1.0: more uniform exploration

    Returns
    -------
    Array
        Normalized policy distribution, shape (num_actions,).
    """
    # ##>: action_weights are already normalized visit counts.
    weights = policy_output.action_weights

    # ##>: Mask illegal actions.
    masked_weights = jnp.where(legal_mask, weights, 0.0)

    # ##>: Apply temperature using log-based softmax (matches core.py:sample_action).
    def with_temperature(w: Array) -> Array:
        # ##&: JAX's lax.cond traces both branches, so guard against division by zero.
        safe_temp = jnp.maximum(temperature, 0.01)
        log_w = jnp.log(w + 1e-8)
        scaled = log_w / safe_temp
        return jax.nn.softmax(scaled)

    def without_temperature(w: Array) -> Array:
        # ##>: Greedy: return one-hot at max.
        return jax.nn.one_hot(jnp.argmax(w), w.shape[-1])

    is_greedy = temperature < 0.01
    policy = lax.cond(is_greedy, without_temperature, with_temperature, masked_weights)

    return policy


@partial(jax.jit, static_argnums=(3,))
def select_action(
    policy_output: mctx.PolicyOutput, key: PRNGKey, legal_mask: Array, temperature: float = 1.0
) -> Array:
    """
    Select an action from MCTS policy output.

    Parameters
    ----------
    policy_output : mctx.PolicyOutput
        Output from mctx search.
    key : PRNGKey
        JAX random key for sampling.
    legal_mask : Array
        Boolean mask of legal actions, shape (num_actions,).
    temperature : float
        Temperature for action selection.
        - 0.0: deterministic (best action)
        - 1.0: proportional to visit counts
        - >1.0: more exploration

    Returns
    -------
    Array
        Selected action index (scalar int).
    """
    policy = get_policy_target(policy_output, legal_mask, temperature)

    # ##>: Sample from the policy.
    def sample_action(p: Array) -> Array:
        return jax.random.choice(key, p.shape[-1], p=p)

    def greedy_action(p: Array) -> Array:
        return jnp.argmax(p)

    is_greedy = temperature < 0.01
    action = lax.cond(is_greedy, greedy_action, sample_action, policy)

    return action


@jax.jit
def get_search_value(policy_output: mctx.PolicyOutput) -> Array:
    """
    Extract the search value from MCTS output.

    The search value is the backed-up value at the root node, representing the expected
    return from the current state.

    Parameters
    ----------
    policy_output : mctx.PolicyOutput
        Output from mctx search.

    Returns
    -------
    Array
        Search value estimate (scalar float).
    """
    # ##>: Cast needed for pyrefly type inference with mctx types.
    return cast(Array, policy_output.search_tree.node_values[..., 0])  # pyrefly: ignore


@jax.jit
def get_visit_counts(policy_output: mctx.PolicyOutput) -> Array:
    """
    Get the raw visit counts for each action.

    Parameters
    ----------
    policy_output : mctx.PolicyOutput
        Output from mctx search.

    Returns
    -------
    Array
        Visit counts for each action, shape (num_actions,).
    """
    # ##>: Cast needed for pyrefly type inference with mctx types.
    return cast(Array, policy_output.search_tree.summary().visit_counts[..., 0, :])  # pyrefly: ignore


@jax.jit
def get_q_values(policy_output: mctx.PolicyOutput) -> Array:
    """
    Get the Q-values for each action from the search tree.

    Parameters
    ----------
    policy_output : mctx.PolicyOutput
        Output from mctx search.

    Returns
    -------
    Array
        Q-values for each action, shape (num_actions,).
    """
    # ##>: Cast needed for pyrefly type inference with mctx types.
    return cast(Array, policy_output.search_tree.summary().qvalues[..., 0, :])  # pyrefly: ignore


def batched_select_action(
    policy_outputs: mctx.PolicyOutput,
    keys: Array,
    legal_masks: Array,
    temperature: float = 1.0,
) -> Array:
    """
    Select actions for a batch of MCTS policy outputs.

    Parameters
    ----------
    policy_outputs : mctx.PolicyOutput
        Batched output from mctx search, with batch dimension in each field.
    keys : Array
        Batch of random keys, shape (batch_size, 2).
    legal_masks : Array
        Batch of legal action masks, shape (batch_size, num_actions).
    temperature : float
        Temperature for action selection (shared across batch).

    Returns
    -------
    Array
        Selected action indices, shape (batch_size,).
    """
    # ##>: Use vmap to apply select_action across the batch.
    # ##>: Temperature is static, so we use a lambda to capture it.
    return jax.vmap(lambda po, k, m: select_action(po, k, m, temperature))(policy_outputs, keys, legal_masks)


def batched_get_policy_target(
    policy_outputs: mctx.PolicyOutput,
    legal_masks: Array,
    temperature: float = 1.0,
) -> Array:
    """
    Convert batched MCTS visit counts to policy targets.

    Parameters
    ----------
    policy_outputs : mctx.PolicyOutput
        Batched output from mctx search.
    legal_masks : Array
        Batch of legal action masks, shape (batch_size, num_actions).
    temperature : float
        Temperature for softening the policy.

    Returns
    -------
    Array
        Normalized policy distributions, shape (batch_size, num_actions).
    """
    return jax.vmap(lambda po, m: get_policy_target(po, m, temperature))(policy_outputs, legal_masks)


def batched_get_search_value(policy_outputs: mctx.PolicyOutput) -> Array:
    """
    Extract search values from batched MCTS outputs.

    Parameters
    ----------
    policy_outputs : mctx.PolicyOutput
        Batched output from mctx search.

    Returns
    -------
    Array
        Search value estimates, shape (batch_size,).
    """
    return jax.vmap(get_search_value)(policy_outputs)


def temperature_schedule(step: int, schedule: list[tuple[int, float]]) -> float:
    """
    Get the temperature for a given training step.

    Parameters
    ----------
    step : int
        Current training step.
    schedule : list[tuple[int, float]]
        List of (step_threshold, temperature) pairs in ascending order.
        Temperature is the value for steps >= threshold and < next threshold.

    Returns
    -------
    float
        Temperature value.

    Examples
    --------
    >>> schedule = [(0, 1.0), (100_000, 0.5), (200_000, 0.1), (300_000, 0.0)]
    >>> temperature_schedule(50_000, schedule)  # Returns 1.0
    >>> temperature_schedule(150_000, schedule)  # Returns 0.5
    >>> temperature_schedule(350_000, schedule)  # Returns 0.0
    """
    temperature = schedule[0][1]
    for threshold, temp in schedule:
        if step >= threshold:
            temperature = temp
    return temperature
