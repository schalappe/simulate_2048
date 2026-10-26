"""
Action selection from MCTS visit distributions.

Temperature is a traced value, so changing it along the schedule never recompiles self-play.
"""

import jax
import jax.numpy as jnp

# ##>: Type aliases.
Array = jax.Array
PRNGKey = jax.Array

# ##>: Temperatures at or below this select the most visited action.
GREEDY_TEMPERATURE = 1e-3


def select_action(key: PRNGKey, action_weights: Array, temperature: Array | float) -> Array:
    """
    Select actions from root visit distributions.

    Parameters
    ----------
    key : PRNGKey
        Random key, shared by the batch.
    action_weights : Array
        Normalized root visit counts, shape (batch, num_actions). Illegal actions have weight 0.
    temperature : Array | float
        0 selects the most visited action; 1 samples proportionally to visits.

    Returns
    -------
    Array
        Selected actions, shape (batch,). Never an action with zero weight.
    """
    greedy = jnp.argmax(action_weights, axis=-1)
    # ##&: Both branches always run, so guard the division and keep zero-weight actions at -inf.
    safe_temperature = jnp.maximum(temperature, GREEDY_TEMPERATURE)
    logits = jnp.where(action_weights > 0, jnp.log(jnp.maximum(action_weights, 1e-30)) / safe_temperature, -jnp.inf)
    sampled = jax.random.categorical(key, logits, axis=-1)
    return jnp.where(temperature <= GREEDY_TEMPERATURE, greedy, sampled)
