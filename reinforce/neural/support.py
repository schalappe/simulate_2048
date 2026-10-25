"""
Categorical value representation for Stochastic MuZero (paper, Appendix C).

Value, reward, and afterstate value heads output logits over 601 bins that cover h(x) in [0, 600],
where h is the invertible transform from MuZero. Training targets go through `scalar_to_two_hot`;
search and logging read heads through `logits_to_scalar`. Nothing else should touch h-space.
"""

import jax
import jax.numpy as jnp

# ##>: Type alias.
Array = jax.Array

# ##>: Paper: 601 bins, values in [0, 600] after h. 2048 rewards and returns are never negative.
SUPPORT_SIZE = 601
VALUE_EPSILON = 0.001


def scale_value(value: Array, epsilon: float = VALUE_EPSILON) -> Array:
    """
    Apply the MuZero transform h(x) = sign(x) * (sqrt(|x| + 1) - 1) + epsilon * x.

    Parameters
    ----------
    value : Array
        Raw values.
    epsilon : float
        Weight of the linear term.

    Returns
    -------
    Array
        Scaled values.
    """
    return jnp.sign(value) * (jnp.sqrt(jnp.abs(value) + 1) - 1) + epsilon * value


def inverse_scale_value(scaled_value: Array, epsilon: float = VALUE_EPSILON) -> Array:
    """
    Invert `scale_value`.

    Parameters
    ----------
    scaled_value : Array
        Values in h-space.
    epsilon : float
        Same epsilon as used in `scale_value`.

    Returns
    -------
    Array
        Raw values.
    """
    inside_sqrt = 1 + 4 * epsilon * (jnp.abs(scaled_value) + 1 + epsilon)
    return jnp.sign(scaled_value) * (jnp.square((jnp.sqrt(inside_sqrt) - 1) / (2 * epsilon)) - 1)


def scalar_to_two_hot(value: Array, epsilon: float = VALUE_EPSILON) -> Array:
    """
    Convert raw scalar targets to two-hot distributions over the h-space bins.

    Parameters
    ----------
    value : Array
        Raw targets, shape (...,).
    epsilon : float
        Value transform epsilon.

    Returns
    -------
    Array
        Target distributions, shape (..., 601). Mass is split between the two nearest bins.
    """
    scaled = jnp.clip(scale_value(value, epsilon), 0.0, SUPPORT_SIZE - 1)
    lower = jnp.floor(scaled)
    upper_weight = scaled - lower
    lower_index = lower.astype(jnp.int32)
    upper_index = jnp.minimum(lower_index + 1, SUPPORT_SIZE - 1)
    lower_part = jax.nn.one_hot(lower_index, SUPPORT_SIZE) * (1.0 - upper_weight)[..., None]
    upper_part = jax.nn.one_hot(upper_index, SUPPORT_SIZE) * upper_weight[..., None]
    return lower_part + upper_part


def logits_to_scalar(logits: Array, epsilon: float = VALUE_EPSILON) -> Array:
    """
    Read a categorical head as a raw scalar: expected bin under the softmax, then h⁻¹.

    Parameters
    ----------
    logits : Array
        Head logits, shape (..., 601).
    epsilon : float
        Value transform epsilon.

    Returns
    -------
    Array
        Raw values, shape (...,).
    """
    probabilities = jax.nn.softmax(logits, axis=-1)
    expected_bin = jnp.sum(probabilities * jnp.arange(SUPPORT_SIZE, dtype=jnp.float32), axis=-1)
    return inverse_scale_value(expected_bin, epsilon)
