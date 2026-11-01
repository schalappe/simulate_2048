"""
Loss functions for Stochastic MuZero training (paper, eq. 6-7 and Appendix A).

For each sampled position the model is unrolled K steps along the real actions:
- MuZero loss: policy cross-entropy, categorical value and reward losses.
- Chance loss: categorical afterstate value loss toward the value target, cross-entropy of the predicted
  chance distribution toward the encoder's code, and the VQ-VAE commitment cost.

The encoder reads the observations before and after each transition. Its straight-through code feeds the
dynamics model, so the encoder learns from the reward and value losses downstream of it.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from reinforce.game.core import encode_exponents
from reinforce.neural.network import NetworkApplyFns, NetworkParams
from reinforce.neural.support import scalar_to_two_hot
from reinforce.training.config import TrainConfig

# ##>: Type aliases. Batches are built as NumPy arrays on the host and read as JAX arrays under jit.
Array = jax.Array
BatchArray = jax.Array | np.ndarray

# ##>: MuZero halves the gradient flowing back into each recurrent step from the steps after it.
DYNAMICS_GRADIENT_SCALE = 0.5


class TrainingBatch(NamedTuple):
    """
    A batch of unroll windows sampled from the replay buffer.

    Shapes are given for one sample; a batch adds a leading batch dimension. K is `num_unroll_steps`.

    Attributes
    ----------
    boards : Array
        Tile exponents of the K + 1 states, shape (K + 1, 16), uint8.
    actions : Array
        Actions of the K transitions, shape (K,).
    rewards : Array
        Reward targets of the K transitions, shape (K,).
    reward_mask : Array
        1 where the reward target is known (real or after a terminal state), shape (K,).
    transition_mask : Array
        1 for real transitions, where the encoder sees two real observations, shape (K,).
    policies : Array
        Policy targets, shape (K + 1, num_actions).
    policy_mask : Array
        1 for real states, shape (K + 1,).
    values : Array
        Raw value targets, shape (K + 1,).
    value_mask : Array
        1 where the value target is known, shape (K + 1,).
    weights : Array
        Normalized importance-sampling weight, shape ().
    """

    boards: BatchArray
    actions: BatchArray
    rewards: BatchArray
    reward_mask: BatchArray
    transition_mask: BatchArray
    policies: BatchArray
    policy_mask: BatchArray
    values: BatchArray
    value_mask: BatchArray
    weights: BatchArray


class LossOutput(NamedTuple):
    """Batch-mean losses, before loss weights, plus the weighted total."""

    total_loss: Array
    policy_loss: Array
    value_loss: Array
    reward_loss: Array
    afterstate_value_loss: Array
    chance_loss: Array
    commitment_loss: Array


def cross_entropy(logits: Array, target: Array) -> Array:
    """
    Cross-entropy between a target distribution and logits.

    Parameters
    ----------
    logits : Array
        Logits, shape (..., n).
    target : Array
        Target distribution, shape (..., n).

    Returns
    -------
    Array
        Loss, shape (...,).
    """
    return -jnp.sum(target * jax.nn.log_softmax(logits, axis=-1), axis=-1)


def straight_through_code(encoder_logits: Array) -> tuple[Array, Array, Array]:
    """
    Turn encoder logits into a one-hot chance code with a straight-through gradient (paper, section 4.1).

    Parameters
    ----------
    encoder_logits : Array
        Encoder logits, shape (..., codebook_size).

    Returns
    -------
    tuple[Array, Array, Array]
        - code: one-hot forward, softmax gradient backward
        - one_hot: the code without gradient, the commitment target
        - probabilities: encoder softmax output
    """
    probabilities = jax.nn.softmax(encoder_logits, axis=-1)
    one_hot = jax.nn.one_hot(jnp.argmax(encoder_logits, axis=-1), encoder_logits.shape[-1])
    code = one_hot + probabilities - lax.stop_gradient(probabilities)
    return code, one_hot, probabilities


def scale_gradient(x: Array, scale: float) -> Array:
    """Keep the forward value and multiply the gradient by `scale`."""
    return x * scale + lax.stop_gradient(x) * (1 - scale)


def sample_losses(
    params: NetworkParams,
    apply_fns: NetworkApplyFns,
    sample: TrainingBatch,
    config: TrainConfig,
) -> LossOutput:
    """
    Unroll the model on one sample and compute its unweighted losses.

    The first state's policy and value losses count fully; each unrolled step counts 1/K, as in MuZero.

    Parameters
    ----------
    params : NetworkParams
        Network parameters.
    apply_fns : NetworkApplyFns
        Network apply functions.
    sample : TrainingBatch
        One sample (no batch dimension).
    config : TrainConfig
        Training configuration.

    Returns
    -------
    LossOutput
        Per-sample losses; `total_loss` is left at 0.
    """
    observations = encode_exponents(sample.boards)
    step_scale = 1.0 / config.num_unroll_steps

    hidden_state = apply_fns.representation(params.representation, observations[0])
    policy_logits, value_logits = apply_fns.prediction(params.prediction, hidden_state)
    initial = (
        cross_entropy(policy_logits, sample.policies[0]) * sample.policy_mask[0],
        cross_entropy(value_logits, scalar_to_two_hot(sample.values[0])) * sample.value_mask[0],
    )

    def unroll_step(state: Array, k: Array) -> tuple[Array, tuple[Array, ...]]:
        action_onehot = jax.nn.one_hot(sample.actions[k], config.action_size)
        afterstate = apply_fns.afterstate_dynamics(params.afterstate_dynamics, state, action_onehot)
        q_logits, chance_logits = apply_fns.afterstate_prediction(params.afterstate_prediction, afterstate)

        observation_pair = jnp.concatenate([observations[k], observations[k + 1]])
        encoder_logits = apply_fns.encoder(params.encoder, observation_pair)
        code, one_hot, probabilities = straight_through_code(encoder_logits)

        next_state, reward_logits = apply_fns.dynamics(params.dynamics, afterstate, code)
        next_policy_logits, next_value_logits = apply_fns.prediction(params.prediction, next_state)

        transition = sample.transition_mask[k]
        losses = (
            cross_entropy(next_policy_logits, sample.policies[k + 1]) * sample.policy_mask[k + 1],
            cross_entropy(next_value_logits, scalar_to_two_hot(sample.values[k + 1])) * sample.value_mask[k + 1],
            cross_entropy(reward_logits, scalar_to_two_hot(sample.rewards[k])) * sample.reward_mask[k],
            cross_entropy(q_logits, scalar_to_two_hot(sample.values[k])) * sample.value_mask[k],
            # ##>: The target keeps its straight-through gradient, as in the paper's pseudocode (Appendix H).
            cross_entropy(chance_logits, code) * transition,
            jnp.sum(jnp.square(one_hot - probabilities)) * transition,
        )
        return scale_gradient(next_state, DYNAMICS_GRADIENT_SCALE), losses

    _, step_losses = lax.scan(unroll_step, hidden_state, jnp.arange(config.num_unroll_steps))
    policy, value, reward, afterstate_value, chance, commitment = (jnp.sum(loss) * step_scale for loss in step_losses)

    return LossOutput(
        total_loss=jnp.array(0.0),
        policy_loss=initial[0] + policy,
        value_loss=initial[1] + value,
        reward_loss=reward,
        afterstate_value_loss=afterstate_value,
        chance_loss=chance,
        commitment_loss=commitment,
    )


def compute_loss(
    params: NetworkParams,
    apply_fns: NetworkApplyFns,
    batch: TrainingBatch,
    config: TrainConfig,
) -> tuple[Array, LossOutput]:
    """
    Compute the importance-weighted total loss of a batch.

    Parameters
    ----------
    params : NetworkParams
        Network parameters.
    apply_fns : NetworkApplyFns
        Network apply functions.
    batch : TrainingBatch
        Batch of samples.
    config : TrainConfig
        Training configuration.

    Returns
    -------
    tuple[Array, LossOutput]
        - Total loss (scalar for gradient computation)
        - Weighted batch-mean of each loss
    """
    per_sample = jax.vmap(lambda sample: sample_losses(params, apply_fns, sample, config))(batch)

    def weighted_mean(loss: Array) -> Array:
        return jnp.mean(batch.weights * loss)

    output = LossOutput(*(weighted_mean(loss) for loss in per_sample))
    total = (
        config.policy_loss_weight * output.policy_loss
        + config.value_loss_weight * (output.value_loss + output.afterstate_value_loss)
        + config.reward_loss_weight * output.reward_loss
        + config.chance_loss_weight * output.chance_loss
        + config.commitment_loss_weight * output.commitment_loss
    )
    return total, output._replace(total_loss=total)
