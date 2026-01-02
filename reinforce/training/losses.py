"""
Loss functions for Stochastic MuZero training.

This module provides JAX-native loss functions that support:
- Policy cross-entropy loss
- Value MSE loss with scaling transform
- Reward MSE loss
- Chance distribution cross-entropy loss
- Commitment loss for VQ-VAE encoder

All losses support unrolling over K steps for training the dynamics models.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from reinforce.mcts.stochastic_mctx import NetworkApplyFns, NetworkParams
from reinforce.training.config import TrainConfig

# ##>: Type aliases.
Array = jax.Array


class LossOutput(NamedTuple):
    """Container for loss values and metrics."""

    total_loss: Array
    policy_loss: Array
    value_loss: Array
    reward_loss: Array
    chance_loss: Array
    commitment_loss: Array


class TrainingTargets(NamedTuple):
    """
    Training targets for a single sample.

    Attributes
    ----------
    observations : Array
        Observations at each step, shape (K+1, observation_dim).
    actions : Array
        Actions taken at each step, shape (K,).
    target_policies : Array
        Target policy distributions, shape (K+1, action_size).
    target_values : Array
        Target value estimates, shape (K+1,).
    target_rewards : Array
        Target rewards, shape (K,).
    """

    observations: Array
    actions: Array
    target_policies: Array
    target_values: Array
    target_rewards: Array


def scale_value(value: Array, epsilon: float = 0.001) -> Array:
    """
    Apply value scaling transform.

    h(x) = sign(x) * (sqrt(|x| + 1) - 1) + epsilon * x

    This transform compresses large values while preserving small values,
    improving training stability for games with large reward ranges.

    Parameters
    ----------
    value : Array
        Value to scale.
    epsilon : float
        Small constant for linear term.

    Returns
    -------
    Array
        Scaled value.
    """
    return jnp.sign(value) * (jnp.sqrt(jnp.abs(value) + 1) - 1) + epsilon * value


def inverse_scale_value(scaled_value: Array, epsilon: float = 0.001) -> Array:
    """
    Inverse of scale_value transform.

    h^{-1}(x) = sign(x) * ((sqrt(1 + 4*epsilon*(|x| + 1 + epsilon)) - 1) / (2*epsilon))^2 - 1)

    Parameters
    ----------
    scaled_value : Array
        Scaled value to invert.
    epsilon : float
        Same epsilon used in scale_value.

    Returns
    -------
    Array
        Original value.
    """
    # ##>: Simplified inverse for small epsilon.
    inside_sqrt = 1 + 4 * epsilon * (jnp.abs(scaled_value) + 1 + epsilon)
    result = jnp.sign(scaled_value) * (jnp.square((jnp.sqrt(inside_sqrt) - 1) / (2 * epsilon)) - 1)
    return result


@jax.jit
def policy_loss(predicted_logits: Array, target_policy: Array) -> Array:
    """
    Compute cross-entropy policy loss.

    Parameters
    ----------
    predicted_logits : Array
        Predicted policy logits, shape (..., action_size).
    target_policy : Array
        Target policy distribution, shape (..., action_size).

    Returns
    -------
    Array
        Cross-entropy loss (scalar).
    """
    # ##>: Compute log-softmax for numerical stability.
    log_probs = jax.nn.log_softmax(predicted_logits, axis=-1)
    return -jnp.sum(target_policy * log_probs, axis=-1)


@jax.jit
def value_loss(predicted_value: Array, target_value: Array, epsilon: float = 0.001) -> Array:
    """
    Compute MSE value loss with scaling.

    Parameters
    ----------
    predicted_value : Array
        Predicted value, shape (...,).
    target_value : Array
        Target value, shape (...,).
    epsilon : float
        Value scaling epsilon.

    Returns
    -------
    Array
        MSE loss (scalar).
    """
    # ##>: Scale targets for more stable training.
    scaled_target = scale_value(target_value, epsilon)
    return jnp.square(predicted_value - scaled_target)


@jax.jit
def reward_loss(predicted_reward: Array, target_reward: Array, epsilon: float = 0.001) -> Array:
    """
    Compute MSE reward loss with scaling.

    Parameters
    ----------
    predicted_reward : Array
        Predicted reward, shape (...,).
    target_reward : Array
        Target reward, shape (...,).
    epsilon : float
        Value scaling epsilon.

    Returns
    -------
    Array
        MSE loss (scalar).
    """
    scaled_target = scale_value(target_reward, epsilon)
    return jnp.square(predicted_reward - scaled_target)


@jax.jit
def chance_loss(predicted_logits: Array, target_code: Array) -> Array:
    """
    Compute cross-entropy loss for chance distribution.

    Parameters
    ----------
    predicted_logits : Array
        Predicted chance logits from afterstate prediction, shape (..., codebook_size).
    target_code : Array
        Target chance code (one-hot), shape (..., codebook_size).

    Returns
    -------
    Array
        Cross-entropy loss (scalar).
    """
    log_probs = jax.nn.log_softmax(predicted_logits, axis=-1)
    return -jnp.sum(target_code * log_probs, axis=-1)


@jax.jit
def commitment_loss(encoder_output: Array, target_code: Array) -> Array:
    """
    Compute VQ-VAE commitment loss.

    Encourages the encoder to commit to codebook entries.

    Parameters
    ----------
    encoder_output : Array
        Encoder output (soft or hard codes), shape (..., codebook_size).
    target_code : Array
        Target one-hot code, shape (..., codebook_size).

    Returns
    -------
    Array
        MSE commitment loss.
    """
    return jnp.sum(jnp.square(encoder_output - target_code), axis=-1)


def compute_loss(
    params: NetworkParams,
    apply_fns: NetworkApplyFns,
    batch: TrainingTargets,
    config: TrainConfig,
) -> tuple[Array, LossOutput]:
    """
    Compute total loss for a batch of training samples.

    This function:
    1. Encodes the initial observation to hidden state
    2. Unrolls the model K steps, computing losses at each step
    3. Aggregates all losses with configured weights

    Parameters
    ----------
    params : NetworkParams
        Network parameters.
    apply_fns : NetworkApplyFns
        Network apply functions.
    batch : TrainingTargets
        Batch of training targets.
    config : TrainConfig
        Training configuration.

    Returns
    -------
    tuple[Array, LossOutput]
        - Total loss (scalar for gradient computation)
        - LossOutput with breakdown of individual losses
    """

    def single_sample_loss(sample: TrainingTargets) -> LossOutput:
        """Compute loss for a single training sample."""
        observations = sample.observations
        actions = sample.actions
        target_policies = sample.target_policies
        target_values = sample.target_values
        target_rewards = sample.target_rewards

        # ##>: Step 0: Encode initial observation.
        hidden_state = apply_fns.representation(params.representation, observations[0])

        # ##>: Get initial prediction.
        policy_logits, value = apply_fns.prediction(params.prediction, hidden_state)

        # ##>: Initial losses.
        p_loss = policy_loss(policy_logits, target_policies[0])
        v_loss = value_loss(value, target_values[0], config.value_epsilon)
        r_loss = jnp.array(0.0)
        c_loss = jnp.array(0.0)
        commit_loss = jnp.array(0.0)

        # ##>: Unroll K steps.
        def unroll_step(carry, step_idx):
            state, total_p, total_v, total_r, total_c, total_commit = carry

            # ##>: Get action one-hot.
            action_onehot = jax.nn.one_hot(actions[step_idx], config.action_size)

            # ##>: Afterstate dynamics: state + action -> afterstate.
            afterstate = apply_fns.afterstate_dynamics(params.afterstate_dynamics, state, action_onehot)

            # ##>: Afterstate prediction: afterstate -> (Q-value, chance_logits).
            q_value, chance_logits = apply_fns.afterstate_prediction(params.afterstate_prediction, afterstate)

            # ##>: Get target chance code from next observation.
            # ##>: Note: This is a simplification - full implementation would use
            # ##>: the encoder model to encode observations[step_idx + 1].
            # ##>: Here we use a placeholder target for the chance.
            target_chance = jnp.zeros(config.codebook_size)
            target_chance = target_chance.at[0].set(1.0)  # Placeholder

            # ##>: Chance loss.
            step_c_loss = chance_loss(chance_logits, target_chance)

            # ##>: Sample chance code (use argmax for deterministic training).
            sampled_chance = jax.nn.one_hot(jnp.argmax(chance_logits), config.codebook_size)

            # ##>: Dynamics: afterstate + chance -> (next_state, reward).
            next_state, pred_reward = apply_fns.dynamics(params.dynamics, afterstate, sampled_chance)

            # ##>: Prediction from next state.
            next_policy_logits, next_value = apply_fns.prediction(params.prediction, next_state)

            # ##>: Compute step losses.
            step_p_loss = policy_loss(next_policy_logits, target_policies[step_idx + 1])
            step_v_loss = value_loss(next_value, target_values[step_idx + 1], config.value_epsilon)
            step_r_loss = reward_loss(pred_reward, target_rewards[step_idx], config.value_epsilon)

            # ##>: Accumulate losses.
            new_carry = (
                next_state,
                total_p + step_p_loss,
                total_v + step_v_loss,
                total_r + step_r_loss,
                total_c + step_c_loss,
                total_commit,
            )

            return new_carry, None

        # ##>: Run unrolling.
        num_unroll = config.num_unroll_steps
        initial_carry = (hidden_state, p_loss, v_loss, r_loss, c_loss, commit_loss)
        (_, total_p, total_v, total_r, total_c, total_commit), _ = lax.scan(
            unroll_step, initial_carry, jnp.arange(num_unroll)
        )

        # ##>: Average losses over unroll steps.
        avg_p = total_p / (num_unroll + 1)
        avg_v = total_v / (num_unroll + 1)
        avg_r = total_r / num_unroll
        avg_c = total_c / num_unroll

        return LossOutput(
            total_loss=jnp.array(0.0),  # Computed below
            policy_loss=avg_p,
            value_loss=avg_v,
            reward_loss=avg_r,
            chance_loss=avg_c,
            commitment_loss=total_commit,
        )

    # ##>: Vectorize over batch.
    batch_losses = jax.vmap(single_sample_loss)(batch)

    # ##>: Average over batch.
    mean_p = jnp.mean(batch_losses.policy_loss)
    mean_v = jnp.mean(batch_losses.value_loss)
    mean_r = jnp.mean(batch_losses.reward_loss)
    mean_c = jnp.mean(batch_losses.chance_loss)
    mean_commit = jnp.mean(batch_losses.commitment_loss)

    # ##>: Weighted total loss.
    total = (
        config.policy_loss_weight * mean_p
        + config.value_loss_weight * mean_v
        + config.reward_loss_weight * mean_r
        + config.chance_loss_weight * mean_c
        + config.commitment_loss_weight * mean_commit
    )

    output = LossOutput(
        total_loss=total,
        policy_loss=mean_p,
        value_loss=mean_v,
        reward_loss=mean_r,
        chance_loss=mean_c,
        commitment_loss=mean_commit,
    )

    return total, output
