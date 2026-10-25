"""
Bridge between the Stochastic MuZero networks and `mctx.stochastic_muzero_policy`.

The networks predict categorical values in h-space; this module converts every value, reward, and
afterstate value to a raw scalar before `mctx` sees it, because the search adds rewards and discounted
values and that sum is only meaningful on raw returns.
"""

from typing import Any

import jax
import jax.numpy as jnp
import mctx

from reinforce.neural.network import NetworkApplyFns, NetworkParams
from reinforce.neural.support import logits_to_scalar

# ##>: Type aliases.
Array = jax.Array
PRNGKey = jax.Array


def run_mcts(
    params: NetworkParams,
    apply_fns: NetworkApplyFns,
    observations: Array,
    legal_mask: Array,
    key: PRNGKey,
    codebook_size: int,
    num_simulations: int,
    discount: float,
    dirichlet_alpha: float,
    dirichlet_fraction: float,
    pb_c_init: float,
    pb_c_base: float,
    max_depth: int | None = None,
) -> mctx.PolicyOutput:
    """
    Run Stochastic MuZero search on a batch of observations.

    Parameters
    ----------
    params : NetworkParams
        Network parameters.
    apply_fns : NetworkApplyFns
        Network apply functions.
    observations : Array
        Encoded observations, shape (batch, 496).
    legal_mask : Array
        True for legal actions, shape (batch, num_actions). Illegal actions are never searched or chosen.
    key : PRNGKey
        Random key for root noise and chance sampling.
    codebook_size : int
        Number of chance codes.
    num_simulations : int
        Simulations per search.
    discount : float
        Reward discount used inside the search.
    dirichlet_alpha : float
        Dirichlet noise alpha at the root.
    dirichlet_fraction : float
        Fraction of Dirichlet noise at the root; 0 for evaluation.
    pb_c_init : float
        PUCT exploration constant.
    pb_c_base : float
        PUCT exploration base.
    max_depth : int | None
        Maximum search depth; None lets mctx go as deep as `num_simulations`.

    Returns
    -------
    mctx.PolicyOutput
        Search output. `action_weights` are normalized root visit counts over legal actions.
    """
    num_actions = legal_mask.shape[-1]

    hidden_state = apply_fns.representation(params.representation, observations)
    policy_logits, value_logits = apply_fns.prediction(params.prediction, hidden_state)
    root = mctx.RootFnOutput(
        prior_logits=policy_logits,
        value=logits_to_scalar(value_logits),
        embedding=hidden_state,
    )

    def decision_recurrent_fn(
        model_params: Any, rng_key: PRNGKey, action: Array, state: Array
    ) -> tuple[mctx.DecisionRecurrentFnOutput, Array]:
        del rng_key
        action_onehot = jax.nn.one_hot(action, num_actions)
        afterstate = apply_fns.afterstate_dynamics(model_params.afterstate_dynamics, state, action_onehot)
        q_logits, chance_logits = apply_fns.afterstate_prediction(model_params.afterstate_prediction, afterstate)
        output = mctx.DecisionRecurrentFnOutput(
            chance_logits=chance_logits,
            afterstate_value=logits_to_scalar(q_logits),
        )
        return output, afterstate

    def chance_recurrent_fn(
        model_params: Any, rng_key: PRNGKey, chance_outcome: Array, afterstate: Array
    ) -> tuple[mctx.ChanceRecurrentFnOutput, Array]:
        del rng_key
        chance_onehot = jax.nn.one_hot(chance_outcome, codebook_size)
        next_state, reward_logits = apply_fns.dynamics(model_params.dynamics, afterstate, chance_onehot)
        next_policy_logits, next_value_logits = apply_fns.prediction(model_params.prediction, next_state)
        value = logits_to_scalar(next_value_logits)
        output = mctx.ChanceRecurrentFnOutput(
            action_logits=next_policy_logits,
            value=value,
            reward=logits_to_scalar(reward_logits),
            discount=jnp.full_like(value, discount),
        )
        return output, next_state

    return mctx.stochastic_muzero_policy(
        params=params,  # pyrefly: ignore[bad-argument-type]
        rng_key=key,
        root=root,
        decision_recurrent_fn=decision_recurrent_fn,  # pyrefly: ignore[bad-argument-type]
        chance_recurrent_fn=chance_recurrent_fn,  # pyrefly: ignore[bad-argument-type]
        num_simulations=num_simulations,
        invalid_actions=~legal_mask,
        max_depth=max_depth,
        dirichlet_fraction=dirichlet_fraction,
        dirichlet_alpha=dirichlet_alpha,
        pb_c_init=pb_c_init,
        pb_c_base=pb_c_base,
    )
