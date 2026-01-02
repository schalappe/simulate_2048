"""
Integration with DeepMind's mctx library for Stochastic MuZero MCTS.

This module provides the bridge between our Stochastic MuZero networks
and mctx's stochastic_muzero_policy function.

The key abstraction is the RecurrentFn, which tells mctx how to:
1. Compute the initial representation from an observation
2. Transition from state to afterstate (afterstate dynamics)
3. Predict chance outcomes from afterstates (afterstate prediction)
4. Transition from afterstate to next state (stochastic dynamics)
5. Predict policy and value from states (prediction)
"""

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mctx

# ##>: Type aliases.
Array = jax.Array
PRNGKey = jax.Array
Params = Any


class NetworkApplyFns(NamedTuple):
    """
    Container for network apply functions.

    Each function takes (params, input) and returns the model output.
    """

    representation: Any  # (params, observation) -> hidden_state
    prediction: Any  # (params, state) -> (policy_logits, value)
    afterstate_dynamics: Any  # (params, state, action_onehot) -> afterstate
    afterstate_prediction: Any  # (params, afterstate) -> (q_value, chance_logits)
    dynamics: Any  # (params, afterstate, chance_onehot) -> (next_state, reward)


class NetworkParams(NamedTuple):
    """
    Container for all network parameters.
    """

    representation: Any
    prediction: Any
    afterstate_dynamics: Any
    afterstate_prediction: Any
    dynamics: Any


class StochasticRecurrentFn:
    """
    Wrapper that creates mctx-compatible recurrent functions for Stochastic MuZero.

    This class generates the functions that mctx needs to perform tree search:
    - decision_recurrent_fn: For expanding decision nodes (state -> action -> afterstate)
    - chance_recurrent_fn: For expanding chance nodes (afterstate -> chance -> next_state)

    Attributes
    ----------
    params : NetworkParams
        Network parameters for all models.
    apply_fns : NetworkApplyFns
        Functions to apply each network.
    num_actions : int
        Number of possible actions (4 for 2048).
    codebook_size : int
        Number of possible chance outcomes.
    discount : float
        Reward discount factor.
    """

    def __init__(
        self,
        params: NetworkParams,
        apply_fns: NetworkApplyFns,
        num_actions: int = 4,
        codebook_size: int = 32,
        discount: float = 0.999,
    ):
        """Initialize the recurrent function wrapper."""
        self.params = params
        self.apply_fns = apply_fns
        self.num_actions = num_actions
        self.codebook_size = codebook_size
        self.discount = discount

    def root_fn(self, observation: Array) -> mctx.RootFnOutput:
        """
        Compute the root state from an observation.

        Parameters
        ----------
        observation : Array
            Encoded observation, shape (..., observation_dim).

        Returns
        -------
        mctx.RootFnOutput
            Initial state with prior policy and value.
        """
        # ##>: Get hidden state representation.
        hidden_state = self.apply_fns.representation(self.params.representation, observation)

        # ##>: Get policy and value from prediction network.
        policy_logits, value = self.apply_fns.prediction(self.params.prediction, hidden_state)

        return mctx.RootFnOutput(
            prior_logits=policy_logits,
            value=value,
            embedding=hidden_state,
        )

    def decision_recurrent_fn(
        self,
        params: Any,
        rng_key: PRNGKey,
        action: Array,
        state: Array,
    ) -> tuple[mctx.DecisionRecurrentFnOutput, Array]:
        """
        Transition from decision state to afterstate.

        This is called when expanding a decision node: given a state and action,
        compute the afterstate and its chance distribution.

        Parameters
        ----------
        params : Any
            Unused (params are stored in self).
        rng_key : PRNGKey
            Random key (unused for deterministic transition).
        action : Array
            Action index, shape (...,).
        state : Array
            Current hidden state, shape (..., hidden_dim).

        Returns
        -------
        tuple[mctx.DecisionRecurrentFnOutput, Array]
            - DecisionRecurrentFnOutput with chance logits and afterstate value
            - Afterstate embedding
        """
        del params, rng_key  # Unused

        # ##>: Convert action index to one-hot.
        action_onehot = jax.nn.one_hot(action, self.num_actions)

        # ##>: Compute afterstate from state and action.
        afterstate = self.apply_fns.afterstate_dynamics(self.params.afterstate_dynamics, state, action_onehot)

        # ##>: Get Q-value and chance distribution from afterstate.
        q_value, chance_logits = self.apply_fns.afterstate_prediction(self.params.afterstate_prediction, afterstate)

        output = mctx.DecisionRecurrentFnOutput(
            chance_logits=chance_logits,
            afterstate_value=q_value,
        )

        return output, afterstate

    def chance_recurrent_fn(
        self,
        params: Any,
        rng_key: PRNGKey,
        chance_outcome: Array,
        afterstate: Array,
    ) -> tuple[mctx.ChanceRecurrentFnOutput, Array]:
        """
        Transition from afterstate to next state via chance outcome.

        This is called when expanding a chance node: given an afterstate and
        chance outcome, compute the next state.

        Parameters
        ----------
        params : Any
            Unused (params are stored in self).
        rng_key : PRNGKey
            Random key (unused for deterministic transition).
        chance_outcome : Array
            Chance outcome index, shape (...,).
        afterstate : Array
            Afterstate embedding, shape (..., hidden_dim).

        Returns
        -------
        tuple[mctx.ChanceRecurrentFnOutput, Array]
            - ChanceRecurrentFnOutput with action logits, value, reward, discount
            - Next state embedding
        """
        del params, rng_key  # Unused

        # ##>: Convert chance outcome index to one-hot.
        chance_onehot = jax.nn.one_hot(chance_outcome, self.codebook_size)

        # ##>: Compute next state and reward from afterstate and chance.
        next_state, reward = self.apply_fns.dynamics(self.params.dynamics, afterstate, chance_onehot)

        # ##>: Get policy and value from the next state.
        policy_logits, value = self.apply_fns.prediction(self.params.prediction, next_state)

        output = mctx.ChanceRecurrentFnOutput(
            action_logits=policy_logits,
            value=value,
            reward=reward,
            discount=jnp.ones_like(value) * self.discount,
        )

        return output, next_state


def run_mcts(
    observation: Array,
    params: NetworkParams,
    apply_fns: NetworkApplyFns,
    key: PRNGKey,
    num_simulations: int = 100,
    num_actions: int = 4,
    codebook_size: int = 32,
    discount: float = 0.999,
    max_depth: int | None = None,
    dirichlet_alpha: float = 0.25,
    dirichlet_fraction: float = 0.1,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
) -> mctx.PolicyOutput:
    """
    Run Stochastic MuZero MCTS on an observation.

    Parameters
    ----------
    observation : Array
        Encoded observation, shape (observation_dim,) or (batch, observation_dim).
    params : NetworkParams
        Network parameters for all models.
    apply_fns : NetworkApplyFns
        Functions to apply each network.
    key : PRNGKey
        JAX random key for exploration.
    num_simulations : int
        Number of MCTS simulations to run.
    num_actions : int
        Number of possible actions.
    codebook_size : int
        Number of possible chance outcomes.
    discount : float
        Reward discount factor.
    max_depth : int | None
        Maximum search depth (None for unlimited).
    dirichlet_alpha : float
        Dirichlet noise alpha for root exploration.
    dirichlet_fraction : float
        Fraction of Dirichlet noise to add to root.
    pb_c_init : float
        PUCT exploration constant.
    pb_c_base : float
        PUCT exploration base.

    Returns
    -------
    mctx.PolicyOutput
        Search results including action weights and value.
    """
    # ##>: Create recurrent function wrapper.
    recurrent_fn = StochasticRecurrentFn(
        params=params,
        apply_fns=apply_fns,
        num_actions=num_actions,
        codebook_size=codebook_size,
        discount=discount,
    )

    # ##>: Compute root state.
    root = recurrent_fn.root_fn(observation)

    # ##>: Add batch dimension if needed.
    is_batched = observation.ndim > 1
    if not is_batched:
        root = jax.tree.map(lambda x: x[None], root)

    # ##>: Run stochastic MuZero policy.
    policy_output = mctx.stochastic_muzero_policy(
        params=None,  # Unused, params are in recurrent_fn
        rng_key=key,
        root=root,
        decision_recurrent_fn=recurrent_fn.decision_recurrent_fn,
        chance_recurrent_fn=recurrent_fn.chance_recurrent_fn,
        num_simulations=num_simulations,
        max_depth=max_depth,
        dirichlet_fraction=dirichlet_fraction,
        dirichlet_alpha=dirichlet_alpha,
        pb_c_init=pb_c_init,
        pb_c_base=pb_c_base,
    )

    # ##>: Remove batch dimension if we added it.
    if not is_batched:
        policy_output = jax.tree.map(lambda x: x[0], policy_output)

    return policy_output


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def run_mcts_jit(
    observation: Array,
    params: NetworkParams,
    apply_fns: NetworkApplyFns,
    key: PRNGKey,
    num_simulations: int = 100,
    num_actions: int = 4,
    codebook_size: int = 32,
    discount: float = 0.999,
    max_depth: int | None = None,
) -> mctx.PolicyOutput:
    """
    JIT-compiled version of run_mcts.

    Same as run_mcts but with static arguments for JIT compilation.
    Use this for maximum performance when hyperparameters are fixed.
    """
    return run_mcts(
        observation=observation,
        params=params,
        apply_fns=apply_fns,
        key=key,
        num_simulations=num_simulations,
        num_actions=num_actions,
        codebook_size=codebook_size,
        discount=discount,
        max_depth=max_depth,
    )


def batched_run_mcts(
    observations: Array,
    params: NetworkParams,
    apply_fns: NetworkApplyFns,
    keys: Array,
    num_simulations: int = 100,
    num_actions: int = 4,
    codebook_size: int = 32,
    discount: float = 0.999,
) -> mctx.PolicyOutput:
    """
    Run MCTS on a batch of observations in parallel.

    Parameters
    ----------
    observations : Array
        Batch of observations, shape (batch_size, observation_dim).
    params : NetworkParams
        Network parameters (shared across batch).
    apply_fns : NetworkApplyFns
        Network apply functions.
    keys : Array
        Batch of random keys, shape (batch_size, 2).
    num_simulations : int
        Number of MCTS simulations.
    num_actions : int
        Number of possible actions.
    codebook_size : int
        Number of possible chance outcomes.
    discount : float
        Reward discount factor.

    Returns
    -------
    mctx.PolicyOutput
        Batched policy outputs.
    """
    # ##>: mctx.stochastic_muzero_policy already handles batched inputs.
    return run_mcts(
        observation=observations,
        params=params,
        apply_fns=apply_fns,
        key=keys[0],  # mctx handles batch internally with single key
        num_simulations=num_simulations,
        num_actions=num_actions,
        codebook_size=codebook_size,
        discount=discount,
    )
