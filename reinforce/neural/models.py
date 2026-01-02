# pyrefly: ignore-errors
# ##>: pyrefly doesn't understand Flax Linen module API correctly.
# ##>: The nn.Dense(features) syntax is correct but pyrefly misinterprets it.
"""
Flax neural network models for Stochastic MuZero.

This module provides Flax Linen implementations of the six Stochastic MuZero models:
- Representation (h): Converts observations to hidden states
- Prediction (f): Outputs policy and value from hidden state
- Afterstate Dynamics (φ): Predicts afterstate given state and action
- Afterstate Prediction (ψ): Outputs Q-value and chance distribution from afterstate
- Dynamics (g): Predicts next state and reward given afterstate and chance code
- Encoder (e): Encodes observation to chance code (VQ-VAE style)

Architecture follows the paper exactly:
- 10 ResNet v2 style pre-activation residual blocks
- 256 hidden units per layer
- Layer Normalization + ReLU activations

Reference: "Planning in Stochastic Environments with a Learned Model" (ICLR 2022)
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

# ##>: Type alias.
Array = jax.Array

# ##>: Paper-specified architecture constants.
NUM_RESIDUAL_BLOCKS = 10
HIDDEN_UNITS = 256


class IdentityBlockDense(nn.Module):
    """
    ResNet v2 style pre-activation identity block with dense layers.

    Architecture: LayerNorm -> ReLU -> Dense -> LayerNorm -> ReLU -> Dense -> Add

    This block preserves the input dimension and adds a residual connection, enabling deeper
    networks without vanishing gradients.

    Attributes
    ----------
    features : int
        Number of features/units in the dense layers.
    """

    features: int = HIDDEN_UNITS

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Apply the identity block.

        Parameters
        ----------
        x : Array
            Input tensor of shape (..., features).

        Returns
        -------
        Array
            Output tensor of same shape as input.
        """
        residual = x

        # ##>: Pre-activation: normalize and activate before transformation.
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)

        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)

        # ##>: Residual connection.
        return x + residual


class ResidualStack(nn.Module):
    """
    Stack of residual identity blocks.

    Attributes
    ----------
    num_blocks : int
        Number of identity blocks to stack.
    features : int
        Number of features in each block.
    """

    num_blocks: int = NUM_RESIDUAL_BLOCKS
    features: int = HIDDEN_UNITS

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Apply the residual stack."""
        for _ in range(self.num_blocks):
            x = IdentityBlockDense(self.features)(x)
        return x


class Representation(nn.Module):
    """
    Representation model (h): observation -> hidden state.

    Converts raw observations into latent hidden state representations
    that capture the relevant features for decision making.

    Attributes
    ----------
    hidden_size : int
        Size of the hidden state output.
    num_blocks : int
        Number of residual blocks.
    """

    hidden_size: int = HIDDEN_UNITS
    num_blocks: int = NUM_RESIDUAL_BLOCKS

    @nn.compact
    def __call__(self, observation: Array) -> Array:
        """
        Encode observation to hidden state.

        Parameters
        ----------
        observation : Array
            Flattened observation, shape (..., observation_dim).

        Returns
        -------
        Array
            Hidden state, shape (..., hidden_size).
        """
        # ##>: Initial projection to hidden size.
        x = nn.Dense(self.hidden_size)(observation)

        # ##>: Stack of residual blocks.
        x = ResidualStack(self.num_blocks, self.hidden_size)(x)

        # ##>: Final normalization and output projection.
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size, name='hidden_state')(x)

        return x


class Prediction(nn.Module):
    """
    Prediction model (f): hidden state -> (policy, value).

    Outputs the action probability distribution and state value estimate.

    Attributes
    ----------
    action_size : int
        Number of possible actions.
    hidden_size : int
        Size of the hidden layers.
    num_blocks : int
        Number of residual blocks.
    """

    action_size: int = 4
    hidden_size: int = HIDDEN_UNITS
    num_blocks: int = NUM_RESIDUAL_BLOCKS

    @nn.compact
    def __call__(self, state: Array) -> tuple[Array, Array]:
        """
        Predict policy and value from hidden state.

        Parameters
        ----------
        state : Array
            Hidden state, shape (..., hidden_size).

        Returns
        -------
        tuple[Array, Array]
            - policy_logits: Logits for each action, shape (..., action_size)
            - value: Value estimate, shape (..., 1)
        """
        # ##>: Initial projection.
        x = nn.Dense(self.hidden_size)(state)

        # ##>: Stack of residual blocks.
        x = ResidualStack(self.num_blocks, self.hidden_size)(x)

        # ##>: Final normalization.
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # ##>: Policy head (logits, not probabilities).
        policy_logits = nn.Dense(self.action_size, name='policy_logits')(x)

        # ##>: Value head.
        value = nn.Dense(1, name='value')(x)
        value = jnp.squeeze(value, axis=-1)

        return policy_logits, value


class AfterstateDynamics(nn.Module):
    """
    Afterstate Dynamics model (φ): (state, action) -> afterstate.

    Predicts the afterstate resulting from taking an action.
    The afterstate represents the state after the action but before
    the stochastic environment response (tile spawn in 2048).

    Attributes
    ----------
    hidden_size : int
        Size of the hidden layers and afterstate output.
    action_size : int
        Number of possible actions.
    num_blocks : int
        Number of residual blocks.
    """

    hidden_size: int = HIDDEN_UNITS
    action_size: int = 4
    num_blocks: int = NUM_RESIDUAL_BLOCKS

    @nn.compact
    def __call__(self, state: Array, action: Array) -> Array:
        """
        Predict afterstate from state and action.

        Parameters
        ----------
        state : Array
            Hidden state, shape (..., hidden_size).
        action : Array
            One-hot encoded action, shape (..., action_size).

        Returns
        -------
        Array
            Afterstate, shape (..., hidden_size).
        """
        # ##>: Project and combine inputs.
        state_proj = nn.Dense(self.hidden_size)(state)
        action_proj = nn.Dense(self.hidden_size)(action)
        x = state_proj + action_proj

        # ##>: Stack of residual blocks.
        x = ResidualStack(self.num_blocks, self.hidden_size)(x)

        # ##>: Final normalization and output.
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        afterstate = nn.Dense(self.hidden_size, name='afterstate')(x)

        return afterstate


class AfterstatePrediction(nn.Module):
    """
    Afterstate Prediction model (ψ): afterstate -> (Q-value, chance_probs).

    Outputs the Q-value of the afterstate and the probability distribution
    over possible chance outcomes (tile spawns).

    Attributes
    ----------
    codebook_size : int
        Number of possible chance outcomes.
    hidden_size : int
        Size of the hidden layers.
    num_blocks : int
        Number of residual blocks.
    """

    codebook_size: int = 32
    hidden_size: int = HIDDEN_UNITS
    num_blocks: int = NUM_RESIDUAL_BLOCKS

    @nn.compact
    def __call__(self, afterstate: Array) -> tuple[Array, Array]:
        """
        Predict Q-value and chance distribution from afterstate.

        Parameters
        ----------
        afterstate : Array
            Afterstate, shape (..., hidden_size).

        Returns
        -------
        tuple[Array, Array]
            - q_value: Q-value estimate, shape (...,)
            - chance_logits: Logits for chance outcomes, shape (..., codebook_size)
        """
        # ##>: Initial projection.
        x = nn.Dense(self.hidden_size)(afterstate)

        # ##>: Stack of residual blocks.
        x = ResidualStack(self.num_blocks, self.hidden_size)(x)

        # ##>: Final normalization.
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # ##>: Q-value head.
        q_value = nn.Dense(1, name='q_value')(x)
        q_value = jnp.squeeze(q_value, axis=-1)

        # ##>: Chance distribution head (logits).
        chance_logits = nn.Dense(self.codebook_size, name='chance_logits')(x)

        return q_value, chance_logits


class Dynamics(nn.Module):
    """
    Stochastic Dynamics model (g): (afterstate, chance_code) -> (next_state, reward).

    Predicts the next hidden state and reward given an afterstate and
    the sampled chance outcome.

    Attributes
    ----------
    hidden_size : int
        Size of the hidden layers and state output.
    codebook_size : int
        Number of possible chance outcomes.
    num_blocks : int
        Number of residual blocks.
    """

    hidden_size: int = HIDDEN_UNITS
    codebook_size: int = 32
    num_blocks: int = NUM_RESIDUAL_BLOCKS

    @nn.compact
    def __call__(self, afterstate: Array, chance_code: Array) -> tuple[Array, Array]:
        """
        Predict next state and reward from afterstate and chance code.

        Parameters
        ----------
        afterstate : Array
            Afterstate, shape (..., hidden_size).
        chance_code : Array
            One-hot encoded chance outcome, shape (..., codebook_size).

        Returns
        -------
        tuple[Array, Array]
            - next_state: Next hidden state, shape (..., hidden_size)
            - reward: Predicted reward, shape (...,)
        """
        # ##>: Project and combine inputs.
        afterstate_proj = nn.Dense(self.hidden_size)(afterstate)
        chance_proj = nn.Dense(self.hidden_size)(chance_code)
        x = afterstate_proj + chance_proj

        # ##>: Stack of residual blocks.
        x = ResidualStack(self.num_blocks, self.hidden_size)(x)

        # ##>: Final normalization.
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # ##>: Next state head.
        next_state = nn.Dense(self.hidden_size, name='next_state')(x)

        # ##>: Reward head.
        reward = nn.Dense(1, name='reward')(x)
        reward = jnp.squeeze(reward, axis=-1)

        return next_state, reward


class Encoder(nn.Module):
    """
    Encoder model (e): observation -> chance_code.

    Encodes an observation into a discrete chance code using
    straight-through gradient estimation for differentiable
    discrete selection.

    Attributes
    ----------
    codebook_size : int
        Number of possible chance codes.
    hidden_size : int
        Size of the hidden layers.
    num_blocks : int
        Number of residual blocks.
    """

    codebook_size: int = 32
    hidden_size: int = HIDDEN_UNITS
    num_blocks: int = NUM_RESIDUAL_BLOCKS

    @nn.compact
    def __call__(self, observation: Array, deterministic: bool = True) -> Array:
        """
        Encode observation to chance code.

        Parameters
        ----------
        observation : Array
            Flattened observation, shape (..., observation_dim).
        deterministic : bool
            If True, use argmax (discrete). If False, use Gumbel-Softmax.

        Returns
        -------
        Array
            One-hot chance code, shape (..., codebook_size).
        """
        # ##>: Initial projection.
        x = nn.Dense(self.hidden_size)(observation)

        # ##>: Stack of residual blocks.
        x = ResidualStack(self.num_blocks, self.hidden_size)(x)

        # ##>: Final normalization.
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # ##>: Output logits for chance code selection.
        logits = nn.Dense(self.codebook_size, name='chance_logits')(x)

        # ##>: Straight-through argmax: forward uses argmax, backward uses identity.
        # ##>: This enables gradient flow through discrete selection.
        if deterministic:
            one_hot = jax.nn.one_hot(jnp.argmax(logits, axis=-1), self.codebook_size)
            # ##>: Straight-through estimator: gradients flow through logits.
            chance_code = logits - jax.lax.stop_gradient(logits) + jax.lax.stop_gradient(one_hot)
        else:
            # ##>: Soft version for exploration during training.
            chance_code = jax.nn.softmax(logits)

        return chance_code
