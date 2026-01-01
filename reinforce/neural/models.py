"""
Neural network model builders for Stochastic MuZero architecture.

This module provides functions to build the five core models:
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

from keras import Model, layers, ops, saving
from numpy import prod

# ##>: Paper-specified architecture constants.
NUM_RESIDUAL_BLOCKS = 10
HIDDEN_UNITS = 256


def identity_block_dense(input_tensor: layers.Layer, units: int = HIDDEN_UNITS) -> layers.Layer:
    """
    Creates a ResNet v2 style pre-activation identity block with dense layers.

    Architecture: LayerNorm -> ReLU -> Dense -> LayerNorm -> ReLU -> Dense -> Add

    Parameters
    ----------
    input_tensor : layers.Layer
        Input layer to the identity block.
    units : int
        Number of units for the dense layers.

    Returns
    -------
    layers.Layer
        Output tensor after applying the identity block.
    """
    # ##>: Pre-activation: normalize and activate before transformation.
    x = layers.LayerNormalization()(input_tensor)
    x = layers.ReLU()(x)
    x = layers.Dense(units)(x)

    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(units)(x)

    # ##>: Residual connection.
    x = layers.Add()([x, input_tensor])

    return x


def build_representation_model(
    input_shape: tuple[int, ...],
    hidden_size: int = HIDDEN_UNITS,
    num_blocks: int = NUM_RESIDUAL_BLOCKS,
) -> Model:
    """
    Build the representation (encoder) model (h).

    This model converts raw observations into hidden state representations.

    Parameters
    ----------
    input_shape : tuple[int, ...]
        Shape of the input observation.
    hidden_size : int
        Size of the hidden state output.
    num_blocks : int
        Number of residual blocks.

    Returns
    -------
    Model
        Keras model for representation.
    """
    inputs = layers.Input(shape=input_shape)

    # ##>: Initial projection to hidden size.
    x = layers.Dense(hidden_size)(inputs)

    # ##>: Stack of residual blocks.
    for _ in range(num_blocks):
        x = identity_block_dense(x, hidden_size)

    # ##>: Final layer norm and output projection.
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(hidden_size, name='hidden_state')(x)

    return Model(inputs=inputs, outputs=outputs, name='representation_model')


def build_dynamics_model(
    state_shape: tuple[int, ...],
    action_size: int,
    hidden_size: int = HIDDEN_UNITS,
    num_blocks: int = NUM_RESIDUAL_BLOCKS,
) -> Model:
    """
    Build the dynamics model (original MuZero style).

    This model predicts the next hidden state and reward given current state and action.

    Parameters
    ----------
    state_shape : tuple[int, ...]
        Shape of the hidden state.
    action_size : int
        Number of possible actions.
    hidden_size : int
        Size of the hidden layers.
    num_blocks : int
        Number of residual blocks.

    Returns
    -------
    Model
        Keras model for dynamics with outputs [next_state, reward].
    """
    # ##>: Input layers.
    input_state = layers.Input(shape=state_shape, name='state_input')
    input_action = layers.Input(shape=(action_size,), name='action_input')

    # ##>: Project and combine inputs.
    dense_state = layers.Dense(hidden_size)(input_state)
    dense_action = layers.Dense(hidden_size)(input_action)
    x = layers.Add()([dense_state, dense_action])

    # ##>: Stack of residual blocks.
    for _ in range(num_blocks):
        x = identity_block_dense(x, hidden_size)

    # ##>: Final normalization.
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)

    # ##>: Next state head.
    next_state = layers.Dense(int(prod(state_shape)), name='next_state')(x)
    next_state = layers.Reshape(state_shape)(next_state)

    # ##>: Reward head.
    reward = layers.Dense(1, name='reward')(x)

    return Model([input_state, input_action], [next_state, reward], name='dynamics_model')


def build_prediction_model(
    state_shape: tuple[int, ...],
    action_size: int,
    hidden_size: int = HIDDEN_UNITS,
    num_blocks: int = NUM_RESIDUAL_BLOCKS,
) -> Model:
    """
    Build the prediction model (f).

    This model outputs policy probabilities and value estimate from a hidden state.

    Parameters
    ----------
    state_shape : tuple[int, ...]
        Shape of the hidden state.
    action_size : int
        Number of possible actions.
    hidden_size : int
        Size of the hidden layers.
    num_blocks : int
        Number of residual blocks.

    Returns
    -------
    Model
        Keras model for prediction with outputs [policy, value].
    """
    inputs = layers.Input(shape=state_shape, name='state_input')

    # ##>: Initial projection.
    x = layers.Dense(hidden_size)(inputs)

    # ##>: Stack of residual blocks.
    for _ in range(num_blocks):
        x = identity_block_dense(x, hidden_size)

    # ##>: Final normalization.
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)

    # ##>: Policy head.
    policy = layers.Dense(action_size, activation='softmax', name='policy')(x)

    # ##>: Value head.
    value = layers.Dense(1, name='value')(x)

    return Model(inputs, [policy, value], name='prediction_model')


def build_afterstate_dynamics_model(
    state_shape: tuple[int, ...],
    action_size: int,
    hidden_size: int = HIDDEN_UNITS,
    num_blocks: int = NUM_RESIDUAL_BLOCKS,
) -> Model:
    """
    Build the afterstate dynamics model (φ).

    This model predicts the afterstate given current state and action.
    The afterstate represents the state after action but before chance event.

    Parameters
    ----------
    state_shape : tuple[int, ...]
        Shape of the hidden state.
    action_size : int
        Number of possible actions.
    hidden_size : int
        Size of the hidden layers.
    num_blocks : int
        Number of residual blocks.

    Returns
    -------
    Model
        Keras model for afterstate dynamics with output afterstate.
    """
    # ##>: Input layers.
    input_state = layers.Input(shape=state_shape, name='state_input')
    input_action = layers.Input(shape=(action_size,), name='action_input')

    # ##>: Project and combine inputs.
    dense_state = layers.Dense(hidden_size)(input_state)
    dense_action = layers.Dense(hidden_size)(input_action)
    x = layers.Add()([dense_state, dense_action])

    # ##>: Stack of residual blocks.
    for _ in range(num_blocks):
        x = identity_block_dense(x, hidden_size)

    # ##>: Final normalization and output.
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    afterstate = layers.Dense(int(prod(state_shape)), name='afterstate')(x)
    afterstate = layers.Reshape(state_shape)(afterstate)

    return Model([input_state, input_action], afterstate, name='afterstate_dynamics_model')


def build_afterstate_prediction_model(
    state_shape: tuple[int, ...],
    codebook_size: int,
    hidden_size: int = HIDDEN_UNITS,
    num_blocks: int = NUM_RESIDUAL_BLOCKS,
) -> Model:
    """
    Build the afterstate prediction model (ψ).

    This model outputs Q-value and chance distribution from an afterstate.
    The chance distribution (σ) predicts probabilities over possible chance codes.

    Parameters
    ----------
    state_shape : tuple[int, ...]
        Shape of the afterstate.
    codebook_size : int
        Number of possible chance outcomes (codebook size for VQ-VAE).
    hidden_size : int
        Size of the hidden layers.
    num_blocks : int
        Number of residual blocks.

    Returns
    -------
    Model
        Keras model for afterstate prediction with outputs [q_value, chance_probs].
    """
    inputs = layers.Input(shape=state_shape, name='afterstate_input')

    # ##>: Initial projection.
    x = layers.Dense(hidden_size)(inputs)

    # ##>: Stack of residual blocks.
    for _ in range(num_blocks):
        x = identity_block_dense(x, hidden_size)

    # ##>: Final normalization.
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)

    # ##>: Q-value head.
    q_value = layers.Dense(1, name='q_value')(x)

    # ##>: Chance distribution head (σ).
    chance_probs = layers.Dense(codebook_size, activation='softmax', name='chance_probs')(x)

    return Model(inputs, [q_value, chance_probs], name='afterstate_prediction_model')


def build_stochastic_dynamics_model(
    state_shape: tuple[int, ...],
    codebook_size: int,
    hidden_size: int = HIDDEN_UNITS,
    num_blocks: int = NUM_RESIDUAL_BLOCKS,
) -> Model:
    """
    Build the stochastic dynamics model (g).

    This model predicts the next state and reward given afterstate and chance code.
    This handles the stochastic transition from afterstate to next state.

    Parameters
    ----------
    state_shape : tuple[int, ...]
        Shape of the afterstate/state.
    codebook_size : int
        Number of possible chance outcomes.
    hidden_size : int
        Size of the hidden layers.
    num_blocks : int
        Number of residual blocks.

    Returns
    -------
    Model
        Keras model for stochastic dynamics with outputs [next_state, reward].
    """
    # ##>: Input layers.
    input_afterstate = layers.Input(shape=state_shape, name='afterstate_input')
    input_chance = layers.Input(shape=(codebook_size,), name='chance_input')

    # ##>: Project and combine inputs.
    dense_afterstate = layers.Dense(hidden_size)(input_afterstate)
    dense_chance = layers.Dense(hidden_size)(input_chance)
    x = layers.Add()([dense_afterstate, dense_chance])

    # ##>: Stack of residual blocks.
    for _ in range(num_blocks):
        x = identity_block_dense(x, hidden_size)

    # ##>: Final normalization.
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)

    # ##>: Next state head.
    next_state = layers.Dense(int(prod(state_shape)), name='next_state')(x)
    next_state = layers.Reshape(state_shape)(next_state)

    # ##>: Reward head.
    reward = layers.Dense(1, name='reward')(x)

    return Model([input_afterstate, input_chance], [next_state, reward], name='stochastic_dynamics_model')


@saving.register_keras_serializable(package='reinforce')
class StraightThroughArgmax(layers.Layer):
    """
    Straight-through estimator for argmax operation.

    Forward pass: returns one-hot of argmax (discrete selection).
    Backward pass: passes gradients through as if identity function.
    This enables learning discrete representations with gradient descent.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Apply straight-through argmax.

        Parameters
        ----------
        inputs : Tensor
            Logits tensor of shape (batch_size, codebook_size).

        Returns
        -------
        Tensor
            One-hot tensor of shape (batch_size, codebook_size).
        """
        # ##>: Forward pass: argmax to one-hot.
        indices = ops.argmax(inputs, axis=-1)
        one_hot = ops.one_hot(indices, ops.shape(inputs)[-1])

        # ##>: Straight-through: use one_hot in forward, but gradients flow through inputs.
        # ##&: This is the Gumbel-softmax trick with temperature=0.
        return inputs + ops.stop_gradient(one_hot - inputs)


def build_encoder_model(
    input_shape: tuple[int, ...],
    codebook_size: int,
    hidden_size: int = HIDDEN_UNITS,
    num_blocks: int = NUM_RESIDUAL_BLOCKS,
) -> Model:
    """
    Build the encoder model (e).

    This model encodes an observation into a discrete chance code.
    Uses straight-through gradient estimation for discrete selection.

    Parameters
    ----------
    input_shape : tuple[int, ...]
        Shape of the input observation.
    codebook_size : int
        Number of possible chance codes (codebook size).
    hidden_size : int
        Size of the hidden layers.
    num_blocks : int
        Number of residual blocks.

    Returns
    -------
    Model
        Keras model for encoding with output chance_code (one-hot).
    """
    inputs = layers.Input(shape=input_shape, name='observation_input')

    # ##>: Initial projection.
    x = layers.Dense(hidden_size)(inputs)

    # ##>: Stack of residual blocks.
    for _ in range(num_blocks):
        x = identity_block_dense(x, hidden_size)

    # ##>: Final normalization.
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)

    # ##>: Output logits for chance code selection.
    logits = layers.Dense(codebook_size, name='chance_logits')(x)

    # ##>: Apply straight-through argmax to get discrete one-hot code.
    chance_code = StraightThroughArgmax(name='chance_code')(logits)

    return Model(inputs, chance_code, name='encoder_model')
