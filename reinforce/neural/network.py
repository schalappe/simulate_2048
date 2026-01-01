"""
Unified neural network wrapper for Stochastic MuZero models.

This module provides network classes that unify the MuZero models:
- Network: Original 3-model wrapper (representation, dynamics, prediction)
- StochasticNetwork: Full 5-model wrapper for Stochastic MuZero

Reference: "Planning in Stochastic Environments with a Learned Model" (ICLR 2022)
"""

from __future__ import annotations

from dataclasses import dataclass

from keras import KerasTensor, Model, models, ops, utils
from numpy import ndarray

NUM_ACTIONS = 4  # 2048 game: left, up, right, down
DEFAULT_CODEBOOK_SIZE = 32  # Default number of chance codes


def ndarray_to_tensor(inputs: ndarray, expand: bool = True) -> KerasTensor:
    """
    Converts a NumPy array to a Keras tensor.

    This function accepts a 2D NumPy array as input and converts it into a Keras Tensor.
    The resulting Keras Tensor is automatically expanded to match the shape (batch_size, num_features).

    Parameters
    ----------
    inputs : ndarray
        A 2D NumPy array with shape (num_samples, num_features).
    expand : bool
        Whether or not expand the tensor.

    Returns
    -------
    KerasTensor
        A Keras tensor with shape (1, num_samples, num_features) or simply (num_samples, num_features)
            if the input is a single sample.

    Note
    ----
    The dtype of the output Keras Tensor is 'float16'.

    Examples
    --------
    >>> import numpy as np
    >>> ndarray_to_tensor(np.array([[3.0], [4.0]]))
    <KerasTensor: shape=(1, 2), dtype=float16, name='ndarray_to_tensor/ExpandDims', tensorflow_grad>
    """
    obs_tensor = ops.convert_to_tensor(inputs, dtype='float16')

    if expand:
        return ops.expand_dims(obs_tensor, 0)
    return obs_tensor


class Network:
    """
    Unified wrapper for MuZero-style neural network models.

    This class provides a clean interface for the three models:
    - Representation (encoder): observation -> hidden state
    - Dynamics: (state, action) -> (next_state, reward)
    - Prediction: state -> (policy, value)

    Attributes
    ----------
    _encoder : Model
        The representation model.
    _dynamic : Model
        The dynamics model.
    _predictor : Model
        The prediction model.
    """

    def __init__(self, encoder: Model, dynamic: Model, predictor: Model):
        """
        Initialize the Network with pre-built models.

        Parameters
        ----------
        encoder : Model
            The representation model.
        dynamic : Model
            The dynamics model.
        predictor : Model
            The prediction model.
        """
        self._encoder = encoder
        self._dynamic = dynamic
        self._predictor = predictor

    @classmethod
    def from_path(cls, encoder_path: str, dynamic_path: str, predictor_path: str) -> Network:
        """
        Load a Network from saved model files.

        Parameters
        ----------
        encoder_path : str
            Path to the saved encoder model.
        dynamic_path : str
            Path to the saved dynamics model.
        predictor_path : str
            Path to the saved predictor model.

        Returns
        -------
        Network
            A Network instance with loaded models.
        """
        return cls(
            encoder=models.load_model(encoder_path),
            dynamic=models.load_model(dynamic_path),
            predictor=models.load_model(predictor_path),
        )

    def representation(self, observation: ndarray) -> ndarray:
        """
        Encode an observation into a hidden state.

        Parameters
        ----------
        observation : ndarray
            The raw observation.

        Returns
        -------
        ndarray
            The hidden state representation.
        """
        return self._encoder(ndarray_to_tensor(observation))[0]

    def dynamics(self, state: ndarray, action: int) -> tuple[ndarray, float]:
        """
        Predict the next state and reward given current state and action.

        Parameters
        ----------
        state : ndarray
            The current hidden state.
        action : int
            The action to take (0-3 for 2048).

        Returns
        -------
        Tuple[ndarray, float]
            The predicted next state and reward.
        """
        next_state, reward = self._dynamic(
            [
                ndarray_to_tensor(state),
                ndarray_to_tensor(utils.to_categorical([action], num_classes=NUM_ACTIONS), expand=False),
            ]
        )
        return next_state[0], reward[0][0]

    def prediction(self, state: ndarray) -> tuple[ndarray, float]:
        """
        Predict policy and value from a hidden state.

        Parameters
        ----------
        state : ndarray
            The hidden state.

        Returns
        -------
        Tuple[ndarray, float]
            The policy probabilities and value estimate.
        """
        policy, value = self._predictor(ndarray_to_tensor(state))
        return policy[0], value[0][0]


@dataclass
class NetworkOutput:
    """
    Container for network prediction outputs.

    Attributes
    ----------
    value : float
        The value or Q-value prediction.
    policy : ndarray | None
        The policy probabilities (for decision nodes).
    reward : float
        The predicted reward (0 for afterstates).
    chance_probs : ndarray | None
        The chance distribution σ (for afterstates).
    """

    value: float
    policy: ndarray | None = None
    reward: float = 0.0
    chance_probs: ndarray | None = None


class StochasticNetwork:
    """
    Unified wrapper for Stochastic MuZero neural network models.

    This class provides a clean interface for the five models:
    - Representation (h): observation -> hidden state
    - Prediction (f): state -> (policy, value)
    - Afterstate Dynamics (φ): (state, action) -> afterstate
    - Afterstate Prediction (ψ): afterstate -> (Q-value, chance_probs)
    - Dynamics (g): (afterstate, chance_code) -> (next_state, reward)
    - Encoder (e): observation -> chance_code

    Attributes
    ----------
    _representation : Model
        The representation model (h).
    _prediction : Model
        The prediction model (f).
    _afterstate_dynamics : Model
        The afterstate dynamics model (φ).
    _afterstate_prediction : Model
        The afterstate prediction model (ψ).
    _dynamics : Model
        The stochastic dynamics model (g).
    _encoder : Model
        The encoder model (e).
    _codebook_size : int
        Number of possible chance codes.
    """

    def __init__(
        self,
        representation: Model,
        prediction: Model,
        afterstate_dynamics: Model,
        afterstate_prediction: Model,
        dynamics: Model,
        encoder: Model,
        codebook_size: int = DEFAULT_CODEBOOK_SIZE,
    ):
        """
        Initialize the StochasticNetwork with pre-built models.

        Parameters
        ----------
        representation : Model
            The representation model (h).
        prediction : Model
            The prediction model (f).
        afterstate_dynamics : Model
            The afterstate dynamics model (φ).
        afterstate_prediction : Model
            The afterstate prediction model (ψ).
        dynamics : Model
            The stochastic dynamics model (g).
        encoder : Model
            The encoder model (e).
        codebook_size : int
            Number of possible chance codes.
        """
        self._representation = representation
        self._prediction = prediction
        self._afterstate_dynamics = afterstate_dynamics
        self._afterstate_prediction = afterstate_prediction
        self._dynamics = dynamics
        self._encoder = encoder
        self._codebook_size = codebook_size

    @property
    def codebook_size(self) -> int:
        """Return the codebook size."""
        return self._codebook_size

    @classmethod
    def from_path(
        cls,
        representation_path: str,
        prediction_path: str,
        afterstate_dynamics_path: str,
        afterstate_prediction_path: str,
        dynamics_path: str,
        encoder_path: str,
        codebook_size: int = DEFAULT_CODEBOOK_SIZE,
    ) -> StochasticNetwork:
        """
        Load a StochasticNetwork from saved model files.

        Parameters
        ----------
        representation_path : str
            Path to the saved representation model.
        prediction_path : str
            Path to the saved prediction model.
        afterstate_dynamics_path : str
            Path to the saved afterstate dynamics model.
        afterstate_prediction_path : str
            Path to the saved afterstate prediction model.
        dynamics_path : str
            Path to the saved dynamics model.
        encoder_path : str
            Path to the saved encoder model.
        codebook_size : int
            Number of possible chance codes.

        Returns
        -------
        StochasticNetwork
            A StochasticNetwork instance with loaded models.
        """
        return cls(
            representation=models.load_model(representation_path),
            prediction=models.load_model(prediction_path),
            afterstate_dynamics=models.load_model(afterstate_dynamics_path),
            afterstate_prediction=models.load_model(afterstate_prediction_path),
            dynamics=models.load_model(dynamics_path),
            encoder=models.load_model(encoder_path),
            codebook_size=codebook_size,
        )

    def representation(self, observation: ndarray) -> ndarray:
        """
        Encode an observation into a hidden state (h).

        Parameters
        ----------
        observation : ndarray
            The raw observation.

        Returns
        -------
        ndarray
            The hidden state representation s^0.
        """
        return self._representation(ndarray_to_tensor(observation))[0]

    def prediction(self, state: ndarray) -> NetworkOutput:
        """
        Predict policy and value from a hidden state (f).

        Parameters
        ----------
        state : ndarray
            The hidden state s^k.

        Returns
        -------
        NetworkOutput
            Contains policy (p^k) and value (v^k).
        """
        policy, value = self._prediction(ndarray_to_tensor(state))
        return NetworkOutput(value=float(value[0][0]), policy=policy[0].numpy())

    def afterstate_dynamics(self, state: ndarray, action: int) -> ndarray:
        """
        Predict afterstate given state and action (φ).

        Parameters
        ----------
        state : ndarray
            The current hidden state s^k.
        action : int
            The action to take (0-3 for 2048).

        Returns
        -------
        ndarray
            The afterstate as^k.
        """
        action_onehot = utils.to_categorical([action], num_classes=NUM_ACTIONS)
        afterstate = self._afterstate_dynamics(
            [ndarray_to_tensor(state), ndarray_to_tensor(action_onehot, expand=False)]
        )
        return afterstate[0]

    def afterstate_prediction(self, afterstate: ndarray) -> NetworkOutput:
        """
        Predict Q-value and chance distribution from afterstate (ψ).

        Parameters
        ----------
        afterstate : ndarray
            The afterstate as^k.

        Returns
        -------
        NetworkOutput
            Contains Q-value (Q^k) and chance_probs (σ^k).
        """
        q_value, chance_probs = self._afterstate_prediction(ndarray_to_tensor(afterstate))
        return NetworkOutput(value=float(q_value[0][0]), chance_probs=chance_probs[0].numpy())

    def dynamics(self, afterstate: ndarray, chance_code: ndarray) -> tuple[ndarray, float]:
        """
        Predict next state and reward given afterstate and chance code (g).

        Parameters
        ----------
        afterstate : ndarray
            The afterstate as^k.
        chance_code : ndarray
            The chance code c (one-hot encoded).

        Returns
        -------
        tuple[ndarray, float]
            The next state s^{k+1} and reward r^{k+1}.
        """
        # ##>: Both inputs need batch dimension for the model.
        next_state, reward = self._dynamics([ndarray_to_tensor(afterstate), ndarray_to_tensor(chance_code)])
        return next_state[0], float(reward[0][0])

    def encoder(self, observation: ndarray) -> ndarray:
        """
        Encode an observation into a chance code (e).

        Parameters
        ----------
        observation : ndarray
            The raw observation o_{≤t}.

        Returns
        -------
        ndarray
            The chance code c (one-hot encoded).
        """
        return self._encoder(ndarray_to_tensor(observation))[0].numpy()


def create_stochastic_network(
    observation_shape: tuple[int, ...],
    hidden_size: int | None = None,
    codebook_size: int = DEFAULT_CODEBOOK_SIZE,
) -> StochasticNetwork:
    """
    Factory function to create a complete StochasticNetwork for 2048.

    This creates all 6 models required for Stochastic MuZero and wraps them
    in a StochasticNetwork instance.

    Parameters
    ----------
    observation_shape : tuple[int, ...]
        Shape of the input observation (e.g., (16,) for flattened 4x4 board).
    hidden_size : int | None
        Size of the hidden state representation. Defaults to paper value (256).
    codebook_size : int
        Number of possible chance codes.

    Returns
    -------
    StochasticNetwork
        A fully initialized StochasticNetwork ready for training.

    Examples
    --------
    >>> network = create_stochastic_network(observation_shape=(16,))
    >>> state = network.representation(observation)
    >>> output = network.prediction(state)
    """
    # ##&: Import here to avoid circular imports.
    from .models import (
        HIDDEN_UNITS,
        build_afterstate_dynamics_model,
        build_afterstate_prediction_model,
        build_encoder_model,
        build_prediction_model,
        build_representation_model,
        build_stochastic_dynamics_model,
    )

    # ##>: Use paper default if not specified.
    if hidden_size is None:
        hidden_size = HIDDEN_UNITS

    state_shape = (hidden_size,)

    # ##>: Build all 6 models.
    representation = build_representation_model(observation_shape, hidden_size)
    prediction = build_prediction_model(state_shape, NUM_ACTIONS)
    afterstate_dynamics = build_afterstate_dynamics_model(state_shape, NUM_ACTIONS)
    afterstate_prediction = build_afterstate_prediction_model(state_shape, codebook_size)
    dynamics = build_stochastic_dynamics_model(state_shape, codebook_size)
    encoder = build_encoder_model(observation_shape, codebook_size)

    return StochasticNetwork(
        representation=representation,
        prediction=prediction,
        afterstate_dynamics=afterstate_dynamics,
        afterstate_prediction=afterstate_prediction,
        dynamics=dynamics,
        encoder=encoder,
        codebook_size=codebook_size,
    )
