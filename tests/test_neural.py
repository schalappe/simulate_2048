"""
Tests for Stochastic MuZero neural network components.

This module tests all model builders and the StochasticNetwork wrapper.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from reinforce.neural.models import (
    StraightThroughArgmax,
    build_afterstate_dynamics_model,
    build_afterstate_prediction_model,
    build_encoder_model,
    build_prediction_model,
    build_representation_model,
    build_stochastic_dynamics_model,
)
from reinforce.neural.network import (
    NUM_ACTIONS,
    NetworkOutput,
    StochasticNetwork,
    create_stochastic_network,
)

# Test constants - smaller values for faster tests
OBSERVATION_SHAPE = (16,)  # Flattened 4x4 board
STATE_SIZE = 64  # Smaller than paper's 256 for faster tests
CODEBOOK_SIZE = 8  # Smaller than paper's 32 for faster tests
NUM_BLOCKS = 2  # Smaller than paper's 10 for faster tests


class TestModelBuilders:
    """Tests for individual model builder functions."""

    def test_build_representation_model(self):
        """Test representation model builds correctly."""
        model = build_representation_model(OBSERVATION_SHAPE, STATE_SIZE, NUM_BLOCKS)

        assert model.input_shape == (None, *OBSERVATION_SHAPE)
        assert model.output_shape == (None, STATE_SIZE)
        assert model.name == 'representation_model'

    def test_build_prediction_model(self):
        """Test prediction model builds correctly."""
        state_shape = (STATE_SIZE,)
        model = build_prediction_model(state_shape, NUM_ACTIONS, STATE_SIZE, NUM_BLOCKS)

        assert model.input_shape == (None, STATE_SIZE)
        # ##>: Output is [policy, value].
        assert len(model.outputs) == 2
        assert model.output_shape[0] == (None, NUM_ACTIONS)  # policy
        assert model.output_shape[1] == (None, 1)  # value
        assert model.name == 'prediction_model'

    def test_build_afterstate_dynamics_model(self):
        """Test afterstate dynamics model builds correctly."""
        state_shape = (STATE_SIZE,)
        model = build_afterstate_dynamics_model(state_shape, NUM_ACTIONS, STATE_SIZE, NUM_BLOCKS)

        # ##>: Two inputs: state and action.
        assert len(model.inputs) == 2
        assert model.input_shape[0] == (None, STATE_SIZE)  # state
        assert model.input_shape[1] == (None, NUM_ACTIONS)  # action (one-hot)
        assert model.output_shape == (None, STATE_SIZE)  # afterstate
        assert model.name == 'afterstate_dynamics_model'

    def test_build_afterstate_prediction_model(self):
        """Test afterstate prediction model builds correctly."""
        state_shape = (STATE_SIZE,)
        model = build_afterstate_prediction_model(state_shape, CODEBOOK_SIZE, STATE_SIZE, NUM_BLOCKS)

        assert model.input_shape == (None, STATE_SIZE)
        # ##>: Output is [q_value, chance_probs].
        assert len(model.outputs) == 2
        assert model.output_shape[0] == (None, 1)  # q_value
        assert model.output_shape[1] == (None, CODEBOOK_SIZE)  # chance_probs
        assert model.name == 'afterstate_prediction_model'

    def test_build_stochastic_dynamics_model(self):
        """Test stochastic dynamics model builds correctly."""
        state_shape = (STATE_SIZE,)
        model = build_stochastic_dynamics_model(state_shape, CODEBOOK_SIZE, STATE_SIZE, NUM_BLOCKS)

        # ##>: Two inputs: afterstate and chance_code.
        assert len(model.inputs) == 2
        assert model.input_shape[0] == (None, STATE_SIZE)  # afterstate
        assert model.input_shape[1] == (None, CODEBOOK_SIZE)  # chance_code
        # ##>: Output is [next_state, reward].
        assert len(model.outputs) == 2
        assert model.output_shape[0] == (None, STATE_SIZE)  # next_state
        assert model.output_shape[1] == (None, 1)  # reward
        assert model.name == 'stochastic_dynamics_model'

    def test_build_encoder_model(self):
        """Test encoder model builds correctly."""
        model = build_encoder_model(OBSERVATION_SHAPE, CODEBOOK_SIZE, STATE_SIZE, NUM_BLOCKS)

        assert model.input_shape == (None, *OBSERVATION_SHAPE)
        assert model.output_shape == (None, CODEBOOK_SIZE)  # one-hot code
        assert model.name == 'encoder_model'


class TestStraightThroughArgmax:
    """Tests for the StraightThroughArgmax layer."""

    def test_output_is_one_hot(self):
        """Test that output is a valid one-hot vector."""
        layer = StraightThroughArgmax()
        logits = np.array([[1.0, 2.0, 0.5, 3.0]])

        output = layer(logits).numpy()

        # ##>: Should be one-hot with max at index 3.
        expected = np.array([[0.0, 0.0, 0.0, 1.0]])
        assert_array_equal(output, expected)

    def test_batch_processing(self):
        """Test that layer handles batches correctly."""
        layer = StraightThroughArgmax()
        logits = np.array(
            [
                [1.0, 2.0, 0.5],
                [3.0, 1.0, 0.5],
                [0.5, 0.5, 2.0],
            ]
        )

        output = layer(logits).numpy()

        # ##>: Each row should be one-hot at the argmax position.
        assert output.shape == (3, 3)
        assert np.allclose(output.sum(axis=1), 1.0)  # Each row sums to 1
        assert np.allclose(output[0], [0.0, 1.0, 0.0])  # max at 1
        assert np.allclose(output[1], [1.0, 0.0, 0.0])  # max at 0
        assert np.allclose(output[2], [0.0, 0.0, 1.0])  # max at 2


class TestModelInference:
    """Tests for model inference (forward pass)."""

    def test_representation_forward(self):
        """Test representation model forward pass."""
        model = build_representation_model(OBSERVATION_SHAPE, STATE_SIZE, NUM_BLOCKS)
        observation = np.random.randn(1, *OBSERVATION_SHAPE).astype(np.float32)

        output = model(observation)

        assert output.shape == (1, STATE_SIZE)

    def test_prediction_forward(self):
        """Test prediction model forward pass."""
        model = build_prediction_model((STATE_SIZE,), NUM_ACTIONS, STATE_SIZE, NUM_BLOCKS)
        state = np.random.randn(1, STATE_SIZE).astype(np.float32)

        policy, value = model(state)

        assert policy.shape == (1, NUM_ACTIONS)
        assert value.shape == (1, 1)
        # ##>: Policy should be valid probabilities.
        assert np.allclose(policy.numpy().sum(), 1.0, atol=1e-5)
        assert np.all(policy.numpy() >= 0)

    def test_afterstate_dynamics_forward(self):
        """Test afterstate dynamics model forward pass."""
        model = build_afterstate_dynamics_model((STATE_SIZE,), NUM_ACTIONS, STATE_SIZE, NUM_BLOCKS)
        state = np.random.randn(1, STATE_SIZE).astype(np.float32)
        action = np.zeros((1, NUM_ACTIONS), dtype=np.float32)
        action[0, 1] = 1.0  # One-hot action

        afterstate = model([state, action])

        assert afterstate.shape == (1, STATE_SIZE)

    def test_afterstate_prediction_forward(self):
        """Test afterstate prediction model forward pass."""
        model = build_afterstate_prediction_model((STATE_SIZE,), CODEBOOK_SIZE, STATE_SIZE, NUM_BLOCKS)
        afterstate = np.random.randn(1, STATE_SIZE).astype(np.float32)

        q_value, chance_probs = model(afterstate)

        assert q_value.shape == (1, 1)
        assert chance_probs.shape == (1, CODEBOOK_SIZE)
        # ##>: Chance probs should be valid probabilities.
        assert np.allclose(chance_probs.numpy().sum(), 1.0, atol=1e-5)
        assert np.all(chance_probs.numpy() >= 0)

    def test_stochastic_dynamics_forward(self):
        """Test stochastic dynamics model forward pass."""
        model = build_stochastic_dynamics_model((STATE_SIZE,), CODEBOOK_SIZE, STATE_SIZE, NUM_BLOCKS)
        afterstate = np.random.randn(1, STATE_SIZE).astype(np.float32)
        chance_code = np.zeros((1, CODEBOOK_SIZE), dtype=np.float32)
        chance_code[0, 2] = 1.0  # One-hot code

        next_state, reward = model([afterstate, chance_code])

        assert next_state.shape == (1, STATE_SIZE)
        assert reward.shape == (1, 1)

    def test_encoder_forward(self):
        """Test encoder model forward pass."""
        model = build_encoder_model(OBSERVATION_SHAPE, CODEBOOK_SIZE, STATE_SIZE, NUM_BLOCKS)
        observation = np.random.randn(1, *OBSERVATION_SHAPE).astype(np.float32)

        chance_code = model(observation)

        assert chance_code.shape == (1, CODEBOOK_SIZE)
        # ##>: Output should be one-hot (due to StraightThroughArgmax).
        assert np.allclose(chance_code.numpy().sum(), 1.0, atol=1e-5)


class TestStochasticNetwork:
    """Tests for the StochasticNetwork wrapper class."""

    @pytest.fixture
    def network(self):
        """Create a StochasticNetwork for testing."""
        return create_stochastic_network(
            observation_shape=OBSERVATION_SHAPE,
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
        )

    def test_create_stochastic_network(self, network):
        """Test factory function creates valid network."""
        assert isinstance(network, StochasticNetwork)
        assert network.codebook_size == CODEBOOK_SIZE

    def test_representation(self, network):
        """Test representation method."""
        observation = np.random.randn(*OBSERVATION_SHAPE).astype(np.float32)

        state = network.representation(observation)

        assert state.shape == (STATE_SIZE,)

    def test_prediction(self, network):
        """Test prediction method."""
        state = np.random.randn(STATE_SIZE).astype(np.float32)

        output = network.prediction(state)

        assert isinstance(output, NetworkOutput)
        assert isinstance(output.value, float)
        assert output.policy is not None
        assert output.policy.shape == (NUM_ACTIONS,)
        # ##>: Policy should be valid probabilities.
        assert np.allclose(output.policy.sum(), 1.0, atol=1e-5)

    def test_afterstate_dynamics(self, network):
        """Test afterstate_dynamics method."""
        state = np.random.randn(STATE_SIZE).astype(np.float32)
        action = 2

        afterstate = network.afterstate_dynamics(state, action)

        assert afterstate.shape == (STATE_SIZE,)

    def test_afterstate_prediction(self, network):
        """Test afterstate_prediction method."""
        afterstate = np.random.randn(STATE_SIZE).astype(np.float32)

        output = network.afterstate_prediction(afterstate)

        assert isinstance(output, NetworkOutput)
        assert isinstance(output.value, float)
        assert output.chance_probs is not None
        assert output.chance_probs.shape == (CODEBOOK_SIZE,)
        # ##>: Chance probs should be valid probabilities.
        assert np.allclose(output.chance_probs.sum(), 1.0, atol=1e-5)

    def test_dynamics(self, network):
        """Test dynamics method."""
        afterstate = np.random.randn(STATE_SIZE).astype(np.float32)
        chance_code = np.zeros(CODEBOOK_SIZE, dtype=np.float32)
        chance_code[3] = 1.0

        next_state, reward = network.dynamics(afterstate, chance_code)

        assert next_state.shape == (STATE_SIZE,)
        assert isinstance(reward, float)

    def test_encoder(self, network):
        """Test encoder method."""
        observation = np.random.randn(*OBSERVATION_SHAPE).astype(np.float32)

        chance_code = network.encoder(observation)

        assert chance_code.shape == (CODEBOOK_SIZE,)
        # ##>: Output should be one-hot.
        assert np.allclose(chance_code.sum(), 1.0, atol=1e-5)


class TestFullDataFlow:
    """Tests for the complete Stochastic MuZero data flow."""

    @pytest.fixture
    def network(self):
        """Create a StochasticNetwork for testing."""
        return create_stochastic_network(
            observation_shape=OBSERVATION_SHAPE,
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
        )

    def test_decision_node_flow(self, network):
        """
        Test data flow for decision nodes.

        observation -> representation -> state -> prediction -> (policy, value)
        """
        observation = np.random.randn(*OBSERVATION_SHAPE).astype(np.float32)

        # ##>: Step 1: Representation.
        state = network.representation(observation)
        assert state.shape == (STATE_SIZE,)

        # ##>: Step 2: Prediction.
        output = network.prediction(state)
        assert output.policy.shape == (NUM_ACTIONS,)

    def test_chance_node_flow(self, network):
        """
        Test data flow for chance nodes.

        state + action -> afterstate_dynamics -> afterstate
        afterstate -> afterstate_prediction -> (Q-value, σ)
        """
        state = np.random.randn(STATE_SIZE).astype(np.float32)
        action = 1

        # ##>: Step 1: Afterstate dynamics.
        afterstate = network.afterstate_dynamics(state, action)
        assert afterstate.shape == (STATE_SIZE,)

        # ##>: Step 2: Afterstate prediction.
        output = network.afterstate_prediction(afterstate)
        assert output.chance_probs.shape == (CODEBOOK_SIZE,)

    def test_stochastic_transition_flow(self, network):
        """
        Test data flow for stochastic transitions.

        afterstate + chance_code -> dynamics -> (next_state, reward)
        """
        afterstate = np.random.randn(STATE_SIZE).astype(np.float32)
        chance_code = np.zeros(CODEBOOK_SIZE, dtype=np.float32)
        chance_code[0] = 1.0

        # ##>: Stochastic dynamics.
        next_state, reward = network.dynamics(afterstate, chance_code)
        assert next_state.shape == (STATE_SIZE,)
        assert isinstance(reward, float)

    def test_full_unroll_step(self, network):
        """
        Test complete unroll step as done during training.

        1. observation -> representation -> state
        2. state -> prediction -> (policy, value)
        3. state + action -> afterstate_dynamics -> afterstate
        4. afterstate -> afterstate_prediction -> (Q, σ)
        5. next_observation -> encoder -> chance_code
        6. afterstate + chance_code -> dynamics -> (next_state, reward)
        7. next_state -> prediction -> (policy, value)
        """
        observation = np.random.randn(*OBSERVATION_SHAPE).astype(np.float32)
        next_observation = np.random.randn(*OBSERVATION_SHAPE).astype(np.float32)
        action = 0

        # ##>: Step 1: Initial representation.
        state = network.representation(observation)

        # ##>: Step 2: Decision node prediction.
        pred_output = network.prediction(state)
        assert pred_output.policy is not None

        # ##>: Step 3: Afterstate dynamics (deterministic action effect).
        afterstate = network.afterstate_dynamics(state, action)

        # ##>: Step 4: Afterstate prediction (Q-value and chance distribution).
        as_output = network.afterstate_prediction(afterstate)
        assert as_output.chance_probs is not None

        # ##>: Step 5: Encode next observation to get chance code.
        chance_code = network.encoder(next_observation)
        assert np.allclose(chance_code.sum(), 1.0, atol=1e-5)

        # ##>: Step 6: Stochastic dynamics (apply chance transition).
        next_state, reward = network.dynamics(afterstate, chance_code)

        # ##>: Step 7: Next state prediction.
        next_pred_output = network.prediction(next_state)
        assert next_pred_output.policy is not None
