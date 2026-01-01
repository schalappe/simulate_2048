"""
Tests for batched MCTS implementation with virtual loss.

This module tests:
- Level 1: Batch network methods (afterstate_dynamics_batch, prediction_batch, etc.)
- Level 2: Leaf batching MCTS (run_batched_mcts)
- Level 3: Threaded MCTS with virtual loss (run_threaded_mcts)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from reinforce.mcts.batched_agent import BatchedMuZeroAgent, MCTSMode
from reinforce.mcts.batched_search import (
    ChanceNode,
    DecisionNode,
    MinMaxStats,
    apply_virtual_loss,
    get_policy_from_visits,
    remove_virtual_loss,
    run_batched_mcts,
    run_threaded_mcts,
    select_action_from_root,
)
from reinforce.neural.network import create_stochastic_network

# Test constants - smaller values for faster tests.
OBSERVATION_SHAPE = (16,)
STATE_SIZE = 64
CODEBOOK_SIZE = 8


@pytest.fixture
def network():
    """Create a small network for testing."""
    return create_stochastic_network(
        observation_shape=OBSERVATION_SHAPE, hidden_size=STATE_SIZE, codebook_size=CODEBOOK_SIZE
    )


@pytest.fixture
def sample_state():
    """Create a sample 2048 game state."""
    return np.array([[2, 4, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])


class TestBatchNetworkMethods:
    """Tests for Level 1: Batch network methods."""

    def test_afterstate_dynamics_batch_empty(self, network):
        """Empty action list returns empty result."""
        state = network.representation(np.zeros(16))
        result = network.afterstate_dynamics_batch(state, [])
        assert result == []

    def test_afterstate_dynamics_batch_single(self, network):
        """Single action batch matches sequential call."""
        state = network.representation(np.zeros(16))

        sequential = network.afterstate_dynamics(state, 0)
        batched = network.afterstate_dynamics_batch(state, [0])

        assert len(batched) == 1
        assert_allclose(batched[0], sequential, rtol=1e-5)

    def test_afterstate_dynamics_batch_multiple(self, network):
        """Multiple actions batched matches sequential calls."""
        state = network.representation(np.zeros(16))
        actions = [0, 1, 2, 3]

        sequential = [network.afterstate_dynamics(state, a) for a in actions]
        batched = network.afterstate_dynamics_batch(state, actions)

        assert len(batched) == 4
        for seq, bat in zip(sequential, batched, strict=True):
            assert_allclose(bat, seq, rtol=1e-5)

    def test_prediction_batch_empty(self, network):
        """Empty state list returns empty result."""
        result = network.prediction_batch([])
        assert result == []

    def test_prediction_batch_single(self, network):
        """Single state batch matches sequential call."""
        state = network.representation(np.zeros(16))

        sequential = network.prediction(state)
        batched = network.prediction_batch([state])

        assert len(batched) == 1
        assert_allclose(batched[0].value, sequential.value, rtol=1e-5)
        assert_allclose(batched[0].policy, sequential.policy, rtol=1e-5)

    def test_prediction_batch_multiple(self, network):
        """Multiple states batched matches sequential calls."""
        states = [network.representation(np.random.randn(16).astype(np.float32)) for _ in range(4)]

        sequential = [network.prediction(s) for s in states]
        batched = network.prediction_batch(states)

        assert len(batched) == 4
        for seq, bat in zip(sequential, batched, strict=True):
            assert_allclose(bat.value, seq.value, rtol=1e-5)
            assert_allclose(bat.policy, seq.policy, rtol=1e-5)

    def test_afterstate_prediction_batch_empty(self, network):
        """Empty afterstate list returns empty result."""
        result = network.afterstate_prediction_batch([])
        assert result == []

    def test_afterstate_prediction_batch_single(self, network):
        """Single afterstate batch matches sequential call."""
        state = network.representation(np.zeros(16))
        afterstate = network.afterstate_dynamics(state, 0)

        sequential = network.afterstate_prediction(afterstate)
        batched = network.afterstate_prediction_batch([afterstate])

        assert len(batched) == 1
        assert_allclose(batched[0].value, sequential.value, rtol=1e-5)
        assert_allclose(batched[0].chance_probs, sequential.chance_probs, rtol=1e-5)

    def test_dynamics_batch_empty(self, network):
        """Empty dynamics batch returns empty result."""
        result = network.dynamics_batch([], [])
        assert result == []

    def test_dynamics_batch_single(self, network):
        """Single dynamics batch matches sequential call."""
        state = network.representation(np.zeros(16))
        afterstate = network.afterstate_dynamics(state, 0)
        chance_code = np.zeros(CODEBOOK_SIZE)
        chance_code[0] = 1.0

        seq_state, seq_reward = network.dynamics(afterstate, chance_code)
        batched = network.dynamics_batch([afterstate], [chance_code])

        assert len(batched) == 1
        bat_state, bat_reward = batched[0]
        assert_allclose(bat_state, seq_state, rtol=1e-5)
        assert_allclose(bat_reward, seq_reward, rtol=1e-5)


class TestVirtualLoss:
    """Tests for virtual loss mechanism."""

    def test_apply_virtual_loss(self):
        """Virtual loss increments correctly."""
        state = np.zeros(STATE_SIZE)
        node = DecisionNode(hidden_state=state)
        path = [node]

        assert node.virtual_loss == 0
        apply_virtual_loss(path)
        assert node.virtual_loss == 1
        apply_virtual_loss(path)
        assert node.virtual_loss == 2

    def test_remove_virtual_loss(self):
        """Virtual loss decrements correctly."""
        state = np.zeros(STATE_SIZE)
        node = DecisionNode(hidden_state=state)
        path = [node]

        node.virtual_loss = 3
        remove_virtual_loss(path)
        assert node.virtual_loss == 2
        remove_virtual_loss(path)
        assert node.virtual_loss == 1

    def test_effective_visit_count(self):
        """Effective visit count includes virtual losses."""
        state = np.zeros(STATE_SIZE)
        node = DecisionNode(hidden_state=state)

        node.visit_count = 5
        node.virtual_loss = 3
        assert node.effective_visit_count == 8

    def test_q_value_with_virtual_loss(self):
        """Q-value calculation accounts for virtual losses as zero-value visits."""
        state = np.zeros(STATE_SIZE)
        node = DecisionNode(hidden_state=state)

        node.visit_count = 4
        node.value_sum = 4.0  # Average of 1.0
        node.virtual_loss = 4  # 4 virtual losses (value = 0)

        # ##>: 8 total visits, value_sum = 4.0, so Q = 4/8 = 0.5.
        assert_allclose(node.q_value, 0.5)


class TestMinMaxStats:
    """Tests for thread-safe min/max tracking."""

    def test_minmax_initial_state(self):
        """Initial min/max values are inf/-inf."""
        stats = MinMaxStats()
        assert stats.minimum == float('inf')
        assert stats.maximum == float('-inf')

    def test_minmax_update(self):
        """Update correctly tracks min and max."""
        stats = MinMaxStats()
        stats.update(5.0)
        stats.update(2.0)
        stats.update(8.0)

        assert stats.minimum == 2.0
        assert stats.maximum == 8.0

    def test_minmax_normalize(self):
        """Normalization scales to [0, 1]."""
        stats = MinMaxStats()
        stats.update(0.0)
        stats.update(10.0)

        assert stats.normalize(0.0) == 0.0
        assert stats.normalize(10.0) == 1.0
        assert stats.normalize(5.0) == 0.5


class TestBatchedMCTS:
    """Tests for Level 2: Leaf batching MCTS."""

    def test_batched_mcts_returns_root(self, network, sample_state):
        """Batched MCTS returns a valid root node."""
        root = run_batched_mcts(game_state=sample_state, network=network, num_simulations=10, batch_size=4)

        assert isinstance(root, DecisionNode)
        assert root.expanded
        assert len(root.children) > 0

    def test_batched_mcts_visit_counts(self, network, sample_state):
        """Visit counts sum to approximately num_simulations."""
        num_simulations = 20
        root = run_batched_mcts(
            game_state=sample_state, network=network, num_simulations=num_simulations, batch_size=4
        )

        total_child_visits = sum(child.visit_count for child in root.children.values())
        # ##>: Total visits should be close to num_simulations (may differ slightly due to batching).
        assert total_child_visits >= num_simulations * 0.5
        assert total_child_visits <= num_simulations * 1.5

    def test_batched_mcts_terminal_state(self, network):
        """Terminal state returns root without children."""
        # ##>: Full board = terminal state.
        terminal_state = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [2, 4, 8, 16]])

        root = run_batched_mcts(game_state=terminal_state, network=network, num_simulations=10, batch_size=4)

        assert root.is_terminal

    def test_batched_mcts_no_exploration_noise(self, network, sample_state):
        """Batched MCTS works without exploration noise."""
        root = run_batched_mcts(
            game_state=sample_state, network=network, num_simulations=10, batch_size=4, add_exploration_noise=False
        )

        assert root.expanded
        assert len(root.children) > 0


class TestThreadedMCTS:
    """Tests for Level 3: Threaded MCTS with virtual loss."""

    def test_threaded_mcts_returns_root(self, network, sample_state):
        """Threaded MCTS returns a valid root node."""
        root = run_threaded_mcts(
            game_state=sample_state, network=network, num_simulations=10, num_workers=2, batch_size=4
        )

        assert isinstance(root, DecisionNode)
        assert root.expanded
        assert len(root.children) > 0

    def test_threaded_mcts_visit_counts(self, network, sample_state):
        """Threaded MCTS produces visit counts."""
        root = run_threaded_mcts(
            game_state=sample_state, network=network, num_simulations=20, num_workers=2, batch_size=4
        )

        total_child_visits = sum(child.visit_count for child in root.children.values())
        assert total_child_visits > 0

    def test_threaded_mcts_terminal_state(self, network):
        """Threaded MCTS handles terminal states."""
        terminal_state = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [2, 4, 8, 16]])

        root = run_threaded_mcts(
            game_state=terminal_state, network=network, num_simulations=10, num_workers=2, batch_size=4
        )

        assert root.is_terminal


class TestBatchedAgent:
    """Tests for the BatchedMuZeroAgent wrapper."""

    def test_agent_sequential_mode(self, sample_state):
        """Agent works in sequential mode."""
        agent = BatchedMuZeroAgent.create_untrained(
            observation_shape=OBSERVATION_SHAPE,
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=10,
            mode=MCTSMode.SEQUENTIAL,
        )

        action = agent.choose_action(sample_state)
        assert 0 <= action <= 3

    def test_agent_batched_mode(self, sample_state):
        """Agent works in batched mode."""
        agent = BatchedMuZeroAgent.create_untrained(
            observation_shape=OBSERVATION_SHAPE,
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=10,
            mode=MCTSMode.BATCHED,
            batch_size=4,
        )

        action = agent.choose_action(sample_state)
        assert 0 <= action <= 3

    def test_agent_threaded_mode(self, sample_state):
        """Agent works in threaded mode."""
        agent = BatchedMuZeroAgent.create_untrained(
            observation_shape=OBSERVATION_SHAPE,
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=10,
            mode=MCTSMode.THREADED,
            num_workers=2,
            batch_size=4,
        )

        action = agent.choose_action(sample_state)
        assert 0 <= action <= 3

    def test_agent_mode_switching(self, sample_state):
        """Agent can switch modes dynamically."""
        agent = BatchedMuZeroAgent.create_untrained(
            observation_shape=OBSERVATION_SHAPE,
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=5,
            mode=MCTSMode.SEQUENTIAL,
        )

        agent.set_mode(MCTSMode.BATCHED)
        action = agent.choose_action(sample_state)
        assert 0 <= action <= 3

        agent.set_mode(MCTSMode.THREADED)
        action = agent.choose_action(sample_state)
        assert 0 <= action <= 3

    def test_agent_training_mode(self, _sample_state):
        """Training mode enables noise and temperature."""
        agent = BatchedMuZeroAgent.create_untrained(
            observation_shape=OBSERVATION_SHAPE,
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=5,
            mode=MCTSMode.BATCHED,
        )

        agent.set_training_mode(True)
        assert agent.temperature == 1.0
        assert agent.add_noise is True

        agent.set_training_mode(False)
        assert agent.temperature == 0.0
        assert agent.add_noise is False


class TestPolicyExtraction:
    """Tests for policy extraction from visit counts."""

    def test_policy_from_visits_temperature_zero(self):
        """Temperature 0 gives argmax selection."""
        state = np.zeros(STATE_SIZE)
        root = DecisionNode(hidden_state=state)

        # ##>: Create children with different visit counts.
        for action in range(4):
            child = ChanceNode(afterstate=np.zeros(STATE_SIZE), action=action, parent=root)
            child.visit_count = action + 1  # Action 3 has most visits.
            root.children[action] = child

        policy = get_policy_from_visits(root, temperature=0.0)

        assert policy[3] == 1.0
        assert policy[0] == 0.0
        assert policy[1] == 0.0
        assert policy[2] == 0.0

    def test_policy_from_visits_temperature_one(self):
        """Temperature 1 gives proportional distribution."""
        state = np.zeros(STATE_SIZE)
        root = DecisionNode(hidden_state=state)

        for action in range(4):
            child = ChanceNode(afterstate=np.zeros(STATE_SIZE), action=action, parent=root)
            child.visit_count = 10  # Equal visits.
            root.children[action] = child

        policy = get_policy_from_visits(root, temperature=1.0)

        # ##>: With equal visits and temp=1, should be roughly uniform.
        for action in range(4):
            assert_allclose(policy[action], 0.25, rtol=0.1)

    def test_select_action_from_root(self):
        """Action selection works correctly."""
        state = np.zeros(STATE_SIZE)
        root = DecisionNode(hidden_state=state)

        for action in range(4):
            child = ChanceNode(afterstate=np.zeros(STATE_SIZE), action=action, parent=root)
            child.visit_count = action + 1
            child.value_sum = float(action + 1)
            root.children[action] = child

        # ##>: With temperature 0, should select action with most visits (action 3).
        selected = select_action_from_root(root, temperature=0.0)
        assert selected == 3
