"""
Tests for parallel MCTS with batched neural network inference.

This module tests:
- Batched network methods
- Parallel MCTS search
- Integration between batched inference and search
- Accelerator detection
"""

import numpy as np
import pytest

from reinforce.mcts.parallel_search import (
    ChanceNode,
    DecisionNode,
    ExpansionRequest,
    ExpansionType,
    ParallelMCTS,
    Trajectory,
    run_parallel_mcts,
)
from reinforce.mcts.stochastic_agent import StochasticMuZeroAgent
from reinforce.neural.accelerator import AcceleratorStrategy, AcceleratorType
from reinforce.neural.network import BatchedNetworkOutput, create_stochastic_network
from twentyfortyeight.core.gameboard import fill_cells

# ##>: Test constants - smaller values for faster tests.
STATE_SIZE = 64
CODEBOOK_SIZE = 8


@pytest.fixture
def network():
    """Create a small StochasticNetwork for testing."""
    return create_stochastic_network(observation_shape=(16,), hidden_size=STATE_SIZE, codebook_size=CODEBOOK_SIZE)


@pytest.fixture
def simple_game_state():
    """Create a deterministic game state for reproducible tests."""
    return np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]], dtype=np.int32)


@pytest.fixture
def game_state():
    """Create a simple game state for testing."""
    state = np.zeros((4, 4), dtype=np.int32)
    fill_cells(state, number_tile=2)
    return state


class TestBatchedNetworkMethods:
    """Tests for batched network inference methods."""

    def test_batch_representation(self, network):
        """Test batched representation encoding."""
        observations = [np.random.randn(16).astype(np.float32) for _ in range(4)]
        hidden_states = network.batch_representation(observations)

        assert hidden_states.shape == (4, STATE_SIZE)

    def test_batch_prediction(self, network):
        """Test batched policy and value prediction."""
        hidden_states = [np.random.randn(STATE_SIZE).astype(np.float32) for _ in range(4)]
        output = network.batch_prediction(hidden_states)

        assert isinstance(output, BatchedNetworkOutput)
        assert output.values.shape == (4,)
        assert output.policies.shape == (4, 4)

    def test_batch_afterstate_dynamics(self, network):
        """Test batched afterstate computation."""
        hidden_states = [np.random.randn(STATE_SIZE).astype(np.float32) for _ in range(4)]
        actions = [0, 1, 2, 3]

        afterstates = network.batch_afterstate_dynamics(hidden_states, actions)

        assert afterstates.shape == (4, STATE_SIZE)

    def test_batch_expand_all_actions(self, network):
        """Test computing all afterstates for multiple states."""
        hidden_states = [np.random.randn(STATE_SIZE).astype(np.float32) for _ in range(3)]

        all_afterstates = network.batch_expand_all_actions(hidden_states)

        # ##>: Shape should be (batch_size, num_actions, hidden_dim).
        assert all_afterstates.shape == (3, 4, STATE_SIZE)

    def test_batch_afterstate_prediction(self, network):
        """Test batched Q-value and chance distribution prediction."""
        afterstates = [np.random.randn(STATE_SIZE).astype(np.float32) for _ in range(4)]
        output = network.batch_afterstate_prediction(afterstates)

        assert output.values.shape == (4,)
        assert output.chance_probs.shape == (4, CODEBOOK_SIZE)

    def test_batch_dynamics(self, network):
        """Test batched dynamics computation."""
        afterstates = [np.random.randn(STATE_SIZE).astype(np.float32) for _ in range(4)]
        chance_codes = [np.eye(CODEBOOK_SIZE)[i] for i in range(4)]

        next_states, rewards = network.batch_dynamics(afterstates, chance_codes)

        assert next_states.shape == (4, STATE_SIZE)
        assert rewards.shape == (4,)


class TestParallelDecisionNode:
    """Tests for parallel MCTS DecisionNode."""

    def test_creation(self):
        """Test basic node creation."""
        hidden_state = np.random.randn(STATE_SIZE).astype(np.float32)
        node = DecisionNode(hidden_state=hidden_state)

        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert not node.expanded
        assert len(node.children) == 0

    def test_q_value_calculation(self):
        """Test Q-value calculation."""
        node = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        node.visit_count = 4
        node.value_sum = 10.0

        assert node.q_value == 2.5


class TestParallelChanceNode:
    """Tests for parallel MCTS ChanceNode."""

    def test_creation(self):
        """Test basic node creation."""
        afterstate = np.random.randn(STATE_SIZE).astype(np.float32)
        parent = DecisionNode(hidden_state=np.zeros(STATE_SIZE))

        node = ChanceNode(afterstate=afterstate, action=1, prior=0.25, parent=parent)

        assert node.action == 1
        assert node.prior == 0.25
        assert not node.expanded

    def test_expanded_property(self):
        """Test expanded property depends on chance_probs."""
        node = ChanceNode(afterstate=np.zeros(STATE_SIZE), action=0, prior=1.0)

        assert not node.expanded
        node.chance_probs = np.array([0.5, 0.5])
        assert node.expanded


class TestExpansionRequest:
    """Tests for ExpansionRequest dataclass."""

    def test_decision_request(self):
        """Test creating a decision expansion request."""
        node = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        request = ExpansionRequest(
            node=node,
            request_type=ExpansionType.DECISION,
            trajectory_idx=0,
            hidden_state=np.zeros(STATE_SIZE),
        )

        assert request.request_type == ExpansionType.DECISION
        assert request.trajectory_idx == 0

    def test_dynamics_request(self):
        """Test creating a dynamics expansion request."""
        parent = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        node = ChanceNode(afterstate=np.zeros(STATE_SIZE), action=0, prior=1.0, parent=parent)
        request = ExpansionRequest(
            node=node,
            request_type=ExpansionType.DYNAMICS,
            trajectory_idx=1,
            afterstate=np.zeros(STATE_SIZE),
            chance_code_idx=5,
        )

        assert request.request_type == ExpansionType.DYNAMICS
        assert request.chance_code_idx == 5


class TestTrajectory:
    """Tests for Trajectory dataclass."""

    def test_creation(self):
        """Test trajectory creation."""
        trajectory = Trajectory()

        assert len(trajectory.path) == 0
        assert trajectory.leaf_value is None
        assert not trajectory.needs_expansion


class TestParallelMCTS:
    """Tests for ParallelMCTS class."""

    def test_creation(self, network):
        """Test creating ParallelMCTS instance."""
        mcts = ParallelMCTS(
            network=network,
            num_simulations=10,
            batch_size=4,
        )

        assert mcts.num_simulations == 10
        assert mcts.batch_size == 4

    def test_basic_search(self, network, simple_game_state):
        """Test that parallel search completes without errors."""
        root = run_parallel_mcts(
            game_state=simple_game_state,
            network=network,
            num_simulations=10,
            batch_size=4,
            add_exploration_noise=False,
        )

        assert root is not None
        assert root.expanded
        assert root.visit_count > 0

    def test_search_with_noise(self, network, simple_game_state):
        """Test parallel search with Dirichlet noise."""
        root = run_parallel_mcts(
            game_state=simple_game_state,
            network=network,
            num_simulations=10,
            batch_size=4,
            add_exploration_noise=True,
            dirichlet_alpha=0.5,
            noise_fraction=0.25,
        )

        assert root is not None
        assert root.expanded

    def test_children_have_visits(self, network, simple_game_state):
        """Test that children accumulate visits."""
        root = run_parallel_mcts(
            game_state=simple_game_state,
            network=network,
            num_simulations=20,
            batch_size=4,
            add_exploration_noise=False,
        )

        total_child_visits = sum(child.visit_count for child in root.children.values())
        assert total_child_visits > 0

    def test_policy_from_visits(self, network, simple_game_state):
        """Test policy computation from visit counts."""
        mcts = ParallelMCTS(network=network, num_simulations=20, batch_size=4)
        root = mcts.search(simple_game_state)

        policy = mcts.get_policy(root, temperature=1.0)

        assert len(policy) > 0
        assert abs(sum(policy.values()) - 1.0) < 1e-6  # Should sum to 1

    def test_action_selection(self, network, simple_game_state):
        """Test action selection from search result."""
        mcts = ParallelMCTS(network=network, num_simulations=20, batch_size=4)
        root = mcts.search(simple_game_state)

        action = mcts.select_action(root, temperature=0.0)

        assert action in [0, 1, 2, 3]


class TestAgentParallelMode:
    """Tests for StochasticMuZeroAgent in parallel mode."""

    def test_create_with_parallel_mode(self):
        """Test creating agent in parallel mode."""
        agent = StochasticMuZeroAgent.create_untrained(
            observation_shape=(16,),
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=10,
            search_mode='parallel',
            batch_size=4,
        )

        assert agent.search_mode == 'parallel'
        assert agent._parallel_mcts is not None

    def test_choose_action_parallel(self, simple_game_state):
        """Test parallel action selection."""
        agent = StochasticMuZeroAgent.create_untrained(
            observation_shape=(16,),
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=10,
            search_mode='parallel',
            batch_size=4,
        )

        action = agent.choose_action(simple_game_state)

        assert action in [0, 1, 2, 3]

    def test_get_search_stats_parallel(self, simple_game_state):
        """Test search stats in parallel mode."""
        agent = StochasticMuZeroAgent.create_untrained(
            observation_shape=(16,),
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=10,
            search_mode='parallel',
            batch_size=4,
        )

        agent.choose_action(simple_game_state)
        stats = agent.get_search_stats()

        assert stats is not None
        assert len(stats.visit_counts) > 0
        assert len(stats.policy) > 0

    def test_switch_search_mode(self, simple_game_state):
        """Test switching between search modes."""
        agent = StochasticMuZeroAgent.create_untrained(
            observation_shape=(16,),
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=10,
            search_mode='sequential',
        )

        # ##>: Start in sequential mode.
        assert agent.search_mode == 'sequential'
        assert agent._parallel_mcts is None

        # ##>: Switch to parallel mode.
        agent.set_search_mode('parallel', batch_size=8)

        assert agent.search_mode == 'parallel'
        assert agent._parallel_mcts is not None
        assert agent.batch_size == 8

        # ##>: Should still work.
        action = agent.choose_action(simple_game_state)
        assert action in [0, 1, 2, 3]

        # ##>: Switch back to sequential.
        agent.set_search_mode('sequential')
        assert agent.search_mode == 'sequential'

    def test_training_mode_parallel(self):
        """Test training mode updates parallel MCTS."""
        agent = StochasticMuZeroAgent.create_untrained(
            num_simulations=10,
            search_mode='parallel',
        )

        agent.set_training_mode(True)
        assert agent._parallel_mcts.add_exploration_noise

        agent.set_training_mode(False)
        assert not agent._parallel_mcts.add_exploration_noise


class TestAcceleratorStrategy:
    """Tests for AcceleratorStrategy class."""

    def test_creation_cpu_fallback(self):
        """Test accelerator defaults to CPU when no GPU/TPU."""
        strategy = AcceleratorStrategy(force_cpu=True)

        assert strategy.accelerator == AcceleratorType.CPU
        assert strategy.info.device_count == 1

    def test_batch_size_multiplier_cpu(self):
        """Test batch size multiplier for CPU."""
        strategy = AcceleratorStrategy(force_cpu=True)

        assert strategy.batch_size_multiplier == 1

    def test_recommended_batch_size(self):
        """Test recommended batch size calculation."""
        strategy = AcceleratorStrategy(force_cpu=True)

        # ##>: CPU: base 16 * 1 = 16.
        assert strategy.recommended_batch_size == 16


class TestParallelVsSequentialConsistency:
    """Tests to verify parallel and sequential MCTS produce similar results."""

    def test_both_modes_find_valid_actions(self, simple_game_state):
        """Test that both modes return valid actions."""
        # ##>: Sequential mode.
        seq_agent = StochasticMuZeroAgent.create_untrained(
            observation_shape=(16,),
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=20,
            search_mode='sequential',
        )

        # ##>: Parallel mode.
        par_agent = StochasticMuZeroAgent.create_untrained(
            observation_shape=(16,),
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=20,
            search_mode='parallel',
            batch_size=4,
        )

        seq_action = seq_agent.choose_action(simple_game_state)
        par_action = par_agent.choose_action(simple_game_state)

        # ##>: Both should return valid actions.
        assert seq_action in [0, 1, 2, 3]
        assert par_action in [0, 1, 2, 3]

    def test_both_modes_complete_game(self, simple_game_state):
        """Test that both modes can complete a game."""
        from twentyfortyeight.core.gameboard import is_done, next_state

        for mode in ['sequential', 'parallel']:
            agent = StochasticMuZeroAgent.create_untrained(
                observation_shape=(16,),
                hidden_size=STATE_SIZE,
                codebook_size=CODEBOOK_SIZE,
                num_simulations=5,
                search_mode=mode,
                batch_size=4 if mode == 'parallel' else 16,
            )

            state = simple_game_state.copy()
            moves_made = 0

            for _ in range(10):
                if is_done(state):
                    break
                action = agent.choose_action(state)
                state, _ = next_state(state, action)
                moves_made += 1

            assert moves_made > 0, f'{mode} mode failed to make moves'
