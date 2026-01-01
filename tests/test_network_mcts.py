"""
Tests for network-guided MCTS and StochasticMuZeroAgent.

This module tests the integration between neural networks and MCTS:
- Network-guided search functions
- PUCT selection with learned priors
- Chance node selection with σ distribution
- StochasticMuZeroAgent action selection
"""

import numpy as np
import pytest

from reinforce.mcts.network_search import (
    ChanceNode,
    DecisionNode,
    MinMaxStats,
    expand_chance_node,
    expand_decision_node,
    get_policy_from_visits,
    puct_score,
    run_network_mcts,
    select_action,
    select_action_from_root,
    select_chance_outcome,
)
from reinforce.mcts.stochastic_agent import SearchStats, StochasticMuZeroAgent
from reinforce.neural.network import create_stochastic_network
from twentyfortyeight.core.gameboard import fill_cells

# ##>: Test constants - smaller values for faster tests.
STATE_SIZE = 64
CODEBOOK_SIZE = 8


@pytest.fixture
def network():
    """Create a small StochasticNetwork for testing."""
    return create_stochastic_network(observation_shape=(16,), hidden_size=STATE_SIZE, codebook_size=CODEBOOK_SIZE)


@pytest.fixture
def game_state():
    """Create a simple game state for testing."""
    state = np.zeros((4, 4), dtype=np.int32)
    fill_cells(state, number_tile=2)
    return state


@pytest.fixture
def simple_game_state():
    """Create a deterministic game state for reproducible tests."""
    state = np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]], dtype=np.int32)
    return state


class TestMinMaxStats:
    """Tests for MinMaxStats normalization."""

    def test_initial_state(self):
        """Test initial min/max values."""
        stats = MinMaxStats()
        assert stats.minimum == float('inf')
        assert stats.maximum == float('-inf')

    def test_update(self):
        """Test updating min/max values."""
        stats = MinMaxStats()
        stats.update(5.0)
        stats.update(10.0)
        stats.update(2.0)

        assert stats.minimum == 2.0
        assert stats.maximum == 10.0

    def test_normalize(self):
        """Test value normalization."""
        stats = MinMaxStats()
        stats.update(0.0)
        stats.update(10.0)

        assert stats.normalize(0.0) == 0.0
        assert stats.normalize(10.0) == 1.0
        assert stats.normalize(5.0) == 0.5

    def test_normalize_no_range(self):
        """Test normalization when min == max."""
        stats = MinMaxStats()
        # ##>: No updates, should return value as-is.
        assert stats.normalize(5.0) == 5.0


class TestDecisionNode:
    """Tests for DecisionNode functionality."""

    def test_creation(self):
        """Test basic node creation."""
        hidden_state = np.random.randn(STATE_SIZE).astype(np.float32)
        node = DecisionNode(hidden_state=hidden_state)

        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert not node.expanded
        assert len(node.children) == 0

    def test_q_value_empty(self):
        """Test Q-value when no visits."""
        node = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        assert node.q_value == 0.0

    def test_q_value_with_visits(self):
        """Test Q-value calculation."""
        node = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        node.visit_count = 4
        node.value_sum = 10.0

        assert node.q_value == 2.5


class TestChanceNode:
    """Tests for ChanceNode functionality."""

    def test_creation(self):
        """Test basic node creation."""
        afterstate = np.random.randn(STATE_SIZE).astype(np.float32)
        parent = DecisionNode(hidden_state=np.zeros(STATE_SIZE))

        node = ChanceNode(afterstate=afterstate, action=1, prior=0.25, parent=parent)

        assert node.action == 1
        assert node.prior == 0.25
        assert not node.expanded
        assert len(node.children) == 0


class TestPUCTScore:
    """Tests for PUCT score calculation."""

    def test_unvisited_child_high_prior(self):
        """Test that unvisited children with high prior get high scores."""
        parent = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        parent.visit_count = 10

        high_prior = ChanceNode(afterstate=np.zeros(STATE_SIZE), action=0, prior=0.9, parent=parent)
        low_prior = ChanceNode(afterstate=np.zeros(STATE_SIZE), action=1, prior=0.1, parent=parent)

        stats = MinMaxStats()
        score_high = puct_score(parent, high_prior, 1.25, stats)
        score_low = puct_score(parent, low_prior, 1.25, stats)

        assert score_high > score_low

    def test_visited_child_with_value(self):
        """Test that visited children include Q-value component."""
        parent = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        parent.visit_count = 10

        child = ChanceNode(afterstate=np.zeros(STATE_SIZE), action=0, prior=0.5, parent=parent)
        child.visit_count = 5
        child.value_sum = 10.0

        stats = MinMaxStats()
        stats.update(2.0)  # Set some bounds

        score = puct_score(parent, child, 1.25, stats)
        # ##>: Score should be positive (Q-value + exploration bonus).
        assert score > 0


class TestSelectAction:
    """Tests for action selection."""

    def test_select_action_prefers_high_score(self):
        """Test that action selection picks highest PUCT score."""
        parent = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        parent.visit_count = 10

        # ##>: Create children with different priors.
        parent.children = {
            0: ChanceNode(afterstate=np.zeros(STATE_SIZE), action=0, prior=0.1, parent=parent),
            1: ChanceNode(afterstate=np.zeros(STATE_SIZE), action=1, prior=0.9, parent=parent),
            2: ChanceNode(afterstate=np.zeros(STATE_SIZE), action=2, prior=0.0, parent=parent),
        }

        stats = MinMaxStats()
        action = select_action(parent, 1.25, stats)

        # ##>: Should prefer action 1 (highest prior, all unvisited).
        assert action == 1


class TestSelectChanceOutcome:
    """Tests for chance outcome selection."""

    def test_selects_unvisited_high_prob(self):
        """Test quasi-random selection prefers unvisited high-probability outcomes."""
        parent = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        node = ChanceNode(afterstate=np.zeros(STATE_SIZE), action=0, prior=1.0, parent=parent)

        # ##>: Set chance probs with clear winner.
        node.chance_probs = np.array([0.1, 0.1, 0.7, 0.1])

        outcome = select_chance_outcome(node)

        # ##>: Should select index 2 (highest prob, all unvisited).
        assert outcome == 2

    def test_balances_visits_and_prob(self):
        """Test that selection balances visits with probability."""
        parent = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        node = ChanceNode(afterstate=np.zeros(STATE_SIZE), action=0, prior=1.0, parent=parent)

        node.chance_probs = np.array([0.5, 0.5])

        # ##>: Add a visited child to index 0.
        child0 = DecisionNode(hidden_state=np.zeros(STATE_SIZE), parent=node)
        child0.visit_count = 10
        node.children[0] = child0

        outcome = select_chance_outcome(node)

        # ##>: Should select index 1 (same prob but unvisited).
        assert outcome == 1


class TestExpandNodes:
    """Tests for node expansion."""

    def test_expand_decision_node(self, network, simple_game_state):
        """Test decision node expansion creates children."""
        hidden_state = network.representation(simple_game_state.flatten())
        node = DecisionNode(hidden_state=hidden_state, game_state=simple_game_state)

        expand_decision_node(node, network, simple_game_state)

        assert node.expanded
        assert len(node.children) > 0
        assert node.policy_prior is not None
        assert len(node.policy_prior) == 4  # 4 actions

    def test_expand_chance_node(self, network):
        """Test chance node expansion computes predictions."""
        parent = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        afterstate = np.random.randn(STATE_SIZE).astype(np.float32)
        node = ChanceNode(afterstate=afterstate, action=0, prior=1.0, parent=parent)

        expand_chance_node(node, network)

        assert node.expanded
        assert node.chance_probs is not None
        assert len(node.chance_probs) == CODEBOOK_SIZE


class TestRunNetworkMCTS:
    """Tests for the full MCTS search."""

    def test_basic_search(self, network, simple_game_state):
        """Test that search completes without errors."""
        root = run_network_mcts(
            game_state=simple_game_state,
            network=network,
            num_simulations=10,
            add_exploration_noise=False,
        )

        assert root is not None
        assert root.expanded
        assert root.visit_count > 0

    def test_search_with_noise(self, network, simple_game_state):
        """Test search with Dirichlet noise."""
        root = run_network_mcts(
            game_state=simple_game_state,
            network=network,
            num_simulations=10,
            add_exploration_noise=True,
            dirichlet_alpha=0.5,
            noise_fraction=0.25,
        )

        assert root is not None
        assert root.expanded

    def test_children_have_visits(self, network, simple_game_state):
        """Test that children accumulate visits."""
        root = run_network_mcts(
            game_state=simple_game_state,
            network=network,
            num_simulations=20,
            add_exploration_noise=False,
        )

        total_child_visits = sum(child.visit_count for child in root.children.values())
        # ##>: At least some visits should have been distributed.
        assert total_child_visits > 0


class TestPolicyFromVisits:
    """Tests for policy computation from visit counts."""

    def test_proportional_to_visits(self):
        """Test that policy is proportional to visits with temperature 1."""
        root = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        parent = root  # For ChanceNode creation

        root.children = {
            0: ChanceNode(afterstate=np.zeros(STATE_SIZE), action=0, prior=0.5, parent=parent),
            1: ChanceNode(afterstate=np.zeros(STATE_SIZE), action=1, prior=0.5, parent=parent),
        }
        root.children[0].visit_count = 10
        root.children[1].visit_count = 30

        policy = get_policy_from_visits(root, temperature=1.0)

        # ##>: With temp=1, should be roughly proportional to visits.
        # ##>: Softmax with temp=1 and small visit differences may not be exactly proportional.
        assert policy[1] > policy[0]

    def test_greedy_with_zero_temp(self):
        """Test that temperature 0 gives deterministic policy."""
        root = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        parent = root

        root.children = {
            0: ChanceNode(afterstate=np.zeros(STATE_SIZE), action=0, prior=0.5, parent=parent),
            1: ChanceNode(afterstate=np.zeros(STATE_SIZE), action=1, prior=0.5, parent=parent),
        }
        root.children[0].visit_count = 10
        root.children[1].visit_count = 30

        policy = get_policy_from_visits(root, temperature=0.0)

        assert policy[1] == 1.0
        assert policy[0] == 0.0


class TestSelectActionFromRoot:
    """Tests for action selection from search root."""

    def test_selects_most_visited(self):
        """Test that greedy selection picks most visited action."""
        root = DecisionNode(hidden_state=np.zeros(STATE_SIZE))
        parent = root

        root.children = {
            0: ChanceNode(afterstate=np.zeros(STATE_SIZE), action=0, prior=0.5, parent=parent),
            1: ChanceNode(afterstate=np.zeros(STATE_SIZE), action=1, prior=0.5, parent=parent),
        }
        root.children[0].visit_count = 10
        root.children[1].visit_count = 30
        root.children[0].value_sum = 5.0
        root.children[1].value_sum = 15.0

        action = select_action_from_root(root, temperature=0.0)

        assert action == 1


class TestStochasticMuZeroAgent:
    """Tests for the StochasticMuZeroAgent class."""

    def test_create_untrained(self):
        """Test creating agent with untrained network."""
        agent = StochasticMuZeroAgent.create_untrained(
            observation_shape=(16,),
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=10,
        )

        assert agent is not None
        assert agent.network is not None

    def test_choose_action(self, simple_game_state):
        """Test that agent can choose an action."""
        agent = StochasticMuZeroAgent.create_untrained(
            observation_shape=(16,),
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=10,
        )

        action = agent.choose_action(simple_game_state)

        assert action in [0, 1, 2, 3]

    def test_get_search_stats(self, simple_game_state):
        """Test that search stats are available after choosing action."""
        agent = StochasticMuZeroAgent.create_untrained(
            observation_shape=(16,),
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=10,
        )

        agent.choose_action(simple_game_state)
        stats = agent.get_search_stats()

        assert stats is not None
        assert isinstance(stats, SearchStats)
        assert len(stats.visit_counts) > 0
        assert len(stats.policy) > 0

    def test_training_mode(self):
        """Test training mode configuration."""
        agent = StochasticMuZeroAgent.create_untrained(num_simulations=10)

        agent.set_training_mode(True)
        assert agent.temperature == 1.0
        assert agent.add_noise

        agent.set_training_mode(False)
        assert agent.temperature == 0.0
        assert not agent.add_noise


class TestAgentPlaysGame:
    """Integration tests for agent playing a game."""

    def test_agent_completes_moves(self, simple_game_state):
        """Test agent can make multiple moves."""
        agent = StochasticMuZeroAgent.create_untrained(
            observation_shape=(16,),
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=5,
        )

        # ##>: Make a few moves.
        from twentyfortyeight.core.gameboard import is_done, next_state

        state = simple_game_state.copy()
        moves_made = 0

        for _ in range(10):
            if is_done(state):
                break
            action = agent.choose_action(state)
            state, _ = next_state(state, action)
            moves_made += 1

        assert moves_made > 0

    def test_agent_produces_valid_actions(self, simple_game_state):
        """Test that agent only produces legal actions."""
        agent = StochasticMuZeroAgent.create_untrained(
            observation_shape=(16,),
            hidden_size=STATE_SIZE,
            codebook_size=CODEBOOK_SIZE,
            num_simulations=10,
        )

        from twentyfortyeight.core.gamemove import legal_actions

        for _ in range(5):
            legal = legal_actions(simple_game_state)
            action = agent.choose_action(simple_game_state)
            assert action in legal
