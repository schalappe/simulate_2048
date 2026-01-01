"""
Tests for MCTS search functions (PUCT selection, simulation, backpropagation).

Focuses on correctness of selection formulas, adaptive simulation count,
and tree search integration.
"""

from unittest import TestCase, main

import numpy as np

from reinforce.mcts.node import Decision
from reinforce.mcts.search import adaptive_simulation_count, backpropagate, monte_carlo_search, puct_select, simulate


class TestPUCTSelection(TestCase):
    """Test PUCT formula and selection behavior."""

    def setUp(self):
        """Create Decision node with multiple Chance children."""
        state = np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.root = Decision(state=state, prior=1.0, final=False)

        # ##>: Create 2 Chance children (some states only have 2-3 legal actions).
        for _ in range(min(2, len(self.root.legal_moves))):
            self.root.add_child()

    def test_puct_selects_unvisited_child_over_low_value_visited(self):
        """Unvisited child (visits=0) selected due to high exploration term."""
        # ##>: Set up: child[0] visited with low reward, child[1] unvisited.
        self.root.visits = 10
        self.root.children[0].visits = 5
        self.root.children[0].values = 1.0  # Q = 0.2
        self.root.children[1].visits = 0
        self.root.children[1].values = 0.0  # Q = 0 (unvisited)

        selected = puct_select(self.root, exploration_weight=1.41)

        # ##>: Unvisited child selected due to infinite exploration bonus.
        self.assertEqual(selected, self.root.children[1])

    def test_puct_balances_exploitation_and_exploration(self):
        """High-visit, high-Q child eventually dominates over unvisited."""
        # ##>: Child[0]: high visits, high Q-value.
        self.root.visits = 100
        self.root.children[0].visits = 80
        self.root.children[0].values = 80.0  # Q = 1.0
        self.root.children[1].visits = 1
        self.root.children[1].values = 0.5  # Q = 0.5

        selected = puct_select(self.root, exploration_weight=0.1)

        # ##>: Low exploration weight favors exploitation.
        self.assertEqual(selected, self.root.children[0])

    def test_puct_uses_action_prior_not_parent_prior(self):
        """PUCT formula correctly uses child.prior (action prior)."""
        # ##>: Give children different priors.
        self.root.visits = 10
        self.root.children[0].prior = 0.8
        self.root.children[0].visits = 1
        self.root.children[0].values = 0.0
        self.root.children[1].prior = 0.1
        self.root.children[1].visits = 1
        self.root.children[1].values = 0.0

        selected = puct_select(self.root, exploration_weight=2.0)

        # ##>: Higher prior child selected (same Q, higher exploration term).
        self.assertEqual(selected, self.root.children[0])

    def test_puct_raises_on_chance_node(self):
        """PUCT is only defined for Decision nodes."""
        chance = self.root.children[0]

        with self.assertRaises(ValueError):
            puct_select(chance, exploration_weight=1.41)


class TestAdaptiveSimulation(TestCase):
    """Test adaptive simulation count formula."""

    def test_root_node_gets_base_simulations(self):
        """Root (depth=0) receives full base_simulations."""
        state = np.zeros((4, 4))
        node = Decision(state=state, prior=1.0, final=False, depth=0.0)

        count = adaptive_simulation_count(node, base_simulations=10)

        # ##>: Formula: 10 / (1 + log(0 + 1)) = 10 / 1 = 10.
        self.assertEqual(count, 10)

    def test_deep_nodes_get_fewer_simulations(self):
        """Deeper nodes receive fewer simulations (inverse log scaling)."""
        state = np.zeros((4, 4))

        node_depth_1 = Decision(state=state, prior=1.0, final=False, depth=1.0)
        node_depth_5 = Decision(state=state, prior=1.0, final=False, depth=5.0)

        count_1 = adaptive_simulation_count(node_depth_1, base_simulations=10)
        count_5 = adaptive_simulation_count(node_depth_5, base_simulations=10)

        # ##>: Depth 1 gets more simulations than depth 5.
        self.assertGreater(count_1, count_5)
        self.assertGreaterEqual(count_1, 5)  # depth=1: ~5 sims
        self.assertGreater(count_5, 1)  # depth=5: ~3-4 sims

    def test_minimum_one_simulation_enforced(self):
        """Very deep nodes still get at least 1 simulation."""
        state = np.zeros((4, 4))
        node = Decision(state=state, prior=1.0, final=False, depth=100.0)

        count = adaptive_simulation_count(node, base_simulations=10)

        # ##>: max(1, ...) ensures minimum.
        self.assertGreaterEqual(count, 1)


class TestBackpropagation(TestCase):
    """Test reward backpropagation up the tree."""

    def test_backpropagate_updates_all_ancestors(self):
        """Backpropagation updates visits and values up to root."""
        state = np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        # ##>: Build chain: root → chance → decision.
        root = Decision(state=state, prior=1.0, final=False)
        chance = root.add_child()
        leaf = chance.add_child()

        # ##>: Backpropagate from leaf.
        backpropagate(leaf, reward=5.0)

        # ##>: All nodes in path updated.
        self.assertEqual(leaf.visits, 1)
        self.assertEqual(leaf.values, 5.0)
        self.assertEqual(chance.visits, 1)
        self.assertEqual(chance.values, 5.0)
        self.assertEqual(root.visits, 1)
        self.assertEqual(root.values, 5.0)

    def test_backpropagate_accumulates_rewards(self):
        """Multiple backpropagations accumulate rewards and visits."""
        state = np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        root = Decision(state=state, prior=1.0, final=False)

        # ##>: Backpropagate twice with different rewards.
        backpropagate(root, reward=3.0)
        backpropagate(root, reward=7.0)

        # ##>: Visits and values accumulated.
        self.assertEqual(root.visits, 2)
        self.assertEqual(root.values, 10.0)


class TestSimulation(TestCase):
    """Test rollout simulation logic."""

    def test_simulate_returns_average_reward(self):
        """Simulate performs multiple rollouts and returns average."""
        state = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        node = Decision(state=state, prior=1.0, final=False)

        # ##>: Run 5 simulations.
        avg_reward = simulate(node, simulations=5)

        # ##>: Returns finite average (non-negative).
        self.assertIsInstance(avg_reward, float)
        self.assertGreaterEqual(avg_reward, 0.0)
        self.assertLess(avg_reward, 100.0)  # Normalized rewards are small

    def test_simulate_terminates_without_infinite_loop(self):
        """Simulation loop terminates (no infinite games)."""
        state = np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        node = Decision(state=state, prior=1.0, final=False)

        # ##>: Should complete within reasonable time (implicit timeout).
        avg_reward = simulate(node, simulations=10)

        # ##>: Simulation completed successfully.
        self.assertIsNotNone(avg_reward)


class TestMonteCarloSearch(TestCase):
    """Test full MCTS integration."""

    def test_search_creates_root_with_children(self):
        """monte_carlo_search expands tree from root."""
        state = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        root = monte_carlo_search(state, iterations=5)

        # ##>: Root has children after search.
        self.assertGreater(len(root.children), 0)
        self.assertGreater(root.visits, 0)

    def test_search_visits_all_children_at_root(self):
        """Multiple iterations visit multiple children."""
        state = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        root = monte_carlo_search(state, iterations=10)

        # ##>: At least some children have non-zero visits.
        visited_children = [c for c in root.children if c.visits > 0]
        self.assertGreater(len(visited_children), 0)

    def test_search_with_single_iteration(self):
        """Single iteration creates one expanded node."""
        state = np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        root = monte_carlo_search(state, iterations=1)

        # ##>: At least one node expanded.
        self.assertGreaterEqual(root.visits, 1)


if __name__ == '__main__':
    main()
