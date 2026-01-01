"""
Tests for MCTS node structures (Decision and Chance nodes).

Focuses on node creation, parent-child relationships, progressive widening,
and lazy outcome generation.
"""

from unittest import TestCase, main

import numpy as np

from reinforce.mcts.node import Chance, Decision


class TestDecisionNode(TestCase):
    """Test Decision node behavior and expansion logic."""

    def setUp(self):
        """Create a sample board state for testing."""
        # ##>: Simple board with tiles in top-left, rest empty.
        self.state = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_decision_node_initialization(self):
        """Decision node initializes legal moves from game state."""
        node = Decision(state=self.state, prior=1.0, final=False)

        # ##>: Non-terminal node discovers legal moves (all 4 directions valid).
        self.assertEqual(len(node.legal_moves), 4)
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.values, 0.0)
        self.assertFalse(node.final)

    def test_decision_terminal_state(self):
        """Terminal Decision node skips legal moves discovery."""
        node = Decision(state=self.state, prior=1.0, final=True)

        # ##>: Terminal node has no legal moves.
        self.assertEqual(len(node.legal_moves), 0)
        self.assertTrue(node.final)
        self.assertTrue(node.fully_expanded())

    def test_decision_add_child_creates_chance_node(self):
        """Adding child to Decision creates Chance node with uniform prior."""
        node = Decision(state=self.state, prior=1.0, final=False)
        child = node.add_child()

        # ##>: Child is Chance node with correct parent relationship.
        self.assertIsInstance(child, Chance)
        self.assertEqual(child.parent, node)
        self.assertEqual(len(node.children), 1)

        # ##>: Uniform prior = 1/num_actions.
        expected_prior = 1.0 / len(node.legal_moves)
        self.assertAlmostEqual(child.prior, expected_prior)

    def test_decision_fully_expanded_after_all_actions_tried(self):
        """Decision node fully expanded when all legal moves have children."""
        node = Decision(state=self.state, prior=1.0, final=False)
        num_legal = len(node.legal_moves)

        # ##>: Not fully expanded initially.
        self.assertFalse(node.fully_expanded())

        # ##>: Add children for all legal moves.
        for _ in range(num_legal):
            node.add_child()

        # ##>: Fully expanded after all actions tried.
        self.assertTrue(node.fully_expanded())
        self.assertEqual(len(node.children), num_legal)

    def test_decision_add_child_raises_when_fully_expanded(self):
        """Cannot add child to fully expanded Decision node."""
        node = Decision(state=self.state, prior=1.0, final=False)

        # ##>: Expand all legal moves.
        for _ in range(len(node.legal_moves)):
            node.add_child()

        # ##>: Adding another child raises ValueError.
        with self.assertRaises(ValueError):
            node.add_child()


class TestChanceNode(TestCase):
    """Test Chance node behavior, progressive widening, and lazy generation."""

    def setUp(self):
        """Create a parent Decision and sample Chance node."""
        # ##>: Board after left move (latent state, no tile added yet).
        self.state = np.array([[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.parent = Decision(state=np.zeros((4, 4)), prior=1.0, final=False)

    def test_chance_node_initialization(self):
        """Chance node initializes lazy generation fields."""
        node = Chance(state=self.state, parent=self.parent, action=0, depth=0.5)

        # ##>: Lazy generation fields populated.
        self.assertEqual(node._num_empty, 15)  # 1 tile placed, 15 empty
        self.assertEqual(len(node._empty_cells), 15)
        self.assertEqual(len(node._unvisited_indices), 30)  # 15 cells × 2 values
        self.assertEqual(node.max_outcomes, 30)

    def test_chance_full_board_no_outcomes(self):
        """Chance node with full board has zero max_outcomes."""
        # ##>: Full board (all cells filled).
        full_board = np.full((4, 4), 2)
        node = Chance(state=full_board, parent=self.parent, action=0, depth=0.5)

        # ##>: No empty cells means no stochastic outcomes.
        self.assertEqual(node._num_empty, 0)
        self.assertEqual(node.max_outcomes, 0)
        self.assertTrue(node.fully_expanded())

    def test_chance_add_child_creates_decision_node(self):
        """Adding child to Chance creates Decision node with stochastic outcome."""
        node = Chance(state=self.state, parent=self.parent, action=0, depth=0.5)
        child = node.add_child()

        # ##>: Child is Decision node with correct parent relationship.
        self.assertIsInstance(child, Decision)
        self.assertEqual(child.parent, node)
        self.assertEqual(len(node.children), 1)

        # ##>: One index removed from unvisited set.
        self.assertEqual(len(node._unvisited_indices), 29)

    def test_chance_progressive_widening_threshold(self):
        """Progressive widening limits children based on visit count."""
        node = Chance(state=self.state, parent=self.parent, action=0, depth=0.5)

        # ##>: visits=1: threshold = min(30, 1.0 * sqrt(1)) = 1.
        node.visits = 1
        self.assertFalse(node.fully_expanded())
        node.add_child()
        self.assertTrue(node.fully_expanded())

        # ##>: visits=4: threshold = min(30, 1.0 * sqrt(4)) = 2.
        node.visits = 4
        self.assertFalse(node.fully_expanded())
        node.add_child()
        self.assertTrue(node.fully_expanded())

        # ##>: visits=9: threshold = min(30, 1.0 * sqrt(9)) = 3.
        node.visits = 9
        self.assertFalse(node.fully_expanded())

    def test_chance_full_board_creates_terminal_child(self):
        """Chance node with no empty cells creates final Decision node."""
        full_board = np.full((4, 4), 2)
        node = Chance(state=full_board, parent=self.parent, action=0, depth=0.5)

        child = node.add_child()

        # ##>: Child is terminal Decision node.
        self.assertTrue(child.final)
        self.assertEqual(len(child.legal_moves), 0)

    def test_chance_add_child_raises_when_all_outcomes_generated(self):
        """Cannot add child when all outcomes have been generated."""
        # ##>: Small board with only 1 empty cell (2 possible outcomes).
        state = np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 0]])
        node = Chance(state=state, parent=self.parent, action=0, depth=0.5)

        # ##>: Generate both outcomes (2 and 4 in last cell).
        node.add_child()
        node.add_child()

        # ##>: All outcomes exhausted.
        self.assertEqual(len(node._unvisited_indices), 0)
        with self.assertRaises(ValueError):
            node.add_child()


class TestNodeRelationships(TestCase):
    """Test parent-child relationships between Decision and Chance nodes."""

    def test_decision_to_chance_to_decision_chain(self):
        """Verify Decision → Chance → Decision parent-child chain."""
        state = np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        # ##>: Create root Decision node.
        root = Decision(state=state, prior=1.0, final=False)
        self.assertIsNone(root.parent)

        # ##>: Add Chance child.
        chance_child = root.add_child()
        self.assertEqual(chance_child.parent, root)
        self.assertIn(chance_child, root.children)

        # ##>: Add Decision grandchild.
        decision_grandchild = chance_child.add_child()
        self.assertEqual(decision_grandchild.parent, chance_child)
        self.assertIn(decision_grandchild, chance_child.children)

    def test_depth_increments_by_half(self):
        """Each node addition increments depth by 0.5."""
        state = np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        root = Decision(state=state, prior=1.0, final=False, depth=0.0)
        chance_child = root.add_child()
        decision_grandchild = chance_child.add_child()

        # ##>: Depth progression: 0 → 0.5 → 1.0.
        self.assertEqual(root.depth, 0.0)
        self.assertEqual(chance_child.depth, 0.5)
        self.assertEqual(decision_grandchild.depth, 1.0)


if __name__ == '__main__':
    main()
