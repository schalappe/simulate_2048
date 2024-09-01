# -*- coding: utf-8 -*-
"""
List of functions for Monte Carlo Tree Search.
"""
from numpy import log, ndarray, sqrt
from numpy.random import PCG64DXSM, default_rng

from simulate.utils import is_done, legal_actions, next_state

from .node import Chance, Decision, Node

GENERATOR = default_rng(PCG64DXSM())


def uct_select(node: Decision, exploration_weight: float) -> Chance:
    """
    Select a child node using the UCT formula.

    Parameters
    ----------
    node : Decision
        The parent node from which to select a child.
    exploration_weight : float
        The exploration weight parameter in the UCB1 formula.

    Returns
    -------
    float
        The calculated UCB1 score.

    Notes
    -----
    Uses the UCB1 formula: exploitation + exploration_weight * sqrt(log(parent_visits) / child_visits)
    """
    if isinstance(node, Chance):
        raise ValueError("UCB1 is only defined for Decision nodes.")

    log_visits = log(node.visits)
    return max(
        node.children,
        key=lambda child: child.values / child.visits + exploration_weight * sqrt(log_visits / child.visits),
    )


def puct_select(node: Decision, exploration_weight: float) -> Chance:
    """
    Select a child node using the PUCT formula.

    Parameters
    ----------
    node : Decision
        The parent node from which to select a child.
    exploration_weight : float
        The exploration weight parameter in the PUCT formula.

    Returns
    -------
    Chance
        The selected child node.

    Notes
    -----
    Uses the PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    where:
    - Q(s,a) is the average value of the child node
    - P(s,a) is the prior probability of selecting the action
    - N(s) is the visit count of the parent node
    - N(s,a) is the visit count of the child node
    - c_puct is the exploration weight
    """
    if isinstance(node, Chance):
        raise ValueError("PUCT is only defined for Decision nodes.")

    def puct_score(child: Chance) -> float:
        q_value = child.values / child.visits if child.visits > 0 else 0
        return q_value + exploration_weight * child.parent.prior * sqrt(node.visits) / (1 + child.visits)

    return max(node.children, key=puct_score)


def select_child(node: Node, exploration_weight: float) -> Node:
    """
    Select a child node for expansion using the UCB1 algorithm with progressive widening.

    This function is a key part of the selection phase in Monte Carlo Tree Search. It chooses the most promising
    child node based on the UCB1 scores.

    Parameters
    ----------
    node : Node
        The current node from which to select a child.
    exploration_weight : float
        The exploration weight for the UCB1 calculation.

    Returns
    -------
    Node
        The selected child node.
    """
    while node.children:
        if not node.fully_expanded():
            return node
        if isinstance(node, Decision):
            node = puct_select(node, exploration_weight)
        else:
            all_visits = sum(child.visits for child in node.children)
            node = GENERATOR.choice(node.children, p=[child.visits / all_visits for child in node.children])
    return node


def simulate(node: Node) -> float:
    """
    Perform a rollout simulation from the given node.

    This function is part of the simulation phase in Monte Carlo Tree Search. It plays out the game from the
    given state using random moves until the game ends.

    Parameters
    ----------
    node : Node
        The starting node for the simulation.

    Returns
    -------
    float
        The total reward (score) from the simulation.

    Notes
    -----
    - The simulation uses random legal moves at each step.
    - The simulation continues until no legal moves are available or the game is done.
    - The total reward is the sum of rewards from each move in the simulation.
    """
    total_reward = 0.0

    # ##: Initialize the state.
    if isinstance(node, Chance):
        states, priors = zip(*node.next_states)
        state = GENERATOR.choice(states, p=priors)
    else:
        state = node.state.copy()

    # ##: Simulate until done.
    while not is_done(state):
        action = GENERATOR.choice(legal_actions(state))
        state, reward = next_state(state, action)
        total_reward += reward

    return total_reward


def backpropagate(node: Node, reward: float) -> None:
    """
    Back-propagate the reward through the tree.

    This function is part of the backpropagation phase in Monte Carlo Tree Search. It updates the visit count
    and value of each node in the path from the simulated leaf node to the root.

    Parameters
    ----------
    node : Node
        The starting node for backpropagation (typically a leaf node).
    reward : float
        The reward value to back-propagate.

    Notes
    -----
    - The function updates each node's visit count and cumulative value.
    - It continues updating until it reaches the root node (node with no parent).
    """
    while node is not None:
        node.update(reward)
        node = node.parent


def monte_carlo_search(state: ndarray, iterations: int, exploration_weight: float = 1.41) -> Decision:
    """
    Perform Monte Carlo Tree Search with Progressive Widening from the given root node.

    This function implements the main loop of the Monte Carlo Tree Search algorithm. It repeatedly selects, expands,
    simulates, and back-propagates for a specified  number of iterations.

    Parameters
    ----------
    state : np.ndarray
        The initial game state.
    iterations : int
        The number of iterations to perform the search.
    exploration_weight : float, optional
        The exploration weight for the UCB1 formula (default is 1.41).

    Returns
    -------
    Decision
        The root node of the search tree.

    Notes
    -----
    - Each iteration consists of four phases: selection, expansion, simulation, and backpropagation.
    - The search builds a tree of possible game states and actions, estimating their values.
    - Progressive Widening is applied to manage the branching factor in outcome spaces.
    - After the search, the root's children represent the possible next moves, with visit counts
      indicating their estimated strength.
    """
    root = Decision(state=state, final=False, prior=0.0)
    for _ in range(iterations):
        # ##: Select a node and expand.
        node = select_child(root, exploration_weight)
        if not is_done(node.state):
            node = node.add_child()

        # ##: Simulate and back-propagate.
        reward = simulate(node)
        backpropagate(node, reward)

    return root
