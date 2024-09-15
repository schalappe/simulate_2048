# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search (MCTS) implementation for the 2048 game.

This module provides a collection of functions that implement the core components of
the Monte Carlo Tree Search algorithm, specifically tailored for the 2048 game.
It includes functions for tree traversal, node selection, expansion, simulation,
and backpropagation.

This implementation takes into account the stochastic nature of the 2048 game,
handling both decision nodes (where the player chooses a move) and chance nodes
(where a new tile is randomly added to the board).
"""
from math import log, sqrt

from numpy import ndarray
from numpy.random import PCG64DXSM, default_rng

from simulate.utils import is_done, legal_actions, next_state
from simulate.wrappers import normalize_reward

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
    Select a child node using the PUCT (Predictor + UCT) formula.

    This function implements the selection strategy used in AlphaGo and similar algorithms,
    balancing exploration and exploitation.

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
    The PUCT formula is: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    where:
    - Q(s,a) is the average value of the child node
    - P(s,a) is the prior probability of selecting the action
    - N(s) is the visit count of the parent node
    - N(s,a) is the visit count of the child node
    - c_puct is the exploration weight

    This method provides a balance between exploring promising actions and exploiting known
    good actions.

    Raises
    ------
    ValueError
        If the input node is not a Decision node.
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

    This function is a key part of the selection phase in Monte Carlo Tree Search. It chooses
    the most promising child node based on the UCB1 scores.

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


def adaptive_simulation_count(node: Node, base_simulations: int) -> int:
    """
    Calculate the adaptive simulation count for a node using logarithmic depth scaling.

    Parameters
    ----------
    node : Node
        The node for which to calculate the simulation count.
    base_simulations : int
        The base number of simulations to perform.

    Returns
    -------
    int
        The number of simulations to perform for this node.

    Notes
    -----
    Uses the formula: N_sim = N_base * (1 + log(depth + 1))
    This formula provides a balanced approach to simulation allocation:
    - Nodes closer to the root get more simulations, but the increase is logarithmic.
    - The log(depth + 1) term ensures that even deep nodes get a reasonable number of simulations.
    - Adding 1 to the depth prevents taking log(0) for the root node.
    """
    return int(base_simulations * (1 + log(node.depth + 1)))


def simulate(node: Node, simulations: int) -> float:
    """
    Perform multiple rollout simulations from the given node.

    Parameters
    ----------
    node : Node
        The starting node for the simulations.
    simulations : int
        The number of simulations to perform.

    Returns
    -------
    float
        The average reward from all simulations.

    Notes
    -----
    - Performs multiple simulations and returns the average reward.
    - Each simulation uses random legal moves until the game ends.
    - The total reward is the sum of rewards from each move in the simulation.
    """
    total_reward = 0.0

    for _ in range(simulations):
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
            total_reward += normalize_reward(reward)

    return total_reward / simulations


def backpropagate(node: Node, reward: float) -> None:
    """
    Back-propagate the reward through the tree.

    This function is part of the backpropagation phase in Monte Carlo Tree Search. It updates
    the visit count and value of each node in the path from the simulated leaf node to the root.

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


def monte_carlo_search(
    state: ndarray, iterations: int, base_simulations: int = 10, exploration_weight: float = 1.41
) -> Decision:
    """
    Perform Monte Carlo Tree Search on the given game state.

    This function implements the main loop of the Monte Carlo Tree Search algorithm, iteratively
    building a search tree to find the best action.

    Parameters
    ----------
    state : ndarray
        The initial game state.
    iterations : int
        The number of iterations to perform the search.
    base_simulations : int, optional
        The base number of simulations to perform for each node, by default 10.
    exploration_weight : float, optional
        The exploration weight for the PUCT formula, by default 1.41.

    Returns
    -------
    Decision
        The root node of the search tree.

    Notes
    -----
    The function uses adaptive simulation count to allocate more computational resources to
    important nodes. Progressive Widening is applied to manage the branching factor in
    outcome spaces.

    The search process consists of four main steps:
    1. Selection: Traverse the tree to select a promising node.
    2. Expansion: Add a new child to the selected node.
    3. Simulation: Perform rollouts from the new node.
    4. Backpropagation: Update node statistics based on simulation results.
    """
    root = Decision(state=state, final=False, prior=1.0)

    for _ in range(iterations):
        # ##: Select a node and expand.
        node = select_child(root, exploration_weight)
        if not is_done(node.state):
            node = node.add_child()

        # ##: Simulate and back-propagate.
        simulation_count = adaptive_simulation_count(node, base_simulations)
        reward = simulate(node, simulation_count)
        backpropagate(node, reward)

    return root
