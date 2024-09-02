# -*- coding: utf-8 -*-
"""
List of functions for Monte Carlo Tree Search.
"""
from math import log, sqrt
from typing import List

from numpy import log2, ndarray
from numpy.random import PCG64DXSM, default_rng

from simulate.utils import is_done, legal_actions, next_state

from .node import Chance, Decision, Node

GENERATOR = default_rng(PCG64DXSM())


def normalize_reward(reward: float, max_tile: int = 2 ** (4**2)) -> float:
    """
    Normalize the reward using logarithmic scaling.

    Parameters
    ----------
    reward : float
        The raw reward obtained from a move.
    max_tile : int
        The maximum tile value for the given board size.

    Returns
    -------
    float
        The normalized reward value between 0 and 1.

    Notes
    -----
    This method uses logarithmic normalization to compress the range of rewards,
    making the learning process more stable across different stages of the game.
    The normalization is based on the maximum theoretical tile value for the given board size.
    """
    if reward == 0:
        return 0.0
    return log2(reward) / log2(max_tile)


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


def monte_carlo_search(
    state: ndarray,
    iterations: int,
    base_simulations: int = 2,
    exploration_weight: float = 1.41,
    convergence_window: int = 100,
    convergence_threshold: float = 0.95,
) -> Decision:
    """
    Perform Monte Carlo Tree Search.

    This function implements the main loop of the Monte Carlo Tree Search algorithm. It repeatedly selects, expands,
    simulates, and back-propagates for a specified  number of iterations.

    Parameters
    ----------
    state : np.ndarray
        The initial game state.
    iterations : int
        The number of iterations to perform the search.
    base_simulations : int, optional
        The base number of simulations to perform for each node (default is 5).
    exploration_weight : float, optional
        The exploration weight for the UCB1 formula (default is 1.41).
    convergence_window : int, optional
        The number of recent iterations to consider for convergence (default is 100).
    convergence_threshold : float, optional
        The threshold for considering the search converged (default is 0.95).

    Returns
    -------
    Decision
        The root node of the search tree.

    Notes
    -----
    - Each iteration consists of four phases: selection, expansion, simulation, and backpropagation.
    - Uses adaptive simulation count to allocate more computational resources to important nodes.
    - The search builds a tree of possible game states and actions, estimating their values.
    - Progressive Widening is applied to manage the branching factor in outcome spaces.
    - After the search, the root's children represent the possible next moves, with visit counts
      indicating their estimated strength.
    - The search is considered converged if the best action remains the same for a certain
      percentage (convergence_threshold) of the recent iterations (convergence_window).
    """
    root = Decision(state=state, final=False, prior=1.0)
    best_action_history: List[int] = []

    for iteration in range(iterations):
        # ##: Select a node and expand.
        node = select_child(root, exploration_weight)
        if not is_done(node.state):
            node = node.add_child()

        # ##: Simulate and back-propagate.
        simulation_count = adaptive_simulation_count(node, base_simulations)
        reward = simulate(node, simulation_count)
        backpropagate(node, reward)

        # ##: Check for convergence.
        current_best_action = max(root.children, key=lambda child: child.visits).action
        best_action_history.append(current_best_action)

        if iteration >= convergence_window:
            recent_actions = best_action_history[-convergence_window:]
            most_common_action = max(set(recent_actions), key=recent_actions.count)
            if recent_actions.count(most_common_action) / convergence_window >= convergence_threshold:
                break

    return root
