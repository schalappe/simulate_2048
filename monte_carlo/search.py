# -*- coding: utf-8 -*-
"""
List of functions for Monte Carlo Tree Search.
"""
from typing import Union

from numpy import log, ndarray, sqrt
from numpy.random import PCG64DXSM, default_rng

from simulate.utils import is_done, legal_actions, next_state

from .node import Chance, Decision

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


def select_child(node: Union[Decision, Chance], exploration_weight: float) -> Union[Decision, Chance]:
    """
    Select a child node for expansion using the UCB1 algorithm.

    This function is a key part of the selection phase in Monte Carlo Tree Search. It chooses the most promising
    child node based on the UCB1 scores.

    Parameters
    ----------
    node : Union[Decision, Chance]
        The current node from which to select a child.
    exploration_weight : float
        The exploration weight for the UCB1 calculation.

    Returns
    -------
    Union[Decision, Chance]
        The selected child node.
    """
    while node.children:
        if not node.fully_expanded():
            return node
        if isinstance(node, Decision):
            node = uct_select(node, exploration_weight)
        else:
            node = GENERATOR.choice(node.children, p=[child.prior for child in node.children])
    return node


def simulate(node: Union[Decision, Chance]) -> float:
    """
    Perform a rollout simulation from the given node.

    This function is part of the simulation phase in Monte Carlo Tree Search. It plays out the game from the
    given state using random moves until the game ends.

    Parameters
    ----------
    node : Union[Decision, Chance]
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


def backpropagate(node: Decision, reward: float) -> None:
    """
    Back-propagate the reward through the tree.

    This function is part of the backpropagation phase in Monte Carlo Tree Search. It updates the visit count
    and value of each node in the path from the simulated leaf node to the root.

    Parameters
    ----------
    node : Decision
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
    Perform Monte Carlo Tree Search from the given root node.

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
    - After the search, the root's children represent the possible next moves, with visit counts
      indicating their estimated strength.
    """
    root = Decision(state=state, final=False, prior=0.0)
    for _ in range(iterations):
        # ##: Select a node and expand.
        node = select_child(root, exploration_weight)
        if not is_done(node.state):
            node = node.add_child()

        # ##: Simulate and backpropagate.
        reward = simulate(node)
        backpropagate(node, reward)

    return root
