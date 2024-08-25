# -*- coding: utf-8 -*-
"""
List of functions for Monte Carlo Tree Search.
"""
from numpy import log, ndarray, sqrt
from numpy.random import default_rng

from simulate.utils import after_state, is_done, latent_state, legal_actions, next_state

from .node import Node

GENERATOR = default_rng(seed=None)


def ucb_score(node: Node, exploration_weight: float) -> float:
    """
    Calculate the Upper Confidence Bound (UCB1) score for a node.

    This function implements the UCB1 formula used in the selection phase of Monte Carlo Tree Search to balance
    exploration and exploitation.

    Parameters
    ----------
    node : Node
        The node for which to calculate the UCB1 score.
    exploration_weight : float
        The exploration weight parameter in the UCB1 formula.

    Returns
    -------
    float
        The calculated UCB1 score.

    Notes
    -----
    The UCB1 score is calculated as:
    exploitation + exploration_weight * sqrt(log(parent_visits) / node_visits)

    If the node has not been visited, it returns positive infinity to ensure unvisited nodes are explored.
    """
    if node.visits == 0:
        return float("inf")
    exploitation = node.exploitation
    exploration = sqrt(log(node.parent.visits) / node.visits)
    return exploitation + exploration_weight * exploration


def select_child(node: Node, exploration_weight: float) -> Node:
    """
    Select a child node for expansion using the UCB1 algorithm.

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

    Notes
    -----
    - If the node has no children, it returns the node itself.
    - For chance nodes, it randomly selects a child based on prior probabilities.
    - For decision nodes, it selects the child with the highest UCB1 score.
    - Unvisited children are prioritized for exploration.
    """
    # ##: Check if the node has any children,
    if not node.children:
        return node

    # ##: If the node is a chance node, sample from the prior probabilities.
    if node.is_chance:
        outcomes, probabilities = zip(*[(child, child.prior) for child in node.children])
        return GENERATOR.choice(outcomes, p=probabilities)

    # ##: Select the child with the highest UCB1 score.
    unvisited_children = [child for child in node.children if child.visits == 0]
    if unvisited_children:
        return unvisited_children[0]

    return max(node.children, key=lambda child: ucb_score(child, exploration_weight))


def expand_node(node: Node) -> None:
    """
    Expand a node by adding its possible child nodes.

    This function is part of the expansion phase in Monte Carlo Tree Search. It generates all possible next states
    from the current node and adds them as child nodes.

    Parameters
    ----------
    node : Node
        The node to expand.

    Notes
    -----
    - For non-chance parent nodes, it generates probable next states.
    - For chance parent nodes or the root, it generates child nodes for all legal actions.
    - Each child node is initialized with appropriate state, prior probability, and parent information.
    """
    if node.parent and not node.parent.is_chance:
        probable_states = after_state(node.state)
        children = [Node(state=state, prior=prior, is_chance=False, parent=node) for state, prior in probable_states]
    else:
        legal_moves = legal_actions(node.state)
        children = [
            Node(
                state=latent_state(node.state, action),
                prior=1.0,
                is_chance=True,
                parent=node,
                action=action,
            )
            for action in legal_moves
        ]
    node.children = children


def simulate(state: ndarray) -> float:
    """
    Simulate a random game from the given state to completion.

    This function is part of the simulation phase in Monte Carlo Tree Search. It plays out the game
    from the given state using random moves until the game ends.

    Parameters
    ----------
    state : np.ndarray
        The starting state for the simulation.

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
    done = False
    current_state = state.copy()

    while not done:
        legal_moves = legal_actions(current_state)
        if not legal_moves:
            break
        current_state, reward = next_state(current_state, GENERATOR.choice(legal_moves))
        done = is_done(current_state)
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
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent


def monte_carlo_search(root: Node, iterations: int, exploration_weight: float = 1.41) -> None:
    """
    Perform Monte Carlo Tree Search from the given root node.

    This function implements the main loop of the Monte Carlo Tree Search algorithm. It repeatedly selects, expands,
    simulates, and back-propagates for a specified  number of iterations.

    Parameters
    ----------
    root : Node
        The root node of the search tree.
    iterations : int
        The number of iterations to perform the search.
    exploration_weight : float, optional
        The exploration weight for the UCB1 formula (default is 1.41).

    Notes
    -----
    - Each iteration consists of four phases: selection, expansion, simulation, and backpropagation.
    - The search builds a tree of possible game states and actions, estimating their values.
    - After the search, the root's children represent the possible next moves, with visit counts
      indicating their estimated strength.
    """
    for _ in range(iterations):
        node = root

        # ##: Select a node to expand.
        while node.expanded:
            node = select_child(node, exploration_weight)

        # ##: Expand the node by adding a child node.
        expand_node(node)
        node = select_child(node, exploration_weight)

        # ##: Simulate a random game.
        reward = simulate(node.state)

        # ##: Back-propagate the reward through the tree.
        backpropagate(node, reward)
