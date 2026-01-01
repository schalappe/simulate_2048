"""
Monte Carlo Tree Search (MCTS) implementation for the 2048 game.

This module provides a collection of functions that implement the core components of the Monte Carlo
Tree Search algorithm, specifically tailored for the 2048 game. It includes functions for tree traversal,
node selection, expansion, simulation, and backpropagation.

This implementation takes into account the stochastic nature of the 2048 game, handling both decision
nodes (where the player chooses a move) and chance nodes (where a new tile is randomly added to the board).
"""

from math import log, sqrt

from numpy import ndarray, zeros
from numpy.random import PCG64DXSM, default_rng

from twentyfortyeight.core.gameboard import TILE_SPAWN_PROBS, is_done, next_state
from twentyfortyeight.core.gamemove import legal_actions
from twentyfortyeight.utils.normalize import normalize_reward

from .node import Chance, Decision

# ##>: Pre-computed tile values and probabilities for simulation sampling.
_TILE_VALUES = list(TILE_SPAWN_PROBS.keys())
_TILE_PROBS = list(TILE_SPAWN_PROBS.values())
_TILE_VALUES_ARR = [2, 4]

GENERATOR = default_rng(PCG64DXSM())

# ##>: Pre-allocated buffers for batched simulation (reused across calls).
_MAX_SIMULATIONS = 32
_BOARD_SIZE = 4
_SIM_BUFFER = zeros((_MAX_SIMULATIONS, _BOARD_SIZE, _BOARD_SIZE), dtype='int32')


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
        raise ValueError('UCB1 is only defined for Decision nodes.')

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
        raise ValueError('PUCT is only defined for Decision nodes.')

    # ##>: Hoist sqrt computation outside the loop for efficiency.
    sqrt_parent_visits = sqrt(node.visits)

    def puct_score(child: Chance) -> float:
        q_value = child.values / child.visits if child.visits > 0 else 0
        return q_value + exploration_weight * child.prior * sqrt_parent_visits / (1 + child.visits)

    return max(node.children, key=puct_score)


def select_child(node: Decision | Chance, exploration_weight: float) -> Decision | Chance:
    """
    Select a child node for expansion using the UCB1 algorithm with progressive widening.

    This function is a key part of the selection phase in Monte Carlo Tree Search. It chooses
    the most promising child node based on the UCB1 scores.

    Parameters
    ----------
    node : Decision | Chance
        The current node from which to select a child.
    exploration_weight : float
        The exploration weight for the UCB1 calculation.

    Returns
    -------
    Decision | Chance
        The selected child node.
    """
    current: Decision | Chance = node
    while current.children:
        if not current.fully_expanded():
            return current
        if isinstance(current, Decision):
            current = puct_select(current, exploration_weight)
        else:
            all_visits = sum(child.visits for child in current.children)
            current = GENERATOR.choice(current.children, p=[child.visits / all_visits for child in current.children])
    return current


def adaptive_simulation_count(node: Decision | Chance, base_simulations: int) -> int:
    """
    Calculate the adaptive simulation count for a node using inverse logarithmic depth scaling.

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
    Uses the formula: N_sim = N_base / (1 + log(depth + 1))
    This formula allocates more simulations near the root where decisions matter most:
    - Depth 0 (root): N_base simulations
    - Depth 1: ~N_base / 1.69 simulations
    - Depth 5: ~N_base / 2.79 simulations
    Deep nodes get fewer simulations since they are rarely revisited.
    """
    return max(1, int(base_simulations / (1 + log(node.depth + 1))))


def simulate(node: Decision | Chance, simulations: int) -> float:
    """
    Perform multiple rollout simulations from the given node.

    Parameters
    ----------
    node : Decision | Chance
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
    - Uses pre-allocated buffer to reduce memory allocation overhead.
    """
    # ##>: Clamp simulations to buffer size.
    num_sims = min(simulations, _MAX_SIMULATIONS)
    total_reward = 0.0

    # ##>: Initialize states in pre-allocated buffer.
    states = _SIM_BUFFER[:num_sims]
    for i in range(num_sims):
        states[i] = node.state

    # ##>: For Chance nodes, sample random tile spawns.
    if isinstance(node, Chance) and node._num_empty > 0:
        cell_indices = GENERATOR.integers(0, node._num_empty, size=num_sims)
        tile_values = GENERATOR.choice(_TILE_VALUES_ARR, size=num_sims, p=_TILE_PROBS)
        for i in range(num_sims):
            cell = node._empty_cells[cell_indices[i]]
            states[i, cell[0], cell[1]] = tile_values[i]

    # ##>: Track which simulations are still active.
    active = [True] * num_sims

    # ##>: Simulate until all games are done.
    while any(active):
        for i in range(num_sims):
            if not active[i]:
                continue

            state = states[i]
            if is_done(state):
                active[i] = False
                continue

            actions = legal_actions(state)
            action = GENERATOR.choice(actions)
            new_state, reward = next_state(state, action)
            states[i] = new_state
            total_reward += normalize_reward(reward)

    return total_reward / simulations


def backpropagate(node: Decision | Chance, reward: float) -> None:
    """
    Back-propagate the reward through the tree.

    This function is part of the backpropagation phase in Monte Carlo Tree Search. It updates
    the visit count and value of each node in the path from the simulated leaf node to the root.

    Parameters
    ----------
    node : Decision | Chance
        The starting node for backpropagation (typically a leaf node).
    reward : float
        The reward value to back-propagate.

    Notes
    -----
    - The function updates each node's visit count and cumulative value.
    - It continues updating until it reaches the root node (node with no parent).
    """
    current: Decision | Chance | None = node
    while current is not None:
        current.update(reward)
        current = current.parent


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
        # ##>: Select a node and expand.
        node = select_child(root, exploration_weight)
        if not is_done(node.state):
            node = node.add_child()

        # ##>: Simulate and back-propagate.
        simulation_count = adaptive_simulation_count(node, base_simulations)
        reward = simulate(node, simulation_count)
        backpropagate(node, reward)

    return root
