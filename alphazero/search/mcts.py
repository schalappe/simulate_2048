# -*- coding: utf-8 -*-
"""
Core Monte Carlo Tree Search algorithm.
"""
from math import log, sqrt
from typing import List

import numpy as np
from numpy import ndarray

from alphazero.addons.config import (
    MonteCarlosConfig,
    NoiseConfig,
    UpperConfidenceBounds,
)
from alphazero.addons.types import NetworkOutput, SimulatorOutput
from alphazero.game.simulator import Simulator
from alphazero.models.network import Network

from .helpers import MinMaxStats
from .node import Node


def ucb_score(config: UpperConfidenceBounds, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
    """
    The score for a node is based on its value, plus an exploration bonus based on the prior.

    Parameters
    ----------
    config: UpperConfidenceBounds
        UBC configuration
    parent: Node
        Parent node
    child: Node
        Child node
    min_max_stats: MinMaxStats
        MinMax class

    Returns
    -------
    float
        Value of UBC
    """
    pb_c = log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(child.reward + config.discount * child.value())
    else:
        value_score = 0
    return prior_score + value_score


def select_child(config: UpperConfidenceBounds, node: Node, min_max_stats: MinMaxStats) -> Node:
    """
    Select the child with the highest UCB score.

    Parameters
    ----------
    config: UpperConfidenceBounds
        UBC configuration
    node: Node
        A node
    min_max_stats: MinMaxStats
        MinMax class

    Returns
    -------
    Node
        A children node selected
    """
    if node.is_chance:
        # ##: If the node is chance, sample from the prior.
        outcomes, probs = zip(*[(o, n.prior) for o, n in node.children.items()])
        outcome = np.random.choice(outcomes, p=probs)
        return node.children[outcome]

    # ##: The decision nodes is based on the pUCT formula.
    _, _, child = max(
        (ucb_score(config, node, child, min_max_stats), action, child) for action, child in node.children.items()
    )
    return child


def expand_node(node: Node, network_output: NetworkOutput, simulator_output: SimulatorOutput):
    """
    Expand a node using the value, reward and policy prediction obtained from the neural network.

    Parameters
    ----------
    node: Node
        A node
    network_output: NetworkOutput
        Output of policy network
    simulator_output: SimulatorOutput
        Output of the simulator
    """
    # ##: Loop over moves distribution.
    for action, prob in network_output.probabilities.items():
        # ##: Create the chance node.
        stochastic_node = Node(prior=prob, is_chance=True)

        # ##: Get stochastic state and reward
        stochastic_states, reward = simulator_output.stochastic_states[action]
        for index, stochastic_state in enumerate(stochastic_states):
            # ##: Create decision node.
            child = Node(prior=stochastic_state.probability, is_chance=False)
            child.state = stochastic_state.state
            child.reward = reward

            # ##: Create father link between stochastic node and his child.
            stochastic_node.children[index] = child

        # ##: Create father link between node and stochastic node.
        node.children[action] = stochastic_node


def backpropagate(search_path: List[Node], value: float, discount: float, min_max_stats: MinMaxStats):
    """
    At the end of a simulation, propagate the evaluation all the way up the tree to the root.

    Parameters
    ----------
    search_path: List
        List of node
    value: float:
        Value given the policy network
    discount: float
        Discount
    min_max_stats: MinMaxStats
        MinMax class
    """
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())
        value = node.reward + discount * value


def add_exploration_noise(config: NoiseConfig, node: Node):
    """
    At the start of each search, we add dirichlet noise to the prior of the root to encourage the search to
    explore new actions.

    Parameters
    ----------
    config: NoiseConfig
        Noise configuration
    node: Node
        Root node
    """
    actions = list(node.children.keys())
    dir_alpha = config.root_dirichlet_alpha
    if config.root_dirichlet_adaptive:
        dir_alpha = 1.0 / np.sqrt(len(actions))

    noise = np.random.dirichlet([dir_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for action, _noise in zip(actions, noise):
        node.children[action].prior = node.children[action].prior * (1 - frac) + _noise * frac


def mask_illegal_actions(state: ndarray, simulator: Simulator, outputs: NetworkOutput) -> NetworkOutput:
    """
    Masks any actions which are illegal at the root.

    Parameters
    ----------
    state: ndarray
        Current state
    simulator: Simulator
        Environment simulation
    outputs: NetworkOutput
        Previous network output

    Returns
    -------
    NetworkOutput
        New network output with legal action
    """

    # ##: We mask out and keep only the legal actions.
    masked_policy = {}
    network_policy = outputs.probabilities
    norm = 0
    for action in simulator.legal_actions(state):
        if action in network_policy:
            masked_policy[action] = network_policy[action]
        else:
            masked_policy[action] = 0.0
        norm += masked_policy[action]

    # ##: Re-normalize the masked policy.
    masked_policy = {a: v / norm for a, v in masked_policy.items()}
    return NetworkOutput(value=outputs.value, probabilities=masked_policy)


def run_mcts(
    config: MonteCarlosConfig, root: Node, network: Network, simulator: Simulator, min_max_stats: MinMaxStats
):
    """
    To decide on an action, run N simulations, always starting at the root of the search tree and traversing the
    tree according to the UCB formula until it reach a leaf node.

    Parameters
    ----------
    config: MonteCarlosConfig
        Monte Carlos Tree Search configuration
    root: Node
        Root node
    network: Network
        Policy network
    simulator: Simulator
        2048 Simulator
    min_max_stats: MinMaxStats
        MinMax class
    """
    for _ in range(config.num_simulations):
        node = root
        search_path = [node]

        while node.expanded():
            node = select_child(config.bounds, node, min_max_stats)
            search_path.append(node)

        # ##: Get value and probability distribution over all moves.
        network_output = network.predictions(node.state)
        network_output = mask_illegal_actions(node.state, simulator, network_output)

        # ##: Get all children possible
        simulator_output = simulator.step(node.state)

        # ##: Expand the node with a simulator.
        expand_node(node, network_output, simulator_output)

        # Back-propagate the value up the tree.
        backpropagate(search_path, network_output.value, config.bounds.discount, min_max_stats)
