# -*- coding: utf-8 -*-
"""
Core Monte Carlo Tree Search algorithm.
"""
from math import log, sqrt
from typing import List

import numpy as np

from alphazero.addons.config import (
    MonteCarlosConfig,
    NoiseConfig,
    UpperConfidenceBounds,
)
from alphazero.addons.simulator import Simulator
from alphazero.addons.types import NetworkOutput, SimulatorOutput
from alphazero.models import Network

from .helpers import MinMaxStats
from .node import Node

# ##: TODO: faire un test unitaire pour tous les functions ici bas.


def ucb_score(config: UpperConfidenceBounds, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
    """
    The score for a node is based on its value, plus an exploration bonus based on the prior.

    Parameters
    ----------
    config
    parent
    child
    min_max_stats

    Returns
    -------

    """
    pb_c = log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(child.reward + config.discount * child.value())
    else:
        value_score = 0
    return prior_score + value_score


def select_child(config: UpperConfidenceBounds, node: Node, min_max_stats: MinMaxStats):
    """
    Select the child with the highest UCB score.
    """
    if node.is_chance:
        # ##: If the node is chance, sample from the prior.
        outcomes, probs = zip(*[(o, n.prob) for o, n in node.children.items()])
        outcome = np.random.choice(outcomes, p=probs)
        return node.children[outcome]

    # ##: The decision nodes is based on the pUCT formula.
    _, _, child = max(
        (ucb_score(config, node, child, min_max_stats), action, child) for action, child in node.children.items()
    )
    return child


def expand_node(node: Node, network_output: NetworkOutput, simulator_output: SimulatorOutput):
    """Expand a node using the value, reward and policy prediction obtained from the neural network."""
    # ##: Loop over moves distribution.
    for action, prob in network_output.probabilities.items():
        # ##: Create the chance node.
        stochastic_node = Node(prior=prob, is_chance=True)
        stochastic_node.to_play = 0

        # ##: Get stochastic state and reward
        stochastic_states, reward = simulator_output.stochastic_states[action]
        for index, state, prior in enumerate(stochastic_states):
            # ##: Create decision node.
            child = Node(prior=prior, is_chance=False)
            child.state = state
            child.reward = reward
            child.to_play = 0

            # ##: Create father link between stochastic node and his child.
            stochastic_node.children[index] = child

        # ##: Create father link between node and stochastic node.
        node.children[action] = stochastic_node


def backpropagate(search_path: List[Node], value: float, to_play: int, discount: float, min_max_stats: MinMaxStats):
    """
    At the end of a simulation, propagate the evaluation all the way up the tree to the root.
    """
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())
        value = node.reward + discount * value


def add_exploration_noise(config: NoiseConfig, node: Node):
    """
    At the start of each search, we add dirichlet noise to the prior of the root
    to encourage the search to explore new actions.
    """
    actions = list(node.children.keys())
    dir_alpha = config.root_dirichlet_alpha
    if config.root_dirichlet_adaptive:
        dir_alpha = 1.0 / np.sqrt(len(actions))

    noise = np.random.dirichlet([dir_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def run_mcts(
    config: MonteCarlosConfig, root: Node, network: Network, simulator: Simulator, min_max_stats: MinMaxStats
):
    """
    To decide on an action, run N simulations, always starting at the root of the search tree and traversing the
    tree according to the UCB formula until it reach a leaf node.
    """
    for _ in range(config.num_simulations):
        node = root
        search_path = [node]

        while node.expanded():
            node = select_child(config.bounds, node, min_max_stats)
            search_path.append(node)

        # ##: Get value and probability distribution over all moves.
        network_output = network.predictions(node.state)

        # ##: Get all children possible
        simulator_output = simulator.step(node.state)

        # ##: Expand the node with a simulator.
        expand_node(node, network_output, simulator_output)

        # Back-propagate the value up the tree.
        backpropagate(search_path, network_output.value, 0, config.bounds.discount, min_max_stats)
