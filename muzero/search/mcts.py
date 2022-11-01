# -*- coding: utf-8 -*-
"""
Core Monte Carlo Tree Search algorithm.
"""
from muzero.addons import MonteCarlosConfig, UpperConfidenceBounds, NoiseConfig
from .node import Node, ActionOutcomeHistory
from muzero.models import Network, NetworkOutput, LatentState, AfterState
from .helpers import MinMaxStats
import numpy as np
from typing import List, Union

from math import log, sqrt


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

    Parameters
    ----------
    config
    node
    min_max_stats

    Returns
    -------

    """
    if node.is_chance:
        # ##: If the node is chance, sample from the prior.
        outcomes, probs = zip(*[(o, n.prob) for o, n in node.children.items()])
        outcome = np.random.choice(outcomes, p=probs)
        return outcome, node.children[outcome]

    # ##: The decision nodes is based on the pUCT formula.
    _, action, child = max((ucb_score(config, node, child, min_max_stats), action, child) for action, child in node.children.items())
    return action, child


def expand_node(node: Node, state: Union[LatentState, AfterState], network_output: NetworkOutput, player: int, is_chance: bool):
    """
    Expand a node using the value, reward and policy prediction obtained from the neural network.

    Parameters
    ----------
    node
    state
    network_output
    player
    is_chance

    Returns
    -------

    """
    node.to_play = player
    node.state = state
    node.is_chance = is_chance
    node.reward = network_output.reward
    for action, prob in network_output.probabilities.items():
        node.children[action] = Node(prob)


def backpropagate(search_path: List[Node], value: float, to_play: int, discount: float, min_max_stats: MinMaxStats):
    """
    At the end of a simulation, propagate the evaluation all the way up the tree to the root.

    Parameters
    ----------
    search_path
    value
    to_play
    discount
    min_max_stats

    Returns
    -------

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

    Parameters
    ----------
    config
    node

    Returns
    -------

    """
    actions = list(node.children.keys())
    dir_alpha = config.root_dirichlet_alpha
    if config.root_dirichlet_adaptive:
        dir_alpha = 1.0 / np.sqrt(len(actions))

    noise = np.random.dirichlet([dir_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def run_mcts(config: MonteCarlosConfig, root: Node, action_outcome_history: ActionOutcomeHistory, network: Network, min_max_stats: MinMaxStats):
    """
    To decide on an action, run N simulations, always starting at the root of the search tree and traversing the
    tree according to the UCB formula until it reach a leaf node.

    Parameters
    ----------
    config
    root
    action_outcome_history
    network
    min_max_stats

    Returns
    -------

    """
    for _ in range(config.num_simulations):
        history = action_outcome_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action_or_outcome, node = select_child(config.bounds, node, min_max_stats)
            history.add_action_or_outcome(action_or_outcome)
            search_path.append(node)

        parent = search_path[-2]
        if parent.is_chance:
            # ##: The parent is a chance node, the last action or outcome is a chance outcome.
            child_state = network.dynamics(parent.state, history.last_action_or_outcome())
            network_output = network.predictions(child_state)

            # ##: This child is a decision node.
            is_child_chance = False
        else:
            # ##: The parent is a decision node, the last action or outcome is an action.
            child_state = network.afterstate_dynamics(parent.state, history.last_action_or_outcome())
            network_output = network.afterstate_predictions(child_state)

            # ##: The child is a chance node.
            is_child_chance = True

        # Expand the node.
        expand_node(node, child_state, network_output, history.to_play(), is_child_chance)

        # Back-propagate the value up the tree.
        backpropagate(search_path, network_output.value, history.to_play(), config.bounds.discount, min_max_stats)
