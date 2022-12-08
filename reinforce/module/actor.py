# -*- coding: utf-8 -*-
"""
Set of class for Self-play.
"""
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import ndarray

from reinforce.addons.config import StochasticAlphaZeroConfig
from reinforce.addons.types import SearchStats
from reinforce.game.simulator import Simulator
from reinforce.models.network import Network
from reinforce.search.helpers import MinMaxStats
from reinforce.search.mcts import (
    add_exploration_noise,
    backpropagate,
    expand_node,
    run_mcts,
)
from reinforce.search.node import Node


class Actor(metaclass=ABCMeta):
    """
    An actor to interact with the environment.
    """

    @abstractmethod
    def reset(self):
        """
        Resets the player for a new episode.
        """

    @abstractmethod
    def select_action(self, state: ndarray) -> int:
        """
        Selects an action for the current state of the environment.

        Parameters
        ----------
        state: ndarray
            Current state

        Returns
        -------
        int
            A selected action.
        """

    @abstractmethod
    def stats(self) -> SearchStats:
        """
        Returns the stats for the player after it has selected an action.

        Returns
        -------
        SearchStats
            Stats of the player
        """


class StochasticMuZeroActor(Actor):
    """
    A MuZero actor for self-play.
    """

    def __init__(self, config: StochasticAlphaZeroConfig, network: Network, epochs: int, train: bool = True):
        self.config = config
        self.network = network
        self.root = None
        self.simulator = Simulator()
        self.epochs = epochs
        self.train = train

    def reset(self):
        """
        Resets the player for a new episode.
        """
        self.root = None

    def _select_action(self, root: Node) -> int:
        """
        Selects an action given the root node.

        Parameters
        ----------
        root: Node
            A tree search root node

        Returns
        -------
        int
            An action chosen with a certain probability
        """

        # ##: Get the visit count distribution.
        actions, visit_counts = zip(*[(action, node.visit_count) for action, node in root.children.items()])

        if not self.train:
            index = np.argmax(np.array(visit_counts))
            return actions[index]

        # Temperature
        temperature = self.config.training.visit_softmax_temperature(self.epochs)

        # ##: Compute the search policy.
        search_policy = [v ** (1.0 / temperature) for v in visit_counts]
        norm = sum(search_policy)
        search_policy = [v / norm for v in search_policy]

        return np.random.choice(actions, p=search_policy)

    def select_action(self, state: ndarray) -> int:
        """
        Selects an action.

        Parameters
        ----------
        state: ndarray
            Current state

        Returns
        -------
        Action
            A selected action
        """
        # ##: Define root node.
        root = Node(0)
        root.state = state

        # ##: New min max stats for the search tree.
        min_max_stats = MinMaxStats(None)

        # ##: Compute the predictions and keep only the legal actions.
        outputs = self.network.predictions(state)
        outputs = self.simulator.mask_illegal_actions(state, outputs)

        # ##: Generate all possibles children
        simulator_outputs = self.simulator.step(state)

        # ##: Expand the root node.
        expand_node(root, network_output=outputs, simulator_output=simulator_outputs)

        # ##: Back-propagate the value.
        backpropagate([root], outputs.value, self.config.search.bounds.discount, min_max_stats)

        # ##: Add exploration noise to the root node.
        if self.train:
            add_exploration_noise(self.config.noise, root)

        # ##: Run a Monte Carlo Tree Search using only action sequences and the model learned by the network.
        run_mcts(self.config.search, root, self.network, self.simulator, min_max_stats)

        # ##: Keep track of the root to return the stats.
        self.root = root

        # ##: Return an action.
        return self._select_action(root)

    def stats(self) -> SearchStats:
        """
        Returns the stats of the latest search.

        Returns
        -------
        SearchStats
            Stat of the latest search
        """

        if self.root is None:
            raise ValueError("No search was executed.")
        return SearchStats(
            search_policy={action: node.visit_count for action, node in self.root.children.items()},
            search_value=self.root.value(),
        )
