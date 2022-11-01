# -*- coding: utf-8 -*-
"""
Set of class for Self-play.
"""
from abc import ABCMeta, abstractmethod
from muzero.addons import Action, SearchStats, StochasticMuZeroConfig
from muzero.models import NetworkCacher, NetworkOutput
from muzero.search import Node, run_mcts, backpropagate, add_exploration_noise, expand_node, ActionOutcomeHistory, MinMaxStats

import numpy as np


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
    def select_action(self, env: Environment) -> Action:
        """
        Selects an action for the current state of the environment.

        Parameters
        ----------
        env: Environment
            Simulator of an environment

        Returns
        -------
        Action
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

    def __int__(self, config: StochasticMuZeroConfig, cacher: NetworkCacher):
        self.config = config
        self.cacher = cacher
        self.training_step = -1
        self.network = None

    def reset(self):
        """
        Resets the player for a new episode.
        """
        # ##: Read a network from the cacher for the new episode.
        self.training_step, self.network = self.cacher.load_network()
        self.root = None

    @classmethod
    def _mask_illegal_actions(cls, env: Environment, outputs: NetworkOutput) -> NetworkOutput:
        """
        Masks any actions which are illegal at the root.

        Parameters
        ----------
        env: Environment
            Simulator of an environment
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
        for action in env.legal_actions():
            if action in network_policy:
                masked_policy[action] = network_policy[action]
        else:
            masked_policy[action] = 0.0
        norm += masked_policy[action]

        # ##: Re-normalize the masked policy.
        masked_policy = {a: v / norm for a, v in masked_policy.items()}
        return NetworkOutput(value=outputs.value, probabilities=masked_policy)

    def _select_action(self, root: Node) -> Action:
        """
        Selects an action given the root node.

        Parameters
        ----------
        root: Node
            A tree search root node

        Returns
        -------
        Action
            An action chosen with a certain probability
        """

        # ##: Get the visit count distribution.
        actions, visit_counts = zip(*[(action, node.visit_counts) for action, node in root.children.items()])

        # ##: Temperature
        temperature = self.config.visit_softmax_temperature_fn(self.training_step)

        # ##: Compute the search policy.
        search_policy = [v ** (1. / temperature) for v in visit_counts]
        norm = sum(search_policy)
        search_policy = [v / norm for v in search_policy]

        return np.random.choice(actions, p=search_policy)

    def select_action(self, env: Environment) -> Action:
        """
        Selects an action.

        Parameters
        ----------
        env: Environment
            Simulator of an environment

        Returns
        -------
        Action
            A selected action
        """

        # ##: New min max stats for the search tree.
        min_max_stats = MinMaxStats(self.config.known_bounds)

        # ##: At the root of the search tree
        # ##: use the representation function to obtain a hidden state given the current observation.
        root = Node(0)
        latent_state = self.network.representation(env.observation())

        # ##: Compute the predictions.
        outputs = self.network.predictions(latent_state)

        # ##: Keep only the legal actions.
        outputs = self._mask_illegal_actions(env, outputs)

        # ##: Expand the root node.
        expand_node(root, latent_state, outputs, env.to_play(), is_chance=False)

        # ##: Back-propagate the value.
        backpropagate([root], outputs.value, env.to_play(), self.config.search.bounds.discount, min_max_stats)

        # ##: Add exploration noise to the root node.
        add_exploration_noise(self.config.noise, root)

        # ##: Run a Monte Carlo Tree Search using only action sequences and the model learned by the network.
        run_mcts(self.config.search, root, ActionOutcomeHistory(env.to_play()), self.network, min_max_stats)

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
            raise ValueError('No search was executed.')
        return SearchStats(
            search_policy={action: node.visit_counts for action, node in self.root.children.items()},
            search_value=self.root.value()
        )
