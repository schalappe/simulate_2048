# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search for 2048.
"""
from numpy import ndarray, sqrt

from .node import Decision
from .search import monte_carlo_search


class MonteCarloAgent:
    """
    An agent that uses Monte Carlo Tree Search to play 2048.

    This agent implements the Monte Carlo Tree Search (MCTS) algorithm to make decisions in the 2048 game.
    It explores possible game states and actions to choose the most promising move.

    Methods
    -------
    choose_action(state: ndarray)
        Choose the best action for the given game state using MCTS.
    """

    def __init__(self, iterations: int = 10, exploration_weight: float = sqrt(2)):
        """
        Initialize the Monte Carlo agent.

        Parameters
        ----------
        iterations : int, optional
            The number of iterations for each search (default is 10).
        exploration_weight : float, optional
            The exploration weight for UCB1 (default is sqrt(2)).
        """
        self._iterations = iterations
        self._exploration_weight = exploration_weight

    @classmethod
    def _best_action(cls, root: Decision) -> int:
        """
        Choose the best action based on visit counts.

        This method selects the child node with the highest number of visits, which represents the most
        promising action.

        Parameters
        ----------
        root : Decision
            The root node of the search tree.

        Returns
        -------
        int
            The action corresponding to the most visited child node.
        """
        return max(root.children, key=lambda c: c.visits).action

    def choose_action(self, state: ndarray) -> int:
        """
        Choose the best action using Monte Carlo Tree Search.

        This method performs MCTS on the given game state to determine the best action to take. It creates a search
        tree, expands it through multiple iterations, and then selects the most promising action.

        Parameters
        ----------
        state : np.ndarray
            The current game state.

        Returns
        -------
        int
            The chosen action.
        """
        root = monte_carlo_search(state, iterations=self._iterations, exploration_weight=self._exploration_weight)
        return self._best_action(root)
