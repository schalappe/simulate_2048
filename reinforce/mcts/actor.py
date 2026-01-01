"""
Monte Carlo Tree Search implementation for the 2048 game.

This module provides a Monte Carlo Tree Search (MCTS) based agent for playing the 2048 game.
It includes classes and functions for performing MCTS, evaluating game states, and making
decisions based on the search results.

The MCTS algorithm used here is specifically tailored for the 2048 game, taking into account
its unique characteristics such as the stochastic nature of tile spawns and the large state space.
"""

from numpy import ndarray, sqrt

from .node import Decision
from .search import monte_carlo_search


class MonteCarloAgent:
    """
    An agent that uses Monte Carlo Tree Search to play 2048.

    This agent implements the Monte Carlo Tree Search (MCTS) algorithm to make decisions in the 2048 game.
    It explores possible game states and actions to choose the most promising move.

    Attributes
    ----------
    iterations : int
        The number of iterations for each search.
    exploration_weight : float
        The exploration weight for the PUCT formula.

    Methods
    -------
    choose_action(state: ndarray)
        Choose the best action for the given game state using MCTS.

    Notes
    -----
    The agent uses the PUCT (Predictor + UCT) formula for node selection,
    which balances exploration and exploitation during the search process.
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
        self.iterations = iterations
        self.exploration_weight = exploration_weight

    @classmethod
    def _best_action(cls, root: Decision) -> int:
        """
        Choose the best action based on visit counts, with Q-value tiebreaker.

        This method selects the child node with the highest number of visits, which represents the most
        promising action. If multiple children have equal visits, the one with higher Q-value wins.

        Parameters
        ----------
        root : Decision
            The root node of the search tree.

        Returns
        -------
        int
            The action corresponding to the most visited child node.
        """

        # ##>: Tiebreak by Q-value (average reward) when visit counts are equal.
        def selection_key(child):
            q_value = child.values / child.visits if child.visits > 0 else 0.0
            return (child.visits, q_value)

        return max(root.children, key=selection_key).action

    def choose_action(self, state: ndarray) -> int:
        """
        Choose the best action using Monte Carlo Tree Search.

        This method performs MCTS on the given game state to determine the best action to take.
        It creates a search tree, expands it through multiple iterations, and then selects the most
        promising action.

        Parameters
        ----------
        state : np.ndarray
            The current game state.

        Returns
        -------
        int
            The chosen action.

        Notes
        -----
        The method uses the number of visits to each child node as the criterion for selecting the best action,
        which is a common approach in MCTS implementations.
        """
        root = monte_carlo_search(state, iterations=self.iterations, exploration_weight=self.exploration_weight)
        return self._best_action(root)
