# -*- coding: utf-8 -*-
"""
Monte carlo Node.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from numpy import ndarray


@dataclass
class Node:
    """
    Represents a node in the Monte Carlo search tree for the 2048 game.

    This class encapsulates the state and metadata of a node in the Monte Carlo Tree Search (MCTS) algorithm. Each
    node represents a game state and contains information about its children, parent, associated action, and
    statistical data used in the MCTS process.

    Attributes
    ----------
    state : ndarray
        The game state represented by this node.
    prior : float
        The prior probability of selecting this node.
    is_chance : bool
        Whether this node represents a chance event.
    children : List[Node]
        Child nodes representing possible next states.
    parent : Optional[Node]
        The parent node in the search tree.
    action : Optional[int]
        The action that led to this node from its parent.
    visits : int
        The number of times this node has been visited during search.
    value : float
        The cumulative value (score) associated with this node.

    Methods
    -------
    expanded : bool
        Property that checks if the node has been expanded (has children).
    exploitation : float
        Property that calculates the average value of the node.
    """

    state: ndarray
    prior: float
    is_chance: bool
    children: List[Node] = field(default_factory=list)
    parent: Optional[Node] = None
    action: Optional[int] = None
    visits: int = 0
    value: float = 0.0

    @property
    def expanded(self) -> bool:
        """
        Check if the node has been expanded.

        Returns
        -------
        bool
            True if the node has children, False otherwise.

        Notes
        -----
        An expanded node is one that has had its possible child states generated and added to its children list.
        """
        return bool(self.children)

    @property
    def exploitation(self) -> float:
        """
        Calculate the average value of the node.

        Returns
        -------
        float
            The average value (score) of the node, or 0 if the node hasn't been visited.

        Notes
        -----
        This property is used in the Upper Confidence Bound (UCB) calculation during the selection phase of
        the Monte Carlo Tree Search algorithm.
        """
        return self.value / self.visits if self.visits else 0.0
