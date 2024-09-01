# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search node classes for decision-making in stochastic environments.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from numpy import array_equal, ndarray, power
from numpy.random import PCG64DXSM, default_rng

from simulate.utils import after_state, is_done, latent_state, legal_actions

GENERATOR = default_rng(PCG64DXSM())


@dataclass(kw_only=True)
class Node(ABC):
    """
    Base class for Monte Carlo Tree Search nodes.

    Attributes
    ----------
    state : np.ndarray
        The current state representation.
    values : float
        Accumulated rewards from simulations.
    visits : int
        Number of times the node has been visited.
    children : List
        List of child nodes.

    Methods
    -------
    fully_expanded()
        Check if the node is fully expanded.
    update(reward)
        Update node statistics after a simulation.
    """

    state: ndarray
    values: float = 0.0
    visits: int = 0
    children: List = field(default_factory=list)

    @abstractmethod
    def fully_expanded(self) -> bool:
        """Check if the node is fully expanded."""

    @abstractmethod
    def add_child(self) -> Node:
        """Add a new chance node as a child."""

    def update(self, reward: float) -> None:
        """
        Update node statistics after a simulation.

        This method updates the accumulated values and visit count of the node.

        Parameters
        ----------
        reward : float
            The reward received from the simulation.
        """
        self.values += reward
        self.visits += 1


@dataclass(kw_only=True)
class Decision(Node):
    """
    Represents a decision node in the Monte Carlo Tree Search.

    Attributes
    ----------
    prior : float
        Prior probability of selecting this node.
    final : bool
        Whether this node represents a terminal state.
    parent : Optional[Chance]
        The parent chance node.
    legal_moves : List[int]
        List of legal actions from this state.

    Methods
    -------
    fully_expanded()
        Check if all legal moves have been explored.
    add_child()
        Add a new chance node as a child.
    """

    prior: float
    final: bool
    parent: Optional[Chance] = None
    legal_moves: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Initialize legal moves if the node is not final."""
        if not self.final:
            self.legal_moves = legal_actions(self.state)

    def fully_expanded(self) -> bool:
        """Check if all legal moves have been explored."""
        return len(self.children) == len(self.legal_moves)

    def add_child(self) -> Chance:
        """
        Add a new chance node as a child.

        This method selects an unexplored action and creates a new Chance node.

        Returns
        -------
        ChanceWithWidening
            The newly created chance node.

        Raises
        ------
        ValueError
            If all actions have been tried. Node should be fully expanded.
        """
        untried_actions = set(self.legal_moves) - {child.action for child in self.children}
        if not untried_actions:
            raise ValueError("All actions have been tried. Node should be fully expanded.")
        action = int(GENERATOR.choice(list(untried_actions)))
        child_state, _ = latent_state(self.state, action)
        child = Chance(state=child_state, parent=self, action=action)
        self.children.append(child)
        return child


@dataclass(kw_only=True)
class Chance(Node):
    """
    Represents a chance node in the Monte Carlo Tree Search.

    Attributes
    ----------
    action : int
        The action taken to reach this node.
    parent : Decision
        The parent decision node.
    next_states : List[Tuple[np.ndarray, float]]
        Possible next states and their probabilities.
    widening_alpha : float
        Exponent for outcome progressive widening.
    widening_constant : float
        Constant for outcome progressive widening.

    Methods
    -------
    fully_expanded()
        Check if all possible next states have been explored.
    add_child()
        Add a new decision node as a child.
    """

    action: int
    parent: Decision
    next_states: List[Tuple[ndarray, float]] = field(default_factory=list)
    widening_alpha: float = 0.25
    widening_constant: float = 1.0

    def __post_init__(self):
        """Initialize possible next states."""
        self.next_states = after_state(self.state)

    def fully_expanded(self) -> bool:
        """Check if all possible next states have been explored according to progressive widening."""
        return len(self.children) >= min(
            len(self.next_states), self.widening_constant * power(self.visits, self.widening_alpha)
        )

    def add_child(self) -> Decision:
        """
        Add a new decision node as a child.

        This method selects an unexplored outcome and creates a new Decision node.

        Returns
        -------
        Decision
            The newly created decision node.

        Raises
        ------
        ValueError
            If all possible outcomes have been tried.
        """
        unvisited_outcomes = [
            (outcome, prior)
            for outcome, prior in self.next_states
            if not any(array_equal(outcome, child.state) for child in self.children)
        ]

        if not unvisited_outcomes:
            raise ValueError("All outcomes have been tried. Node should be fully expanded.")

        outcome, prior = unvisited_outcomes[GENERATOR.choice(len(unvisited_outcomes))]
        child = Decision(state=outcome, prior=prior, final=is_done(outcome), parent=self)
        self.children.append(child)
        return child
