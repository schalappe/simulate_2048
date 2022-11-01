# -*- coding: utf-8 -*-
"""
Component of a Monte Carlos Tree.
"""
from typing import List, Optional, Union, Any


# ##: An object that holds an action or a chance outcome.
ActionOrOutcome = Union[int, Any]


class ActionOutcomeHistory:
    """
    Simple history container used inside the search.
    Only used to keep track of the actions and chance outcomes executed..
    """

    def __init__(self, history: Optional[List[ActionOrOutcome]] = None):
        self._history = list(history)

    def clone(self):
        """
        Create a new container with the same history of this container.

        Returns
        -------
        ActionOutcomeHistory
            New history container
        """
        return ActionOutcomeHistory(self._history)

    def add_action_or_outcome(self, action_or_outcome: ActionOrOutcome):
        """
        Add action or outcome to the history.

        Parameters
        ----------
        action_or_outcome: ActionOrOutcome
            An executed action
        """
        self._history.append(action_or_outcome)

    def last_action_or_outcome(self) -> ActionOrOutcome:
        """
        Return the last executed action or outcome.

        Returns
        -------
        ActionOrOutcome
            Executed action
        """
        return self._history[-1]

    def to_play(self) -> int:
        """
        Returns the next player to play based on the initial player.

        Returns
        -------
        int
            Player to play
        """
        # ##: for 2048 it is always the same player.
        return 0

    @classmethod
    def action_space(cls) -> List[int]:
        """
        Action space.

        Returns
        -------
        list
            All possible action
        """
        return list(range(4))


class Node(object):
    """
    A Node in the Monte Carlos Tree Search.
    """

    def __init__(self, prior: float, is_chance: bool = False):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.is_chance = is_chance
        self.reward = 0

    def expanded(self) -> bool:
        """
        It's a leaf or not.

        Returns
        -------
        bool
            True if expanded
            False else
        """
        return len(self.children) > 0

    def value(self) -> float:
        """
        Compute the value of the node.

        Returns
        -------
        float
            Value of the node.
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
