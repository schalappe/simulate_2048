# -*- coding: utf-8 -*-
"""
Component of a Monte Carlos Tree.
"""


class Node:
    """
    A Node in the Monte Carlos Tree Search.
    """

    def __init__(self, prior: float, is_chance: bool = False):
        self.visit_count = 0
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
