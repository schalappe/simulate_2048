# -*- coding: utf-8 -*-
"""
Helper for Tree search
"""
from typing import NamedTuple, Optional

MAXIMUM_FLOAT_VALUE = float("inf")


class KnownBounds(NamedTuple):
    min: float
    max: float


class MinMaxStats(object):
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        """
        Update a border value.

        Parameters
        ----------
        value: float
            Node's value
        """
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        """
        Normalize the node's value.

        Parameters
        ----------
        value: float
            Node's value

        Returns
        -------
        float:
            Normalized node's value
        """
        if self.maximum > self.minimum:
            # ##: Normalize only when there are a maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
