# -*- coding:utf-8 -*-
"""
Simulator for helping during Monte Carlos Tree Search
"""
from typing import Tuple

from numpy import ndarray

from .types import Action, SimulatorOutput


class Simulator:
    """
    Simulator class.
    Implement the rules of the 2048 Game.
    """

    def step(self, state: ndarray) -> SimulatorOutput:
        pass
