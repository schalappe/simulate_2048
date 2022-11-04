# -*- coding: utf-8 -*-
"""
New types for AlphaZero.
"""
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from numpy import ndarray


class SearchStats(NamedTuple):
    search_policy: Dict[int, int]
    search_value: float


class State(NamedTuple):
    """
    Data for a single state.
    """

    observation: List[float]
    reward: float
    discount: float
    player: int
    action: int
    search_stats: SearchStats


class NetworkOutput(NamedTuple):
    """
    The network's output.
    """

    value: float
    probabilities: Dict[int, float]


class SimulatorOutput(NamedTuple):
    """
    The simulator's output.
    """

    stochastic_states = Dict[int, Tuple[Tuple[ndarray, float], int]]


# ##: Sequence of state.
Trajectory = Sequence[State]


# ##: Action for environment.
Action: int


# ##: Player
Player: int
