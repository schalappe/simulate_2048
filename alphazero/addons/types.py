# -*- coding: utf-8 -*-
"""
New types for AlphaZero.
"""
from typing import Dict, List, NamedTuple, Sequence, Tuple

from numpy import ndarray


class SearchStats(NamedTuple):
    """
    Statistic throw Tree Search.
    """

    search_policy: Dict[int, int]
    search_value: float


class State(NamedTuple):
    """
    Data for a single state.
    """

    observation: List[float]
    reward: float
    discount: float
    action: int
    search_stats: SearchStats


# ##: Sequence of state.
Trajectory = Sequence[State]


class NetworkOutput(NamedTuple):
    """
    The network's output.
    """

    value: float
    probabilities: Dict[int, float]


class StochasticState(NamedTuple):
    """
    Stochastic state for Tree Search.
    """

    state: ndarray
    probability: float


class SimulatorOutput(NamedTuple):
    """
    The simulator's output.
    """

    stochastic_states: Dict[int, Tuple[List[StochasticState], int]]
