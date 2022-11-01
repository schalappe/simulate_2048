# -*- coding: utf-8 -*-
"""
New types for MuZero.
"""
from typing import Sequence, List, Dict, NamedTuple, Any


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


# ##: Sequence of state.
Trajectory = Sequence[State]


# ##: Action for environment.
Action: int


# ##: Player
Player: int


# ##: A chance outcome.
Outcome = Any


# ##: State in the network mind.
LatentState = List[float]
AfterState = List[float]


class NetworkOutput(NamedTuple):
    """
    The network's output.
    """
    value: float
    probabilities: Dict[int, float]
    reward: Optional[float] = 0.0

