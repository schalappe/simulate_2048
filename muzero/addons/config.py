# -*- coding: utf-8 -*-
"""
Set of config for MuZero.
"""
from dataclasses import dataclass


INPUT_SIZE = 496


@dataclass
class BufferConfig:
    """
    Configuration for the replay buffer.
    """
    td_steps: int
    batch_size: int
    num_unroll_steps: int
    num_trajectories: int


@dataclass
class UpperConfidenceBounds:
    discount: float
    pb_c_base: float
    pb_c_init: float


@dataclass
class NoiseConfig:
    root_dirichlet_alpha: float
    root_dirichlet_adaptive: float
    root_exploration_fraction: float


@dataclass
class MonteCarlosConfig:
    bounds: UpperConfidenceBounds
    num_simulations: int


@dataclass
class StochasticMuZeroConfig:
    """
    Configuration for MuZero.
    """
    noise: NoiseConfig
    search: MonteCarlosConfig
    replay: BufferConfig