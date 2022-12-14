# -*- coding: utf-8 -*-
"""
Set of config for AlphaZero.
"""
from dataclasses import dataclass
from typing import Callable

ENCODAGE_SIZE = 31


@dataclass
class BufferConfig:
    """
    Configuration for the replay buffer.
    """

    td_steps: int
    td_lambda: float
    batch_size: int
    num_trajectories: int


@dataclass
class UpperConfidenceBounds:
    """
    Configuration for compute UBC.
    """

    discount: float
    pb_c_base: float
    pb_c_init: float


@dataclass
class NoiseConfig:
    """
    Configuration for adding noise.
    """

    root_dirichlet_alpha: float
    root_dirichlet_adaptive: bool
    root_exploration_fraction: float


@dataclass
class MonteCarlosConfig:
    """
    Configuration for Monte Carlos Tree Search.
    """

    bounds: UpperConfidenceBounds
    num_simulations: int


@dataclass
class SelfPlayConfig:
    """
    Self play configuration.
    """

    episodes: int
    evaluation: int


@dataclass
class Factory:
    """
    Set factory function for AlphaZero.
    """

    network_factory: Callable
    environment_factory: Callable


@dataclass
class TrainingConfig:
    """
    Training configuration.
    """

    epochs: int
    export: int
    store_path: str
    training_step: int
    learning_rate: float
    visit_softmax_temperature: Callable


@dataclass
class StochasticAlphaZeroConfig:
    """
    Configuration for AlphaZero.
    """

    noise: NoiseConfig
    search: MonteCarlosConfig
    replay: BufferConfig
    factory: Factory
    training: TrainingConfig
    self_play: SelfPlayConfig
