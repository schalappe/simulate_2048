# -*- coding: utf-8 -*-
"""
Set of types for this project.
"""
from dataclasses import dataclass

INPUT_SIZE = 496


@dataclass
class TrainingConfiguration:
    """
    Train configuration.
    """

    store_history: str
    training_steps: int
    learning_rate: float
    discount: float
    save_steps: int
    max_steps: int


@dataclass
class TrainingConfigurationDQN(TrainingConfiguration):
    """
    Training configuration for DQN
    """

    batch_size: int
    replay_step: int
    update_step: int
    greedy_step: int
    memory_size: int


@dataclass
class TrainingConfigurationA2C(TrainingConfiguration):
    """
    Training configuration for A2C.
    """


@dataclass
class TrainingConfigurationPPO(TrainingConfiguration):
    """
    Training configuration for PPO.
    """
