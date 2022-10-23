# -*- coding: utf-8 -*-
"""
Set of types for this project.
"""
from collections import namedtuple
from dataclasses import dataclass

Experience = namedtuple(
    typename="Experience",
    field_names=[
        "state",
        "next_state",
        "action",
        "reward",
        "done",
    ],
)


@dataclass
class AgentConfiguration:
    """
    Agent configuration.
    """

    discount: float
    store_model: str
    learning_rate: float


@dataclass
class AgentConfigurationDQN(AgentConfiguration):
    """
    Agent configuration for DQN.
    """

    type_model: str
    epsilon_max: float
    epsilon_min: float
    epsilon_decay: float


@dataclass
class AgentConfigurationPPO(AgentConfiguration):
    """
    Agent configuration for PPO.
    """

    second_learning_rate: float
    clip_ratio: float


@dataclass
class TrainingConfiguration:
    """
    Train configuration.
    """

    observation_type: str
    store_history: str
    epoch: int


@dataclass
class TrainingConfigurationDQN(TrainingConfiguration):
    """
    Training configuration for DQN
    """

    batch_size: int
    update_target: int
    memory_size: int
    agent_configuration: AgentConfigurationDQN
    agent_type: str


@dataclass
class TrainingConfigurationA2C(TrainingConfiguration):
    """
    Training configuration for A2C.
    """

    discount: float
    learning_rate: float


@dataclass
class TrainingConfigurationPPO(TrainingConfiguration):
    """
    Training configuration for PPO.
    """
