# -*- coding: utf-8 -*-
"""
Set of types for this project.
"""
from collections import namedtuple
from dataclasses import dataclass

from numpy import ndarray

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
    type_model: str
    store_model: str
    learning_rate: float


@dataclass
class AgentConfigurationDQN(AgentConfiguration):
    discount: float
    epsilon_max: float
    epsilon_min: float
    epsilon_decay: float
    batch_size: int
    memory_size: int


@dataclass
class AgentConfigurationPPO(AgentConfiguration):
    batch_size: int
    second_learning_rate: float


@dataclass
class TrainingConfigurationDQN:
    reward_type: str
    observation_type: str
    store_history: str
    epoch: int
    update_target: int
    agent_configuration: AgentConfiguration


@dataclass
class TrainingConfigurationPPO:
    reward_type: str
    observation_type: str
    store_history: str
    epoch: int
