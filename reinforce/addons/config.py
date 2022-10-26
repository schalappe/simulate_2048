# -*- coding: utf-8 -*-
"""
Set of types for this project.
"""
from collections import namedtuple
from dataclasses import dataclass
import tensorflow as tf

INPUT_SIZE = 496


# Experience = namedtuple(
#    typename="Experience",
#    field_names=[
#        "state",
#        "next_state",
#        "action",
#        "reward",
#        "done",
#    ],
# )
class Experience(namedtuple):
    state: tf.Tensor
    next_state: tf.Tensor
    action: int
    reward: tf.float32
    done: tf.bool


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

    store_history: str
    training_steps: int


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
    save_steps: int


@dataclass
class TrainingConfigurationPPO(TrainingConfiguration):
    """
    Training configuration for PPO.
    """
