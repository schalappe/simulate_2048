# -*- coding: utf-8 -*-
"""
Set of types for this project.
"""
from collections import namedtuple

Experience = namedtuple(typename="Experience", field_names=["state", "action", "reward", "done", "new_state"])

ConfigurationDQN = namedtuple(
    typename="ConfigurationDQN",
    field_names=[
        "env_size",
        "reward_type",
        "discount",
        "epsilon",
        "min_epsilon",
        "decay_epsilon",
        "batch_size",
        "learning_rate",
        "memory_size",
        "max_steps",
        "target_update",
    ],
)
