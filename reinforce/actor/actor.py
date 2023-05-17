# -*- coding: utf-8 -*-
"""
Actor to interact with the environment.
"""
from typing import Protocol

import tensorflow as tf


class Actor(Protocol):
    """An actor to interact with the environment."""

    def play(self):
        """Takes network, produces episodes and stores ten into replay buffer."""
        ...

    def run_episode(self, initial_state: tf.Tensor):
        """Runs a single episode to collect training data."""
        ...
