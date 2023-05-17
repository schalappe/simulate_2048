# -*- coding: utf-8 -*-
"""
Learner to update the network weight.
"""
from typing import Protocol, Sequence

import tensorflow as tf


class Learner(Protocol):
    """A learner to update the network weights based."""

    def train_step(self, experiences: Sequence[Sequence[tf.Tensor]]) -> tf.Tensor:
        ...

    def learn(self) -> float:
        """Single training step of the learner."""
        ...
