# -*- coding: utf-8 -*-
"""
Class and function of experience buffer.
"""
from typing import Protocol, Sequence

import tensorflow as tf

Trajectories = Sequence[tf.TensorArray]


class BufferReplay(Protocol):
    """Buffer Replay class."""

    _batch_size: int

    def store(self, trajectories: Trajectories) -> None:
        """
        Save a sequence of state.

        Parameters
        ----------
        trajectories: Trajectory
            A sequence of experience
        """
        ...

    def sample(self) -> Sequence[Trajectories]:
        """
        Samples a training batch.

        Returns
        -------
        trajectories: Sequence[Trajectories]
            A sequence of experience
        """
        ...
