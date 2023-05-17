# -*- coding: utf-8 -*-
"""
Class and function of experiences replay.
"""
from collections import deque
from random import sample as random_sample
from typing import Sequence

from numpy import random as numpy_random

from .replay import Trajectories


class ExperienceReplay:
    """Class used to train an agent."""

    def __init__(self, *, capacity: int, batch_size: int):
        self._buffer = deque(maxlen=capacity)
        self._batch_size = batch_size

    def store(self, trajectories: Trajectories) -> None:
        """
        Save a sequence of state.

        Parameters
        ----------
        trajectories: Trajectory
            A sequence of experience
        """
        self._buffer.append(trajectories)

    def sample(self) -> Sequence[Trajectories]:
        """
        Samples a training batch.

        Returns
        -------
        trajectories: Sequence[Trajectories]
            A sequence of experience
        """
        indice = numpy_random.choice(len(self._buffer), self._batch_size, replace=False)
        return [self._buffer[index] for index in indice]

    def __len__(self) -> int:
        return len(self._buffer)
