# -*- coding: utf-8 -*-
"""
Class used to hold experience.
"""
from collections import deque, namedtuple

from numpy.random import choice

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


class ReplayMemory:
    """
    Memory buffer for Experience Replay.
    """

    def __init__(self, buffer_length: int):
        self.memory = deque(maxlen=buffer_length)

    def __len__(self) -> int:
        return len(self.memory)

    def append(self, experience: Experience):
        """
        Add experience to the buffer.

        Parameters
        ----------
        experience: Experience
            Experience to add to the buffer
        """
        self.memory.append(experience)

    def sample(self, batch_size: int) -> list:
        """
        Sample a batch of experiences from the buffer.

        Parameters
        ----------
        batch_size: int
            Number of experiences to randomly select

        Returns
        -------
        list
            List of selected experiences
        """
        # ## ----> Choose randomly indice.
        indices = choice(len(self.memory), batch_size, replace=False)
        return [self.memory[indice] for indice in indices]
