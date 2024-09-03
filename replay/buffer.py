# -*- coding: utf-8 -*-
"""
Replay buffer for reinforcement learning.
"""
from abc import ABC, abstractmethod
from typing import Tuple

from numpy import ndarray


class ReplayBuffer(ABC):
    """
    Abstract base class for replay buffers.

    This class defines the interface for replay buffers used in reinforcement learning algorithms.

    Attributes
    ----------
    capacity : int
        Maximum number of experiences that can be stored in the buffer.
    """

    def __init__(self, capacity: int):
        """
        Initialize the AbstractReplayBuffer.

        Parameters
        ----------
        capacity : int
            Maximum number of experiences that can be stored.
        """
        self.capacity = capacity

    @abstractmethod
    def add(self, state: ndarray, action: int, reward: float, next_state: ndarray, done: bool) -> None:
        """
        Add a new experience to the buffer.

        Parameters
        ----------
        state : ndarray
            The current state.
        action : int
            The action taken.
        reward : float
            The reward received.
        next_state : ndarray
            The resulting next state.
        done : bool
            Whether the episode has ended.
        """
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Tuple[ndarray, ...]:
        """
        Sample a batch of experiences from the buffer.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        Tuple[ndarray, ...]
            A tuple containing batches of experiences.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        pass
