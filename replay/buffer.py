# -*- coding: utf-8 -*-
"""
Replay buffer for storing and sampling experiences in reinforcement learning. Enables efficient training
by allowing random access to past interactions.
"""
from typing import Tuple, Protocol, Any


class ReplayBuffer(Protocol):
    """
    Protocol defining the interface for replay buffers in reinforcement learning.

    A replay buffer stores and manages experiences (state, action, reward, next_state, done) collected during
    the agent's interaction with the environment. It provides methods for adding new experiences and sampling
    batches of experiences for training.

    Implementing classes should define the following methods:
    - add: Add a new experience to the buffer
    - sample: Retrieve a random batch of experiences
    - __len__: Return the current number of experiences in the buffer

    This interface allows for different implementations of replay buffers, such as uniform sampling,
    prioritized experience replay, or other variants.
    """

    def add(self, experience: Tuple[Any, ...]) -> None:
        """
        Add a new experience to the buffer.

        Parameters
        ----------
        experience : Tuple[Any]
            Experience to add to the buffer.
        """

    def sample(self, batch_size: int) -> Tuple[Any, ...]:
        """
        Sample a batch of experiences from the buffer.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        Tuple[Any, ...]
            A tuple containing batches of experiences.
        """

    def __len__(self) -> int:
        """Return the current size of the buffer."""
