# -*- coding: utf-8 -*-
"""
Deep Q-Network (DQN) implementation for playing the 2048 game.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from keras import Model, models, ops
from numpy import ndarray
from numpy.random import Generator, default_rng

from neurals import make_prediction


class DQNActor:
    """
    A Deep Q-Network (DQN) actor for the 2048 game.

    This class implements an agent that uses a DQN to choose actions in the 2048 game.
    It supports both epsilon-greedy exploration and deterministic action selection.
    """

    action_size: ClassVar[int] = 4
    GENERATOR: ClassVar[Generator] = default_rng(seed=1335)

    def __init__(self, model: Model, epsilon: float = 0.1, encodage_size: int = 31):
        """
        Initialize a DQNActor instance.

        Parameters
        ----------
        model : Model
            The Keras model representing the DQN.
        epsilon : float, optional
            The probability of choosing a random action for exploration (default is 0.1).
        encodage_size : int, optional
            The size of the encoded state representation (default is 31).
        """
        self.model = model
        self.epsilon = epsilon
        self.encodage_size = encodage_size

    @classmethod
    def from_path(cls, path: Path, epsilon: float = 0.1) -> DQNActor:
        """
        Create a `DQNActor` instance by loading a model from a file.

        Parameters
        ----------
        path : Path
            The path to the saved model file.
        epsilon : float, optional
            The probability of choosing a random action for exploration (default is 0.1).

        Returns
        -------
        DQNActor
            A new `DQNActor` instance with the loaded model.
        """
        return cls(models.load_model(path), epsilon=epsilon)

    def choose_action(self, state: ndarray) -> int:
        """
        Choose an action based on the current game state.

        This method implements an epsilon-greedy strategy, where with probability  epsilon, a random action is chosen,
        and with probability 1-epsilon, the action with the highest Q-value is chosen.

        Parameters
        ----------
        state : ndarray
            The current game state.

        Returns
        -------
        int
            The chosen action (0-3).
        """
        if self.GENERATOR.random() < self.epsilon:
            return self.GENERATOR.integers(self.action_size)
        q_values = make_prediction(self.model, state=state)
        return int(ops.argmax(q_values).numpy())
