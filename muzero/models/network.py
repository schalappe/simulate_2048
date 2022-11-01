# -*- coding: utf-8 -*-
"""
Set of class for network use by Alpha Zero.
"""
from typing import Tuple
from muzero.addons import LatentState, AfterState, Outcome, NetworkOutput


# ##: TODO: Complete ...
class Network:
    """
    An instance of the network used by Alpha Zero.
    """

    def representation(self, observation) -> LatentState:
        """Representation function maps from observation to latent state."""
        return []

    def predictions(self, state: LatentState) -> NetworkOutput:
        """
        Returns the network predictions for a state.

        Parameters
        ----------
        state: LatentState
            The current state of the game

        Returns
        -------
        NetworkOutput
            The value of given state and the probabilities distribution over all moves
        """
        return NetworkOutput(0, {})

    def afterstate_dynamics(self, state: LatentState, action: int) -> AfterState:
        """Implements the dynamics from latent state and action to after-state."""
        return []

    def afterstate_predictions(self, state: AfterState) -> NetworkOutput:
        """
        Returns the network predictions for an after-state.

        Parameters
        ----------
        state: AfterState
            The current state of the game

        Returns
        -------
        NetworkOutput
            The value of given state and the probabilities distribution over all moves
        """

        # No reward for after-state transitions.
        return NetworkOutput(0, {})

    def dynamics(self, state: AfterState, action: Outcome) -> LatentState:
        """Implements the dynamics from afterstate and chance outcome to
        state."""
        return []

    def encoder(self, observation) -> Outcome:
        """An encoder maps an observation to an outcome."""


class NetworkCacher:
    """
    An object to share the network between the self-play and training jobs.
    """
    def __init__(self):
        self._networks = {}

    def save_network(self, step: int, network: Network):
        """
        Save a network in the cacher.

        Parameters
        ----------
        step: int
            The training step
        network: Network
            The network to store
        """
        self._networks[step] = network

    def load_network(self) -> Tuple[int, Network]:
        """
        Return the latest stored network.

        Returns
        -------
        tuple
            The latest training step and his network.
        """
        training_step = max(self._networks.keys())
        return training_step, self._networks[training_step]
