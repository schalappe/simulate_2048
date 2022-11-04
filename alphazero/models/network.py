# -*- coding: utf-8 -*-
"""
Set of class for network use by Alpha Zero.
"""
from typing import Tuple

from numpy import ndarray

from alphazero.addons.types import NetworkOutput


# ##: TODO: Complete ...
class Network:
    """
    An instance of the network used by Alpha Zero.
    """

    def predictions(self, state: ndarray) -> NetworkOutput:
        """
        Returns the network predictions for a state.

        Parameters
        ----------
        state: ndarray
            The current state of the game

        Returns
        -------
        NetworkOutput
            The value of given state and the probabilities distribution over all moves
        """
        return NetworkOutput(0, {})


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
