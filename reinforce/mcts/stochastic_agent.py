"""
Stochastic MuZero Agent for playing 2048.

This module provides an agent that uses the full Stochastic MuZero algorithm:
neural network predictions combined with MCTS for decision making.

The agent can operate in two modes:
1. With trained network: Uses network for policy priors and value estimates
2. Without network (evaluation): Uses random initialization for benchmarking
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from numpy import ndarray

from reinforce.neural.network import StochasticNetwork, create_stochastic_network

from .network_search import DecisionNode, get_policy_from_visits, run_network_mcts, select_action_from_root


@dataclass
class SearchStats:
    """
    Statistics from a single MCTS search.

    Useful for training data collection and analysis.

    Attributes
    ----------
    root_value : float
        Value estimate at the root node.
    visit_counts : dict[int, int]
        Visit counts for each action.
    policy : dict[int, float]
        Search policy (normalized visit counts).
    selected_action : int
        The action that was selected.
    """

    root_value: float
    visit_counts: dict[int, int]
    policy: dict[int, float]
    selected_action: int


class StochasticMuZeroAgent:
    """
    Agent using Stochastic MuZero for 2048.

    This agent combines a learned neural network model with Monte Carlo Tree Search
    to make decisions. The network provides:
    - Policy priors for guiding search
    - Value estimates for leaf evaluation
    - Afterstate predictions for stochastic transitions

    Attributes
    ----------
    network : StochasticNetwork
        The neural network for predictions.
    num_simulations : int
        Number of MCTS simulations per move.
    exploration_weight : float
        The c_puct exploration constant.
    temperature : float
        Action selection temperature (0 = greedy, 1 = proportional).
    add_noise : bool
        Whether to add Dirichlet noise at root.

    Examples
    --------
    >>> from reinforce.mcts.stochastic_agent import StochasticMuZeroAgent
    >>> agent = StochasticMuZeroAgent.create_untrained()
    >>> action = agent.choose_action(game_state)
    """

    def __init__(
        self,
        network: StochasticNetwork,
        num_simulations: int = 100,
        exploration_weight: float = 1.25,
        temperature: float = 0.0,
        add_noise: bool = False,
        dirichlet_alpha: float = 0.25,
        noise_fraction: float = 0.25,
    ):
        """
        Initialize the Stochastic MuZero agent.

        Parameters
        ----------
        network : StochasticNetwork
            The neural network for predictions.
        num_simulations : int
            Number of MCTS simulations per move.
        exploration_weight : float
            The c_puct exploration constant.
        temperature : float
            Action selection temperature.
        add_noise : bool
            Whether to add Dirichlet exploration noise at root.
        dirichlet_alpha : float
            Dirichlet noise alpha parameter.
        noise_fraction : float
            Fraction of noise to mix with priors.
        """
        self.network = network
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.temperature = temperature
        self.add_noise = add_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction

        # ##>: Store last search for stats retrieval.
        self._last_root: DecisionNode | None = None

    @classmethod
    def create_untrained(
        cls,
        observation_shape: tuple[int, ...] = (16,),
        hidden_size: int = 256,
        codebook_size: int = 32,
        num_simulations: int = 100,
        **kwargs,
    ) -> StochasticMuZeroAgent:
        """
        Create an agent with a randomly initialized network.

        Useful for:
        - Benchmarking MCTS without training
        - Testing the architecture
        - Debugging

        Parameters
        ----------
        observation_shape : tuple[int, ...]
            Shape of the observation (flattened board).
        hidden_size : int
            Hidden state dimension.
        codebook_size : int
            Number of chance codes.
        num_simulations : int
            Number of MCTS simulations.
        **kwargs
            Additional arguments for the agent.

        Returns
        -------
        StochasticMuZeroAgent
            An agent with untrained network.
        """
        network = create_stochastic_network(
            observation_shape=observation_shape, hidden_size=hidden_size, codebook_size=codebook_size
        )
        return cls(network=network, num_simulations=num_simulations, **kwargs)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str | Path, num_simulations: int = 100, **kwargs
    ) -> StochasticMuZeroAgent:
        """
        Load an agent from a training checkpoint.

        Parameters
        ----------
        checkpoint_dir : str | Path
            Directory containing saved model files.
        num_simulations : int
            Number of MCTS simulations.
        **kwargs
            Additional arguments for the agent.

        Returns
        -------
        StochasticMuZeroAgent
            An agent with loaded network.
        """
        checkpoint_dir = Path(checkpoint_dir)
        network = StochasticNetwork.from_path(
            representation_path=str(checkpoint_dir / 'representation'),
            prediction_path=str(checkpoint_dir / 'prediction'),
            afterstate_dynamics_path=str(checkpoint_dir / 'afterstate_dynamics'),
            afterstate_prediction_path=str(checkpoint_dir / 'afterstate_prediction'),
            dynamics_path=str(checkpoint_dir / 'dynamics'),
            encoder_path=str(checkpoint_dir / 'encoder'),
        )
        return cls(network=network, num_simulations=num_simulations, **kwargs)

    def choose_action(self, state: ndarray) -> int:
        """
        Choose the best action for the given game state.

        Runs MCTS with network guidance and selects an action based on
        the resulting visit distribution.

        Parameters
        ----------
        state : ndarray
            The current game board state (4x4 array).

        Returns
        -------
        int
            The chosen action (0-3 for left/up/right/down).
        """
        root = run_network_mcts(
            game_state=state,
            network=self.network,
            num_simulations=self.num_simulations,
            exploration_weight=self.exploration_weight,
            add_exploration_noise=self.add_noise,
            dirichlet_alpha=self.dirichlet_alpha,
            noise_fraction=self.noise_fraction,
        )
        self._last_root = root
        return select_action_from_root(root, temperature=self.temperature)

    def get_search_stats(self) -> SearchStats | None:
        """
        Get statistics from the last search.

        Returns
        -------
        SearchStats | None
            Statistics from the last search, or None if no search has been performed.
        """
        if self._last_root is None:
            return None

        root = self._last_root
        visit_counts = {action: child.visit_count for action, child in root.children.items()}
        policy = get_policy_from_visits(root, temperature=1.0)
        selected = select_action_from_root(root, temperature=self.temperature)

        # ##>: Root value is the average of Q-values weighted by visits.
        total_visits = sum(visit_counts.values())
        if total_visits > 0:
            root_value = sum(child.value_sum for child in root.children.values()) / total_visits
        else:
            root_value = root.value

        return SearchStats(root_value=root_value, visit_counts=visit_counts, policy=policy, selected_action=selected)

    def set_training_mode(self, training: bool = True) -> None:
        """
        Configure agent for training or evaluation.

        In training mode:
        - Higher temperature for exploration
        - Dirichlet noise at root

        In evaluation mode:
        - Greedy action selection
        - No exploration noise

        Parameters
        ----------
        training : bool
            Whether to enable training mode.
        """
        if training:
            self.temperature = 1.0
            self.add_noise = True
        else:
            self.temperature = 0.0
            self.add_noise = False
