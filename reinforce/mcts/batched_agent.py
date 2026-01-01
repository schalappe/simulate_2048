"""
Batched Stochastic MuZero Agent for high-performance 2048 play.

This module provides agents that use the batched MCTS implementations
for significantly faster inference during self-play and evaluation.

Three optimization levels are available:
- SEQUENTIAL: Original implementation with Level 1 batching (4x faster)
- BATCHED: Level 2 leaf batching (8-16x faster)
- THREADED: Level 3 parallel MCTS with virtual loss (25-50x faster)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from numpy import ndarray

from reinforce.neural.network import StochasticNetwork, create_stochastic_network

from .batched_search import (
    DecisionNode,
    get_policy_from_visits,
    run_batched_mcts,
    run_threaded_mcts,
    select_action_from_root,
)
from .network_search import run_network_mcts
from .network_search import select_action_from_root as seq_select_action_from_root


class MCTSMode(str, Enum):
    """MCTS execution mode."""

    SEQUENTIAL = 'sequential'  # Original with Level 1 batching
    BATCHED = 'batched'  # Level 2 leaf batching
    THREADED = 'threaded'  # Level 3 parallel with virtual loss


@dataclass
class SearchStats:
    """Statistics from a single MCTS search."""

    root_value: float
    visit_counts: dict[int, int]
    policy: dict[int, float]
    selected_action: int


class BatchedMuZeroAgent:
    """
    High-performance Stochastic MuZero agent with configurable batching.

    This agent provides three MCTS modes:
    - SEQUENTIAL: Uses Level 1 batching (batched afterstate_dynamics)
    - BATCHED: Uses Level 2 leaf batching for ~10x speedup
    - THREADED: Uses Level 3 parallel MCTS for maximum throughput

    Attributes
    ----------
    network : StochasticNetwork
        The neural network for predictions.
    num_simulations : int
        Number of MCTS simulations per move.
    mode : MCTSMode
        The MCTS execution mode.
    batch_size : int
        Batch size for leaf batching (Level 2+).
    num_workers : int
        Number of worker threads (Level 3 only).
    """

    def __init__(
        self,
        network: StochasticNetwork,
        num_simulations: int = 100,
        mode: MCTSMode = MCTSMode.BATCHED,
        batch_size: int = 8,
        num_workers: int = 4,
        exploration_weight: float = 1.25,
        temperature: float = 0.0,
        add_noise: bool = False,
        dirichlet_alpha: float = 0.25,
        noise_fraction: float = 0.25,
    ):
        """
        Initialize the batched agent.

        Parameters
        ----------
        network : StochasticNetwork
            The neural network.
        num_simulations : int
            Number of MCTS simulations.
        mode : MCTSMode
            MCTS execution mode (sequential, batched, or threaded).
        batch_size : int
            Batch size for leaf collection.
        num_workers : int
            Number of parallel workers (threaded mode only).
        exploration_weight : float
            PUCT exploration constant.
        temperature : float
            Action selection temperature.
        add_noise : bool
            Whether to add Dirichlet noise at root.
        dirichlet_alpha : float
            Dirichlet noise alpha parameter.
        noise_fraction : float
            Fraction of noise to mix with priors.
        """
        self.network = network
        self.num_simulations = num_simulations
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.exploration_weight = exploration_weight
        self.temperature = temperature
        self.add_noise = add_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction

        self._last_root: DecisionNode | None = None

    @classmethod
    def create_untrained(
        cls,
        observation_shape: tuple[int, ...] = (16,),
        hidden_size: int = 256,
        codebook_size: int = 32,
        num_simulations: int = 100,
        mode: MCTSMode = MCTSMode.BATCHED,
        **kwargs,
    ) -> BatchedMuZeroAgent:
        """
        Create an agent with a randomly initialized network.

        Parameters
        ----------
        observation_shape : tuple[int, ...]
            Shape of the observation.
        hidden_size : int
            Hidden state dimension.
        codebook_size : int
            Number of chance codes.
        num_simulations : int
            Number of MCTS simulations.
        mode : MCTSMode
            MCTS execution mode.
        **kwargs
            Additional arguments.

        Returns
        -------
        BatchedMuZeroAgent
            An agent with untrained network.
        """
        network = create_stochastic_network(
            observation_shape=observation_shape, hidden_size=hidden_size, codebook_size=codebook_size
        )
        return cls(network=network, num_simulations=num_simulations, mode=mode, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        num_simulations: int = 100,
        mode: MCTSMode = MCTSMode.BATCHED,
        **kwargs,
    ) -> BatchedMuZeroAgent:
        """
        Load an agent from a training checkpoint.

        Parameters
        ----------
        checkpoint_dir : str | Path
            Directory containing saved model files.
        num_simulations : int
            Number of MCTS simulations.
        mode : MCTSMode
            MCTS execution mode.
        **kwargs
            Additional arguments.

        Returns
        -------
        BatchedMuZeroAgent
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
        return cls(network=network, num_simulations=num_simulations, mode=mode, **kwargs)

    def choose_action(self, state: ndarray) -> int:
        """
        Choose the best action for the given game state.

        Uses the configured MCTS mode for search.

        Parameters
        ----------
        state : ndarray
            The current game board state (4x4 array).

        Returns
        -------
        int
            The chosen action (0-3).
        """
        if self.mode == MCTSMode.SEQUENTIAL:
            root = run_network_mcts(
                game_state=state,
                network=self.network,
                num_simulations=self.num_simulations,
                exploration_weight=self.exploration_weight,
                add_exploration_noise=self.add_noise,
                dirichlet_alpha=self.dirichlet_alpha,
                noise_fraction=self.noise_fraction,
            )
            # ##>: Convert to batched DecisionNode type for stats.
            self._last_root = None
            return self._select_action_sequential(root)

        elif self.mode == MCTSMode.BATCHED:
            root = run_batched_mcts(
                game_state=state,
                network=self.network,
                num_simulations=self.num_simulations,
                batch_size=self.batch_size,
                exploration_weight=self.exploration_weight,
                add_exploration_noise=self.add_noise,
                dirichlet_alpha=self.dirichlet_alpha,
                noise_fraction=self.noise_fraction,
            )
            self._last_root = root
            return select_action_from_root(root, temperature=self.temperature)

        else:  # THREADED
            root = run_threaded_mcts(
                game_state=state,
                network=self.network,
                num_simulations=self.num_simulations,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                exploration_weight=self.exploration_weight,
                add_exploration_noise=self.add_noise,
                dirichlet_alpha=self.dirichlet_alpha,
                noise_fraction=self.noise_fraction,
            )
            self._last_root = root
            return select_action_from_root(root, temperature=self.temperature)

    def _select_action_sequential(self, root) -> int:
        """Select action from sequential MCTS root (different node type)."""
        return seq_select_action_from_root(root, temperature=self.temperature)

    def get_search_stats(self) -> SearchStats | None:
        """Get statistics from the last search."""
        if self._last_root is None:
            return None

        root = self._last_root
        visit_counts = {action: child.visit_count for action, child in root.children.items()}
        policy = get_policy_from_visits(root, temperature=1.0)
        selected = select_action_from_root(root, temperature=self.temperature)

        total_visits = sum(visit_counts.values())
        if total_visits > 0:
            root_value = sum(child.value_sum for child in root.children.values()) / total_visits
        else:
            root_value = root.value

        return SearchStats(root_value=root_value, visit_counts=visit_counts, policy=policy, selected_action=selected)

    def set_training_mode(self, training: bool = True) -> None:
        """Configure agent for training or evaluation."""
        if training:
            self.temperature = 1.0
            self.add_noise = True
        else:
            self.temperature = 0.0
            self.add_noise = False

    def set_mode(self, mode: MCTSMode) -> None:
        """Change the MCTS execution mode."""
        self.mode = mode
