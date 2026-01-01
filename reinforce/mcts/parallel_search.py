"""
Parallel Monte Carlo Tree Search with batched neural network inference.

This module provides a parallelized version of MCTS that:
1. Collects multiple trajectories before making network calls
2. Batches all expansion requests into single network calls
3. Distributes backpropagation across collected trajectories

The key insight is that neural networks are throughput-optimized, not latency-optimized.
A single call with 32 states takes nearly the same time as a call with 1 state on GPU/TPU.

Reference: "Planning in Stochastic Environments with a Learned Model" (ICLR 2022)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from math import sqrt

from keras import utils
from numpy import argmax, ndarray, zeros
from numpy.random import PCG64DXSM, default_rng

from reinforce.neural.network import StochasticNetwork
from twentyfortyeight.core.gameboard import is_done
from twentyfortyeight.core.gamemove import legal_actions

GENERATOR = default_rng(PCG64DXSM())


class ExpansionType(Enum):
    """Type of node expansion required."""

    DECISION = auto()  # Need to expand a decision node (policy + afterstates)
    CHANCE = auto()  # Need to expand a chance node (Q-value + chance_probs)
    DYNAMICS = auto()  # Need to create a new decision child (dynamics transition)


@dataclass(kw_only=True)
class DecisionNode:
    """
    Decision node where the agent chooses an action.

    Same structure as network_search.py but with additional tracking
    for parallel expansion.
    """

    hidden_state: ndarray
    game_state: ndarray | None = None
    is_terminal: bool = False
    policy_prior: ndarray | None = None
    value: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    parent: ChanceNode | None = None
    children: dict[int, ChanceNode] = field(default_factory=dict)
    legal_moves: list[int] = field(default_factory=list)
    reward: float = 0.0

    def __post_init__(self):
        """Compute legal moves from game state if at root."""
        if self.game_state is not None and not self.is_terminal:
            self.legal_moves = legal_actions(self.game_state)

    @property
    def expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0

    @property
    def q_value(self) -> float:
        """Average value from visits."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(kw_only=True)
class ChanceNode:
    """Chance node representing stochastic environment transition."""

    afterstate: ndarray
    action: int
    prior: float = 1.0
    q_value: float = 0.0
    chance_probs: ndarray | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    parent: DecisionNode | None = None
    children: dict[int, DecisionNode] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        """Check if node has been expanded."""
        return self.chance_probs is not None


@dataclass
class ExpansionRequest:
    """
    Request to expand a node, collected for batching.

    Attributes
    ----------
    node : DecisionNode | ChanceNode
        The node to expand.
    request_type : ExpansionType
        What kind of expansion is needed.
    trajectory_idx : int
        Index of the trajectory this request belongs to.
    hidden_state : ndarray | None
        Hidden state for decision expansion.
    afterstate : ndarray | None
        Afterstate for chance/dynamics expansion.
    chance_code_idx : int | None
        Chance code index for dynamics expansion.
    """

    node: DecisionNode | ChanceNode
    request_type: ExpansionType
    trajectory_idx: int
    hidden_state: ndarray | None = None
    afterstate: ndarray | None = None
    chance_code_idx: int | None = None


@dataclass
class Trajectory:
    """
    A single MCTS trajectory from root to leaf.

    Attributes
    ----------
    path : list[DecisionNode | ChanceNode]
        Nodes visited during traversal.
    leaf_value : float | None
        Value at the leaf for backpropagation.
    needs_expansion : bool
        Whether this trajectory needs node expansion.
    """

    path: list[DecisionNode | ChanceNode] = field(default_factory=list)
    leaf_value: float | None = None
    needs_expansion: bool = False


class MinMaxStats:
    """Tracks min/max values for Q-value normalization."""

    def __init__(self):
        self.minimum = float('inf')
        self.maximum = float('-inf')

    def update(self, value: float) -> None:
        """Update min/max with new value."""
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range."""
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def puct_score(
    parent: DecisionNode, child: ChanceNode, exploration_weight: float, min_max_stats: MinMaxStats
) -> float:
    """Calculate PUCT score for action selection."""
    q_value = min_max_stats.normalize(child.value_sum / child.visit_count) if child.visit_count > 0 else 0.0
    pb_c = exploration_weight * sqrt(parent.visit_count) / (1 + child.visit_count)
    prior_score = pb_c * child.prior
    return q_value + prior_score


def select_action(node: DecisionNode, exploration_weight: float, min_max_stats: MinMaxStats) -> int:
    """Select action using PUCT formula."""
    best_action = -1
    best_score = float('-inf')
    for action, child in node.children.items():
        score = puct_score(node, child, exploration_weight, min_max_stats)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def select_chance_outcome(node: ChanceNode) -> int:
    """Select chance outcome using quasi-random sampling."""
    if node.chance_probs is None:
        raise ValueError('Cannot select from unexpanded chance node')
    visits = zeros(len(node.chance_probs))
    for code_idx, child in node.children.items():
        visits[code_idx] = child.visit_count
    scores = node.chance_probs / (1 + visits)
    return int(argmax(scores))


class ParallelMCTS:
    """
    Parallel MCTS with batched neural network inference.

    This class implements a parallelized version of MCTS that collects
    multiple trajectories before making batched network calls.

    Attributes
    ----------
    network : StochasticNetwork
        The neural network for predictions.
    num_simulations : int
        Total number of MCTS simulations to run.
    batch_size : int
        Number of trajectories to collect before batching.
    exploration_weight : float
        The c_puct exploration constant.

    Examples
    --------
    >>> mcts = ParallelMCTS(network, num_simulations=100, batch_size=16)
    >>> root = mcts.search(game_state)
    >>> policy = mcts.get_policy(root)
    """

    def __init__(
        self,
        network: StochasticNetwork,
        num_simulations: int = 100,
        batch_size: int = 16,
        exploration_weight: float = 1.25,
        add_exploration_noise: bool = True,
        dirichlet_alpha: float = 0.25,
        noise_fraction: float = 0.25,
    ):
        """
        Initialize parallel MCTS.

        Parameters
        ----------
        network : StochasticNetwork
            The neural network for predictions.
        num_simulations : int
            Total number of MCTS simulations.
        batch_size : int
            Number of trajectories to collect before batching.
        exploration_weight : float
            The c_puct exploration constant.
        add_exploration_noise : bool
            Whether to add Dirichlet noise at root.
        dirichlet_alpha : float
            Dirichlet noise alpha parameter.
        noise_fraction : float
            Fraction of noise to mix with priors.
        """
        self.network = network
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.exploration_weight = exploration_weight
        self.add_exploration_noise = add_exploration_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction

    def search(self, game_state: ndarray) -> DecisionNode:
        """
        Run parallel MCTS from a game state.

        Parameters
        ----------
        game_state : ndarray
            The current game board state (4x4 array).

        Returns
        -------
        DecisionNode
            The root node after search.
        """
        min_max_stats = MinMaxStats()

        # ##>: Create and expand root node.
        root = self._create_root(game_state)
        if root.is_terminal:
            return root

        self._expand_root(root, game_state)
        self._add_exploration_noise(root)

        # ##>: Run simulations in batches.
        simulations_done = 0
        while simulations_done < self.num_simulations:
            # ##>: Determine batch size for this iteration.
            remaining = self.num_simulations - simulations_done
            current_batch = min(self.batch_size, remaining)

            # ##>: Phase 1: Collect trajectories until expansion points.
            trajectories, expansion_requests = self._collect_trajectories(root, current_batch, min_max_stats)

            # ##>: Phase 2: Batch process all expansion requests.
            if expansion_requests:
                self._batch_expand(expansion_requests, trajectories)

            # ##>: Phase 3: Backpropagate all trajectories.
            for trajectory in trajectories:
                if trajectory.leaf_value is not None:
                    self._backpropagate(trajectory.path, trajectory.leaf_value, min_max_stats)

            simulations_done += len(trajectories)

        return root

    def _create_root(self, game_state: ndarray) -> DecisionNode:
        """Create the root node with hidden state encoding."""
        hidden_state = self.network.representation(game_state.flatten())
        root = DecisionNode(hidden_state=hidden_state, game_state=game_state)
        if is_done(game_state):
            root.is_terminal = True
        return root

    def _expand_root(self, root: DecisionNode, game_state: ndarray) -> None:
        """
        Expand the root node using batched afterstate computation.

        Uses batch_expand_all_actions to compute all 4 afterstates in one call.
        """
        # ##>: Get policy and value.
        output = self.network.prediction(root.hidden_state)
        root.policy_prior = output.policy
        root.value = output.value
        root.legal_moves = legal_actions(game_state)

        # ##>: Compute all afterstates in one batched call.
        all_afterstates = self.network.batch_expand_all_actions([root.hidden_state])[0]

        # ##>: Create chance node children.
        for action in root.legal_moves:
            afterstate = all_afterstates[action]
            prior = float(root.policy_prior[action]) if root.policy_prior is not None else 1.0 / len(root.legal_moves)
            child = ChanceNode(afterstate=afterstate, action=action, prior=prior, parent=root)
            root.children[action] = child

    def _add_exploration_noise(self, root: DecisionNode) -> None:
        """Add Dirichlet exploration noise to root priors."""
        if not self.add_exploration_noise or not root.legal_moves:
            return
        noise = GENERATOR.dirichlet([self.dirichlet_alpha] * len(root.legal_moves))
        for i, action in enumerate(root.legal_moves):
            child = root.children[action]
            child.prior = (1 - self.noise_fraction) * child.prior + self.noise_fraction * noise[i]

    def _collect_trajectories(
        self, root: DecisionNode, count: int, min_max_stats: MinMaxStats
    ) -> tuple[list[Trajectory], list[ExpansionRequest]]:
        """
        Collect multiple trajectories until they hit expansion points.

        Returns
        -------
        tuple[list[Trajectory], list[ExpansionRequest]]
            List of trajectories and their expansion requests.
        """
        trajectories = []
        expansion_requests = []

        for traj_idx in range(count):
            trajectory = Trajectory()
            node: DecisionNode | ChanceNode = root
            trajectory.path.append(node)

            while True:
                if isinstance(node, DecisionNode):
                    if node.is_terminal:
                        trajectory.leaf_value = 0.0
                        break

                    if not node.expanded and node is not root:
                        # ##>: Need to expand this decision node.
                        trajectory.needs_expansion = True
                        expansion_requests.append(
                            ExpansionRequest(
                                node=node,
                                request_type=ExpansionType.DECISION,
                                trajectory_idx=traj_idx,
                                hidden_state=node.hidden_state,
                            )
                        )
                        break

                    # ##>: Select action using PUCT.
                    action = select_action(node, self.exploration_weight, min_max_stats)
                    node = node.children[action]
                    trajectory.path.append(node)

                else:
                    # ##>: ChanceNode.
                    if not node.expanded:
                        # ##>: Need to expand chance node first.
                        trajectory.needs_expansion = True
                        expansion_requests.append(
                            ExpansionRequest(
                                node=node,
                                request_type=ExpansionType.CHANCE,
                                trajectory_idx=traj_idx,
                                afterstate=node.afterstate,
                            )
                        )
                        break

                    # ##>: Select outcome.
                    outcome_idx = select_chance_outcome(node)

                    if outcome_idx not in node.children:
                        # ##>: Need dynamics expansion to create child.
                        trajectory.needs_expansion = True
                        expansion_requests.append(
                            ExpansionRequest(
                                node=node,
                                request_type=ExpansionType.DYNAMICS,
                                trajectory_idx=traj_idx,
                                afterstate=node.afterstate,
                                chance_code_idx=outcome_idx,
                            )
                        )
                        break

                    # ##>: Continue to existing child.
                    node = node.children[outcome_idx]
                    trajectory.path.append(node)

            trajectories.append(trajectory)

        return trajectories, expansion_requests

    def _batch_expand(self, requests: list[ExpansionRequest], trajectories: list[Trajectory]) -> None:
        """
        Process all expansion requests in batched network calls.

        Groups requests by type and processes each group in a single batch.
        """
        # ##>: Group requests by type.
        decision_requests = [r for r in requests if r.request_type == ExpansionType.DECISION]
        chance_requests = [r for r in requests if r.request_type == ExpansionType.CHANCE]
        dynamics_requests = [r for r in requests if r.request_type == ExpansionType.DYNAMICS]

        # ##>: Process decision node expansions.
        if decision_requests:
            self._batch_expand_decision_nodes(decision_requests, trajectories)

        # ##>: Process chance node expansions.
        if chance_requests:
            self._batch_expand_chance_nodes(chance_requests, trajectories)

        # ##>: Process dynamics expansions.
        if dynamics_requests:
            self._batch_expand_dynamics(dynamics_requests, trajectories)

    def _batch_expand_decision_nodes(self, requests: list[ExpansionRequest], trajectories: list[Trajectory]) -> None:
        """Expand multiple decision nodes in batched calls."""
        # ##>: Filter out None hidden_states (should not happen for decision requests).
        hidden_states = [r.hidden_state for r in requests if r.hidden_state is not None]
        if len(hidden_states) != len(requests):
            raise ValueError('All decision requests must have hidden_state')

        # ##>: Batch prediction for policy and value.
        pred_output = self.network.batch_prediction(hidden_states)

        # ##>: Batch compute all afterstates.
        all_afterstates = self.network.batch_expand_all_actions(hidden_states)

        # ##>: Apply results to each node.
        for i, request in enumerate(requests):
            node = request.node
            assert isinstance(node, DecisionNode)

            if pred_output.policies is not None:
                node.policy_prior = pred_output.policies[i]
            node.value = float(pred_output.values[i])

            # ##>: Determine legal moves if not set.
            if not node.legal_moves:
                # ##>: For non-root nodes, assume all moves legal in latent space.
                node.legal_moves = list(range(4))

            # ##>: Create chance children.
            for action in node.legal_moves:
                afterstate = all_afterstates[i, action]
                prior = float(node.policy_prior[action]) if node.policy_prior is not None else 0.25
                child = ChanceNode(afterstate=afterstate, action=action, prior=prior, parent=node)
                node.children[action] = child

            # ##>: Set leaf value for this trajectory.
            trajectories[request.trajectory_idx].leaf_value = node.value

    def _batch_expand_chance_nodes(self, requests: list[ExpansionRequest], trajectories: list[Trajectory]) -> None:
        """Expand multiple chance nodes in batched calls."""
        afterstates = [r.afterstate for r in requests if r.afterstate is not None]
        if len(afterstates) != len(requests):
            raise ValueError('All chance requests must have afterstate')

        # ##>: Step 1: Batch afterstate prediction for Q-values and chance_probs.
        output = self.network.batch_afterstate_prediction(afterstates)

        # ##>: Step 2: Apply Q-values and chance_probs, then select outcomes.
        outcome_indices = []
        for i, request in enumerate(requests):
            node = request.node
            assert isinstance(node, ChanceNode)
            node.q_value = float(output.values[i])
            if output.chance_probs is not None:
                node.chance_probs = output.chance_probs[i]
            outcome_indices.append(select_chance_outcome(node))

        # ##>: Step 3: Batch dynamics call for all selected outcomes.
        chance_codes = [
            utils.to_categorical([idx], num_classes=self.network.codebook_size)[0] for idx in outcome_indices
        ]
        next_states, rewards = self.network.batch_dynamics(afterstates, chance_codes)

        # ##>: Step 4: Batch prediction for all new decision nodes.
        hidden_states_list = [next_states[i] for i in range(len(requests))]
        pred_output = self.network.batch_prediction(hidden_states_list)

        # ##>: Step 5: Batch afterstate expansion for all new nodes.
        all_afterstates = self.network.batch_expand_all_actions(hidden_states_list)

        # ##>: Step 6: Apply results to each node.
        for i, request in enumerate(requests):
            node = request.node
            assert isinstance(node, ChanceNode)

            child = DecisionNode(
                hidden_state=next_states[i], reward=float(rewards[i]), parent=node, legal_moves=list(range(4))
            )
            node.children[outcome_indices[i]] = child

            if pred_output.policies is not None:
                child.policy_prior = pred_output.policies[i]
            child.value = float(pred_output.values[i])

            # ##>: Create chance children.
            for action in child.legal_moves:
                afterstate = all_afterstates[i, action]
                prior = float(child.policy_prior[action]) if child.policy_prior is not None else 0.25
                grandchild = ChanceNode(afterstate=afterstate, action=action, prior=prior, parent=child)
                child.children[action] = grandchild

            trajectories[request.trajectory_idx].path.append(child)
            trajectories[request.trajectory_idx].leaf_value = child.value

    def _batch_expand_dynamics(self, requests: list[ExpansionRequest], trajectories: list[Trajectory]) -> None:
        """Create decision node children via batched dynamics calls."""
        afterstates = [r.afterstate for r in requests if r.afterstate is not None]
        chance_code_indices = [r.chance_code_idx for r in requests if r.chance_code_idx is not None]
        if len(afterstates) != len(requests) or len(chance_code_indices) != len(requests):
            raise ValueError('All dynamics requests must have afterstate and chance_code_idx')

        chance_codes = [
            utils.to_categorical([idx], num_classes=self.network.codebook_size)[0] for idx in chance_code_indices
        ]

        # ##>: Batch dynamics call.
        next_states, rewards = self.network.batch_dynamics(afterstates, chance_codes)

        # ##>: Batch prediction for new nodes.
        hidden_states_list = [next_states[i] for i in range(len(requests))]
        pred_output = self.network.batch_prediction(hidden_states_list)

        # ##>: Batch expand all afterstates for new nodes.
        all_afterstates = self.network.batch_expand_all_actions(hidden_states_list)

        # ##>: Apply results.
        for i, request in enumerate(requests):
            parent = request.node
            assert isinstance(parent, ChanceNode)

            # ##>: Create decision child.
            child = DecisionNode(
                hidden_state=next_states[i], reward=float(rewards[i]), parent=parent, legal_moves=list(range(4))
            )
            if request.chance_code_idx is not None:
                parent.children[request.chance_code_idx] = child

            # ##>: Apply prediction.
            if pred_output.policies is not None:
                child.policy_prior = pred_output.policies[i]
            child.value = float(pred_output.values[i])

            # ##>: Create chance children.
            for action in child.legal_moves:
                afterstate = all_afterstates[i, action]
                prior = float(child.policy_prior[action]) if child.policy_prior is not None else 0.25
                grandchild = ChanceNode(afterstate=afterstate, action=action, prior=prior, parent=child)
                child.children[action] = grandchild

            # ##>: Update trajectory.
            trajectories[request.trajectory_idx].path.append(child)
            trajectories[request.trajectory_idx].leaf_value = child.value

    def _backpropagate(self, path: list[DecisionNode | ChanceNode], value: float, min_max_stats: MinMaxStats) -> None:
        """Backpropagate value through the trajectory path."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            min_max_stats.update(value)

    def get_policy(self, root: DecisionNode, temperature: float = 1.0) -> dict[int, float]:
        """
        Compute action probabilities from visit counts.

        Parameters
        ----------
        root : DecisionNode
            The root node after search.
        temperature : float
            Temperature for softmax. 0 = argmax, 1 = proportional.

        Returns
        -------
        dict[int, float]
            Action probabilities keyed by action.
        """
        from numpy import exp

        visits = {action: child.visit_count for action, child in root.children.items()}

        if temperature == 0.0:
            best_action = max(visits, key=lambda a: visits[a])
            return {action: 1.0 if action == best_action else 0.0 for action in visits}

        total_visits = sum(visits.values())
        if total_visits == 0:
            return {action: 1.0 / len(visits) for action in visits}

        visit_counts = list(visits.values())
        max_count = max(visit_counts)

        probs = {}
        exp_sum = 0.0
        for action, count in visits.items():
            exp_val = exp((count - max_count) / temperature)
            probs[action] = exp_val
            exp_sum += exp_val

        return {action: p / exp_sum for action, p in probs.items()}

    def select_action(self, root: DecisionNode, temperature: float = 0.0) -> int:
        """
        Select an action from the root node.

        Parameters
        ----------
        root : DecisionNode
            The root node after search.
        temperature : float
            Selection temperature.

        Returns
        -------
        int
            The selected action.
        """
        if not root.children:
            raise ValueError('Cannot select from root with no children')

        policy = self.get_policy(root, temperature)

        if temperature == 0.0:
            best_action = -1
            best_visits = -1
            best_q = float('-inf')

            for action, child in root.children.items():
                if child.visit_count > best_visits or (
                    child.visit_count == best_visits and child.value_sum / max(1, child.visit_count) > best_q
                ):
                    best_visits = child.visit_count
                    best_q = child.value_sum / max(1, child.visit_count)
                    best_action = action
            return best_action

        actions = list(policy.keys())
        probs = [policy[a] for a in actions]
        return int(GENERATOR.choice(actions, p=probs))


def run_parallel_mcts(
    game_state: ndarray,
    network: StochasticNetwork,
    num_simulations: int = 100,
    batch_size: int = 16,
    exploration_weight: float = 1.25,
    add_exploration_noise: bool = True,
    dirichlet_alpha: float = 0.25,
    noise_fraction: float = 0.25,
) -> DecisionNode:
    """
    Run parallel MCTS from a game state.

    Convenience function that creates a ParallelMCTS instance and runs search.

    Parameters
    ----------
    game_state : ndarray
        The current game board state.
    network : StochasticNetwork
        The Stochastic MuZero neural network.
    num_simulations : int
        Number of MCTS simulations.
    batch_size : int
        Number of trajectories to batch together.
    exploration_weight : float
        The c_puct exploration constant.
    add_exploration_noise : bool
        Whether to add Dirichlet noise at root.
    dirichlet_alpha : float
        Dirichlet distribution alpha parameter.
    noise_fraction : float
        Fraction of noise to mix with priors.

    Returns
    -------
    DecisionNode
        The root node after search.
    """
    mcts = ParallelMCTS(
        network=network,
        num_simulations=num_simulations,
        batch_size=batch_size,
        exploration_weight=exploration_weight,
        add_exploration_noise=add_exploration_noise,
        dirichlet_alpha=dirichlet_alpha,
        noise_fraction=noise_fraction,
    )
    return mcts.search(game_state)
