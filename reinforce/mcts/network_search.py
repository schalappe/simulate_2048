"""
Network-guided Monte Carlo Tree Search for Stochastic MuZero.

This module provides MCTS that uses neural network predictions for:
- Policy priors at decision nodes (replaces uniform 1/N)
- Q-values at chance nodes (replaces random rollouts)
- Chance distribution σ for stochastic outcome selection

The implementation follows the Stochastic MuZero paper (ICLR 2022):
- Decision nodes use PUCT with network policy priors
- Chance nodes use quasi-random sampling: argmax_c [σ(c|as) / (N(c)+1)]
- No random rollouts - value estimates come directly from the network
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from keras import utils
from numpy import argmax, exp, ndarray, zeros
from numpy.random import PCG64DXSM, default_rng

from reinforce.neural.network import StochasticNetwork
from twentyfortyeight.core.gameboard import is_done
from twentyfortyeight.core.gamemove import legal_actions

GENERATOR = default_rng(PCG64DXSM())


@dataclass(kw_only=True)
class DecisionNode:
    """
    Decision node where the agent chooses an action.

    In Stochastic MuZero, decision nodes correspond to states s^k where
    the agent must select an action. The policy prior comes from the
    prediction network f(s^k) -> (policy, value).

    Attributes
    ----------
    hidden_state : ndarray
        The latent hidden state from the network (s^k).
    game_state : ndarray | None
        The actual game board state (only at root, None elsewhere).
    is_terminal : bool
        Whether this is a terminal state.
    policy_prior : ndarray
        Policy probabilities from the network for each action.
    value : float
        Value estimate v^k from the network.
    visit_count : int
        Number of times this node has been visited.
    value_sum : float
        Sum of backpropagated values.
    parent : ChanceNode | None
        The parent chance node (None for root).
    children : dict[int, ChanceNode]
        Child chance nodes keyed by action.
    legal_moves : list[int]
        List of legal actions from this state.
    reward : float
        Reward received transitioning to this state.
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
        """Check if node has been expanded (children created)."""
        return len(self.children) > 0

    @property
    def q_value(self) -> float:
        """Average value from visits."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(kw_only=True)
class ChanceNode:
    """
    Chance node representing stochastic environment transition.

    In Stochastic MuZero, chance nodes correspond to afterstates as^k.
    They represent the state after an action but before the environment's
    stochastic response (tile spawn in 2048).

    Attributes
    ----------
    afterstate : ndarray
        The latent afterstate from the network (as^k).
    action : int
        The action that led to this afterstate.
    prior : float
        Prior probability of selecting this action (from parent's policy).
    q_value : float
        Q-value estimate Q^k from afterstate prediction.
    chance_probs : ndarray
        Distribution σ^k over chance outcomes from the network.
    visit_count : int
        Number of times this node has been visited.
    value_sum : float
        Sum of backpropagated values.
    parent : DecisionNode
        The parent decision node.
    children : dict[int, DecisionNode]
        Child decision nodes keyed by chance code index.
    """

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
        """Check if node has been expanded (predictions computed)."""
        return self.chance_probs is not None


class MinMaxStats:
    """
    Tracks min/max values for Q-value normalization in PUCT.

    This ensures Q-values are normalized to [0, 1] range for fair comparison
    with prior probabilities in the PUCT formula.
    """

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
    """
    Calculate PUCT score for action selection.

    Uses the AlphaZero/MuZero PUCT formula:
    score = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

    Parameters
    ----------
    parent : DecisionNode
        The parent decision node.
    child : ChanceNode
        The child chance node to score.
    exploration_weight : float
        The c_puct exploration constant.
    min_max_stats : MinMaxStats
        For Q-value normalization.

    Returns
    -------
    float
        The PUCT score.
    """
    # ##>: Q-value component (exploitation).
    q_value = min_max_stats.normalize(child.value_sum / child.visit_count) if child.visit_count > 0 else 0.0

    # ##>: Prior component (exploration).
    pb_c = exploration_weight * sqrt(parent.visit_count) / (1 + child.visit_count)
    prior_score = pb_c * child.prior

    return q_value + prior_score


def select_action(node: DecisionNode, exploration_weight: float, min_max_stats: MinMaxStats) -> int:
    """
    Select action using PUCT formula.

    Parameters
    ----------
    node : DecisionNode
        The decision node to select from.
    exploration_weight : float
        The c_puct constant.
    min_max_stats : MinMaxStats
        For Q-value normalization.

    Returns
    -------
    int
        The selected action.

    Raises
    ------
    ValueError
        If node has no children.
    """
    if not node.children:
        raise ValueError('Cannot select action from node with no children')

    best_action = -1
    best_score = float('-inf')

    for action, child in node.children.items():
        score = puct_score(node, child, exploration_weight, min_max_stats)
        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def select_chance_outcome(node: ChanceNode) -> int:
    """
    Select chance outcome using quasi-random sampling.

    Uses the formula from the paper: argmax_c [σ(c|as) / (N(c)+1)]
    This balances exploration (less visited) with prior probability.

    Parameters
    ----------
    node : ChanceNode
        The chance node to select from.

    Returns
    -------
    int
        The selected chance code index.

    Raises
    ------
    ValueError
        If chance_probs is None (node not expanded).
    """
    if node.chance_probs is None:
        raise ValueError('Cannot select chance outcome from unexpanded node')

    # ##>: Compute score for each possible outcome.
    scores = node.chance_probs / (1 + get_outcome_visits(node))
    return int(argmax(scores))


def get_outcome_visits(node: ChanceNode) -> ndarray:
    """
    Get visit counts for all possible outcomes.

    Parameters
    ----------
    node : ChanceNode
        The chance node (must have chance_probs set).

    Returns
    -------
    ndarray
        Visit counts per outcome (0 for unexplored).

    Raises
    ------
    ValueError
        If chance_probs is None.
    """
    if node.chance_probs is None:
        raise ValueError('Cannot get outcome visits from node without chance_probs')

    visits = zeros(len(node.chance_probs))
    for code_idx, child in node.children.items():
        visits[code_idx] = child.visit_count
    return visits


def expand_decision_node(node: DecisionNode, network: StochasticNetwork, game_state: ndarray | None = None) -> None:
    """
    Expand a decision node by computing policy and creating chance node children.

    Parameters
    ----------
    node : DecisionNode
        The decision node to expand.
    network : StochasticNetwork
        The neural network for predictions.
    game_state : ndarray | None
        Game state if at root (used to compute legal moves).
    """
    # ##>: Get policy and value from prediction network.
    output = network.prediction(node.hidden_state)
    node.policy_prior = output.policy
    node.value = output.value

    # ##>: Determine legal moves.
    if game_state is not None:
        node.game_state = game_state
        node.legal_moves = legal_actions(game_state)

    # ##>: Create chance node children for each legal action.
    for action in node.legal_moves:
        # ##>: Compute afterstate using network.
        afterstate = network.afterstate_dynamics(node.hidden_state, action)

        # ##>: Prior from the network policy (masked to legal moves).
        prior = float(node.policy_prior[action]) if node.policy_prior is not None else 1.0 / len(node.legal_moves)

        child = ChanceNode(afterstate=afterstate, action=action, prior=prior, parent=node)
        node.children[action] = child


def expand_chance_node(node: ChanceNode, network: StochasticNetwork) -> None:
    """
    Expand a chance node by computing Q-value and chance distribution.

    Parameters
    ----------
    node : ChanceNode
        The chance node to expand.
    network : StochasticNetwork
        The neural network for predictions.
    """
    output = network.afterstate_prediction(node.afterstate)
    node.q_value = output.value
    node.chance_probs = output.chance_probs


def add_decision_child(parent: ChanceNode, chance_code_idx: int, network: StochasticNetwork) -> DecisionNode:
    """
    Add a decision node child to a chance node.

    Parameters
    ----------
    parent : ChanceNode
        The parent chance node.
    chance_code_idx : int
        The index of the selected chance code.
    network : StochasticNetwork
        The neural network.

    Returns
    -------
    DecisionNode
        The newly created decision node.
    """
    chance_code = utils.to_categorical([chance_code_idx], num_classes=network.codebook_size)[0]

    # ##>: Compute next state and reward using dynamics network.
    hidden_state, reward = network.dynamics(parent.afterstate, chance_code)

    # ##>: Create decision node.
    child = DecisionNode(hidden_state=hidden_state, reward=reward, parent=parent)
    parent.children[chance_code_idx] = child

    return child


def backpropagate(node: DecisionNode | ChanceNode, value: float, min_max_stats: MinMaxStats) -> None:
    """
    Backpropagate value through the tree.

    Parameters
    ----------
    node : DecisionNode | ChanceNode
        Starting node for backpropagation.
    value : float
        The value to backpropagate.
    min_max_stats : MinMaxStats
        For tracking value bounds.
    """
    current: DecisionNode | ChanceNode | None = node
    while current is not None:
        current.visit_count += 1
        current.value_sum += value
        min_max_stats.update(value)
        current = current.parent


def run_network_mcts(
    game_state: ndarray,
    network: StochasticNetwork,
    num_simulations: int = 100,
    exploration_weight: float = 1.25,
    add_exploration_noise: bool = True,
    dirichlet_alpha: float = 0.25,
    noise_fraction: float = 0.25,
) -> DecisionNode:
    """
    Run network-guided MCTS from a game state.

    Parameters
    ----------
    game_state : ndarray
        The current game board state.
    network : StochasticNetwork
        The Stochastic MuZero neural network.
    num_simulations : int
        Number of MCTS simulations to run.
    exploration_weight : float
        The c_puct exploration constant.
    add_exploration_noise : bool
        Whether to add Dirichlet noise to root priors.
    dirichlet_alpha : float
        Dirichlet distribution alpha parameter.
    noise_fraction : float
        Fraction of noise to mix with priors.

    Returns
    -------
    DecisionNode
        The root node after search.
    """
    min_max_stats = MinMaxStats()

    # ##>: Create root node.
    hidden_state = network.representation(game_state.flatten())
    root = DecisionNode(hidden_state=hidden_state, game_state=game_state)

    # ##>: Check terminal state.
    if is_done(game_state):
        root.is_terminal = True
        return root

    # ##>: Expand root.
    expand_decision_node(root, network, game_state)

    # ##>: Add exploration noise to root priors.
    if add_exploration_noise and len(root.legal_moves) > 0:
        noise = GENERATOR.dirichlet([dirichlet_alpha] * len(root.legal_moves))
        for i, action in enumerate(root.legal_moves):
            child = root.children[action]
            child.prior = (1 - noise_fraction) * child.prior + noise_fraction * noise[i]

    # ##>: Run simulations.
    for _ in range(num_simulations):
        node: DecisionNode | ChanceNode = root
        search_path: list[DecisionNode | ChanceNode] = [node]

        # ##>: Selection: traverse tree until we find a node to expand.
        while True:
            if isinstance(node, DecisionNode):
                if node.is_terminal:
                    # ##>: Terminal node - backpropagate 0.
                    backpropagate(node, 0.0, min_max_stats)
                    break

                if not node.expanded:
                    # ##>: Expand unexpanded decision node.
                    expand_decision_node(node, network)
                    backpropagate(node, node.value, min_max_stats)
                    break

                # ##>: Select action using PUCT.
                action = select_action(node, exploration_weight, min_max_stats)
                node = node.children[action]
                search_path.append(node)

            else:
                # ##>: Chance node.
                if not node.expanded:
                    # ##>: Expand chance node (compute σ distribution).
                    expand_chance_node(node, network)

                # ##>: Select outcome using quasi-random sampling.
                outcome_idx = select_chance_outcome(node)

                if outcome_idx not in node.children:
                    # ##>: Create new decision child.
                    child = add_decision_child(node, outcome_idx, network)
                    search_path.append(child)

                    # ##>: Expand and evaluate the new node.
                    expand_decision_node(child, network)
                    backpropagate(child, child.value, min_max_stats)
                    break
                else:
                    # ##>: Continue to existing child.
                    node = node.children[outcome_idx]
                    search_path.append(node)

    return root


def get_policy_from_visits(root: DecisionNode, temperature: float = 1.0) -> dict[int, float]:
    """
    Compute action probabilities from visit counts.

    Parameters
    ----------
    root : DecisionNode
        The root node after search.
    temperature : float
        Temperature for softmax. 0 = argmax, 1 = proportional to visits.

    Returns
    -------
    dict[int, float]
        Action probabilities keyed by action.
    """
    visits = {action: child.visit_count for action, child in root.children.items()}

    if temperature == 0.0:
        # ##>: Argmax selection.
        best_action = max(visits, key=lambda a: visits[a])
        return {action: 1.0 if action == best_action else 0.0 for action in visits}

    # ##>: Softmax with temperature.
    total_visits = sum(visits.values())
    if total_visits == 0:
        # ##>: Uniform if no visits.
        return {action: 1.0 / len(visits) for action in visits}

    # ##>: Apply temperature.
    visit_counts = list(visits.values())
    max_count = max(visit_counts)

    # ##>: Stable softmax with temperature.
    probs = {}
    exp_sum = 0.0
    for action, count in visits.items():
        exp_val = exp((count - max_count) / temperature)
        probs[action] = exp_val
        exp_sum += exp_val

    return {action: p / exp_sum for action, p in probs.items()}


def select_action_from_root(root: DecisionNode, temperature: float = 0.0) -> int:
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

    Raises
    ------
    ValueError
        If root has no children.
    """
    if not root.children:
        raise ValueError('Cannot select action from root with no children')

    policy = get_policy_from_visits(root, temperature)

    if temperature == 0.0:
        # ##>: Deterministic selection (most visits, Q-value tiebreaker).
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

    # ##>: Sample from policy.
    actions = list(policy.keys())
    probs = [policy[a] for a in actions]
    return int(GENERATOR.choice(actions, p=probs))
