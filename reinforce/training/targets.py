"""
Target computation for Stochastic MuZero training.

Implements n-step TD(λ) returns for value targets.
"""

from __future__ import annotations

from numpy import ndarray, zeros

from .replay_buffer import TransitionData


def compute_n_step_return(
    rewards: list[float],
    discounts: list[float],
    bootstrap_value: float,
    n_steps: int,
) -> float:
    """
    Compute n-step return: R_t = Σ γ^k r_{t+k} + γ^n v(s_{t+n}).

    Parameters
    ----------
    rewards : list[float]
        Rewards from t+1 to t+n.
    discounts : list[float]
        Discount factors (0 if terminal).
    bootstrap_value : float
        Value estimate at s_{t+n}.
    n_steps : int
        Number of steps to accumulate.

    Returns
    -------
    float
        The n-step return.
    """
    value = bootstrap_value
    cumulative_discount = 1.0

    # ##>: Compute cumulative discount for bootstrapping.
    for i in range(min(n_steps, len(discounts))):
        cumulative_discount *= discounts[i]

    value *= cumulative_discount

    # ##>: Add discounted rewards.
    cumulative_discount = 1.0
    for i in range(min(n_steps, len(rewards))):
        value += cumulative_discount * rewards[i]
        if i < len(discounts):
            cumulative_discount *= discounts[i]

    return value


def compute_td_lambda_targets(
    transitions: list[TransitionData],
    n_steps: int,
    td_lambda: float,
    discount: float,  # noqa: ARG001 - kept for API compatibility
) -> list[float]:
    """
    Compute TD(λ) value targets for a trajectory.

    TD(λ) combines n-step returns with exponential weighting:
    G_t^λ = (1-λ) Σ λ^{n-1} G_t^{(n)} + λ^{N-1} G_t^{(N)}

    For 2048 paper: n=10, λ=0.5.

    Parameters
    ----------
    transitions : list[TransitionData]
        List of transitions in the trajectory.
    n_steps : int
        Maximum number of steps for returns.
    td_lambda : float
        TD(λ) weighting parameter.
    discount : float
        Base discount factor γ.

    Returns
    -------
    list[float]
        Value targets for each position.
    """
    num_transitions = len(transitions)
    targets = []

    for t in range(num_transitions):
        # ##>: Collect rewards and discounts from position t.
        rewards = []
        discounts = []
        for k in range(min(n_steps, num_transitions - t)):
            if t + k < num_transitions:
                rewards.append(transitions[t + k].reward)
                discounts.append(transitions[t + k].discount)

        # ##>: Get bootstrap value from the end of the window.
        bootstrap_idx = min(t + n_steps, num_transitions - 1)
        bootstrap_value = transitions[bootstrap_idx].search_value

        if td_lambda == 1.0:
            # ##>: Pure Monte Carlo return.
            target = compute_n_step_return(rewards, discounts, bootstrap_value, n_steps)
        elif td_lambda == 0.0:
            # ##>: Pure 1-step TD.
            target = compute_n_step_return(rewards[:1], discounts[:1], bootstrap_value, 1)
        else:
            # ##>: TD(λ) weighted combination.
            target = 0.0
            remaining_weight = 1.0

            for n in range(1, min(n_steps, len(rewards)) + 1):
                # ##>: Bootstrap value for n-step return.
                n_bootstrap = transitions[t + n].search_value if t + n < num_transitions else 0.0

                n_return = compute_n_step_return(rewards[:n], discounts[:n], n_bootstrap, n)

                if n < n_steps and t + n < num_transitions:
                    # ##>: Weight for this n-step return.
                    weight = (1 - td_lambda) * (td_lambda ** (n - 1))
                    target += weight * n_return
                    remaining_weight -= weight
                else:
                    # ##>: Final return gets remaining weight.
                    target += remaining_weight * n_return
                    break

        targets.append(target)

    return targets


def compute_policy_target(search_policy: ndarray, action_size: int) -> ndarray:
    """
    Convert search policy dict to full action-size array.

    Parameters
    ----------
    search_policy : ndarray
        Policy from MCTS (may be sparse).
    action_size : int
        Total number of actions.

    Returns
    -------
    ndarray
        Full policy array of size action_size.
    """
    # ##>: search_policy should already be the right shape.
    if len(search_policy) == action_size:
        return search_policy

    # ##>: If sparse, expand to full size.
    full_policy = zeros(action_size)
    for i, p in enumerate(search_policy):
        if i < action_size:
            full_policy[i] = p
    return full_policy


def prepare_training_targets(
    transitions: list[TransitionData],
    start_idx: int,
    unroll_steps: int,
    n_steps: int,
    td_lambda: float,
    discount: float,
    action_size: int,
) -> dict[str, list]:
    """
    Prepare all training targets for a trajectory segment.

    Parameters
    ----------
    transitions : list[TransitionData]
        Full trajectory.
    start_idx : int
        Starting position.
    unroll_steps : int
        Number of steps to unroll.
    n_steps : int
        TD steps.
    td_lambda : float
        TD(λ) parameter.
    discount : float
        Discount factor.
    action_size : int
        Number of actions.

    Returns
    -------
    dict[str, list]
        Dictionary with observations, actions, policy targets, value targets, reward targets.
    """
    # ##>: Extract the relevant segment.
    end_idx = min(start_idx + unroll_steps + n_steps, len(transitions))
    segment = transitions[start_idx:end_idx]

    # ##>: Compute value targets for the segment.
    value_targets = compute_td_lambda_targets(segment, n_steps, td_lambda, discount)

    # ##>: Prepare data for unroll_steps.
    observations = []
    actions = []
    policy_targets = []
    rewards = []

    for k in range(min(unroll_steps + 1, len(segment))):
        t = segment[k]
        observations.append(t.observation)
        actions.append(t.action)
        policy_targets.append(compute_policy_target(t.search_policy, action_size))
        rewards.append(t.reward)

    return {
        'observations': observations,
        'actions': actions,
        'policy_targets': policy_targets,
        'value_targets': value_targets[: len(observations)],
        'reward_targets': rewards,
    }
