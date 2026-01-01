"""
Loss functions for Stochastic MuZero training.

Implements the losses from Equations 1, 6, and 7 of the paper:
- L^MuZero = Σ l^p(π, p) + Σ l^v(z, v) + Σ l^r(u, r)
- L^chance = Σ l^Q(z, Q) + Σ l^σ(c, σ) + β * commitment_cost
- L^total = L^MuZero + L^chance
"""

from __future__ import annotations

from numpy import ndarray, sign, sqrt


def scalar_to_support(
    scalar: float,
    support_min: float,
    support_max: float,
    support_size: int,
) -> ndarray:
    """
    Convert a scalar to a categorical distribution over support.

    Uses the two-hot encoding from MuZero: distribute probability between
    the two nearest support points.

    Parameters
    ----------
    scalar : float
        The scalar value to encode.
    support_min : float
        Minimum value of the support.
    support_max : float
        Maximum value of the support.
    support_size : int
        Number of bins in the support.

    Returns
    -------
    ndarray
        Categorical distribution over the support.
    """
    from numpy import clip, zeros

    # ##>: Clip to support range.
    scalar = clip(scalar, support_min, support_max)

    # ##>: Compute position in support.
    support_range = support_max - support_min
    position = (scalar - support_min) / support_range * (support_size - 1)

    # ##>: Two-hot encoding.
    lower = int(position)
    upper = lower + 1
    upper_weight = position - lower
    lower_weight = 1.0 - upper_weight

    distribution = zeros(support_size)
    distribution[lower] = lower_weight
    if upper < support_size:
        distribution[upper] = upper_weight

    return distribution


def support_to_scalar(
    distribution: ndarray,
    support_min: float,
    support_max: float,
    support_size: int,
) -> float:
    """
    Convert a categorical distribution back to a scalar.

    Parameters
    ----------
    distribution : ndarray
        Categorical distribution over the support.
    support_min : float
        Minimum value of the support.
    support_max : float
        Maximum value of the support.
    support_size : int
        Number of bins in the support.

    Returns
    -------
    float
        The scalar value.
    """
    from numpy import arange
    from numpy import sum as np_sum

    support = arange(support_size)
    support_values = support_min + support * (support_max - support_min) / (support_size - 1)
    return float(np_sum(distribution * support_values))


def value_transform(x: float, epsilon: float = 0.001) -> float:
    """
    Apply the invertible transform h(x) to scale values.

    From MuZero: h(x) = sign(x) * (sqrt(|x| + 1) - 1 + ε*x)
    This compresses large values while preserving sign.

    Parameters
    ----------
    x : float
        Value to transform.
    epsilon : float
        Small constant (paper: 0.001).

    Returns
    -------
    float
        Transformed value.
    """
    return sign(x) * (sqrt(abs(x) + 1) - 1 + epsilon * x)


def inverse_value_transform(x: float, epsilon: float = 0.001) -> float:
    """
    Apply inverse transform h^{-1}(x).

    Parameters
    ----------
    x : float
        Transformed value.
    epsilon : float
        Small constant (paper: 0.001).

    Returns
    -------
    float
        Original value.
    """
    # ##>: h^{-1}(x) = sign(x) * ((sqrt(1 + 4ε(|x| + 1 + ε)) - 1) / (2ε))^2 - 1
    if abs(x) < 1e-8:
        return 0.0
    sign_x = sign(x)
    abs_x = abs(x)
    inner = sqrt(1 + 4 * epsilon * (abs_x + 1 + epsilon)) - 1
    return sign_x * (((inner / (2 * epsilon)) ** 2) - 1)


def compute_policy_loss(predicted_policy: ndarray, target_policy: ndarray) -> float:
    """
    Compute cross-entropy loss for policy prediction.

    l^p(π, p) = -π^T log(p)

    Parameters
    ----------
    predicted_policy : ndarray
        Predicted policy probabilities from network.
    target_policy : ndarray
        Target policy from MCTS search (visit distribution).

    Returns
    -------
    float
        Cross-entropy loss.
    """
    from numpy import clip, log
    from numpy import sum as np_sum

    # ##>: Clip for numerical stability.
    predicted_policy = clip(predicted_policy, 1e-8, 1.0)
    return -float(np_sum(target_policy * log(predicted_policy)))


def compute_value_loss(
    predicted_value: float,
    target_value: float,
    use_categorical: bool = True,
    support_min: float = 0.0,
    support_max: float = 600.0,
    support_size: int = 601,
    epsilon: float = 0.001,
) -> float:
    """
    Compute loss for value prediction.

    For 2048, uses categorical representation with value transform:
    l^v(z, v) = Cat(h(z))^T log(v)

    Parameters
    ----------
    predicted_value : float | ndarray
        Predicted value (scalar or distribution).
    target_value : float
        Target value (scalar).
    use_categorical : bool
        Whether to use categorical representation.
    support_min : float
        Minimum support value.
    support_max : float
        Maximum support value.
    support_size : int
        Number of support bins.
    epsilon : float
        Value transform epsilon.

    Returns
    -------
    float
        Value loss.
    """
    if use_categorical:
        # ##>: Transform target and convert to categorical.
        transformed_target = value_transform(target_value, epsilon)
        target_dist = scalar_to_support(transformed_target, support_min, support_max, support_size)

        # ##>: Cross-entropy between distributions.
        from numpy import clip, log
        from numpy import sum as np_sum

        predicted_value = clip(predicted_value, 1e-8, 1.0)
        return -float(np_sum(target_dist * log(predicted_value)))
    else:
        # ##>: Simple MSE loss.
        return (predicted_value - target_value) ** 2


def compute_reward_loss(
    predicted_reward: float,
    target_reward: float,
    use_categorical: bool = True,
    support_min: float = 0.0,
    support_max: float = 600.0,
    support_size: int = 601,
    epsilon: float = 0.001,
) -> float:
    """
    Compute loss for reward prediction.

    Same as value loss but for intermediate rewards.

    Parameters
    ----------
    predicted_reward : float | ndarray
        Predicted reward.
    target_reward : float
        Target reward.
    use_categorical : bool
        Whether to use categorical representation.
    support_min : float
        Minimum support value.
    support_max : float
        Maximum support value.
    support_size : int
        Number of support bins.
    epsilon : float
        Value transform epsilon.

    Returns
    -------
    float
        Reward loss.
    """
    return compute_value_loss(
        predicted_reward,
        target_reward,
        use_categorical=use_categorical,
        support_min=support_min,
        support_max=support_max,
        support_size=support_size,
        epsilon=epsilon,
    )


def compute_chance_loss(
    predicted_chance_probs: ndarray,
    target_chance_code: ndarray,
) -> float:
    """
    Compute cross-entropy loss for chance distribution prediction.

    l^σ(c, σ) = -log σ(c)

    Parameters
    ----------
    predicted_chance_probs : ndarray
        Predicted chance distribution σ from afterstate prediction.
    target_chance_code : ndarray
        Target one-hot chance code from encoder.

    Returns
    -------
    float
        Cross-entropy loss.
    """
    from numpy import clip, log
    from numpy import sum as np_sum

    predicted_chance_probs = clip(predicted_chance_probs, 1e-8, 1.0)
    return -float(np_sum(target_chance_code * log(predicted_chance_probs)))


def compute_q_value_loss(
    predicted_q: float,
    target_value: float,
    use_categorical: bool = True,
    support_min: float = 0.0,
    support_max: float = 600.0,
    support_size: int = 601,
    epsilon: float = 0.001,
) -> float:
    """
    Compute loss for Q-value prediction at afterstates.

    Same as value loss - Q^k is trained towards z_{t+k}.

    Parameters
    ----------
    predicted_q : float | ndarray
        Predicted Q-value.
    target_value : float
        Target value.
    use_categorical : bool
        Whether to use categorical representation.
    support_min : float
        Minimum support value.
    support_max : float
        Maximum support value.
    support_size : int
        Number of support bins.
    epsilon : float
        Value transform epsilon.

    Returns
    -------
    float
        Q-value loss.
    """
    return compute_value_loss(
        predicted_q,
        target_value,
        use_categorical=use_categorical,
        support_min=support_min,
        support_max=support_max,
        support_size=support_size,
        epsilon=epsilon,
    )


def compute_commitment_loss(
    encoder_output: ndarray,
    chance_code: ndarray,
) -> float:
    """
    Compute VQ-VAE commitment loss.

    ||c - c^e||^2 where c is the quantized code and c^e is encoder output.

    Parameters
    ----------
    encoder_output : ndarray
        Raw encoder output (logits before argmax).
    chance_code : ndarray
        Quantized one-hot chance code.

    Returns
    -------
    float
        Commitment loss.
    """
    from numpy import sum as np_sum

    return float(np_sum((chance_code - encoder_output) ** 2))


def compute_total_loss(
    policy_losses: list[float],
    value_losses: list[float],
    reward_losses: list[float],
    q_value_losses: list[float],
    chance_losses: list[float],
    commitment_losses: list[float],
    commitment_weight: float = 0.25,
) -> dict[str, float]:
    """
    Compute total Stochastic MuZero loss.

    L^total = L^MuZero + L^chance
    L^MuZero = Σ l^p + Σ l^v + Σ l^r
    L^chance = Σ l^Q + Σ l^σ + β * Σ commitment

    Parameters
    ----------
    policy_losses : list[float]
        Policy losses for each timestep.
    value_losses : list[float]
        Value losses for each timestep.
    reward_losses : list[float]
        Reward losses for each timestep (k >= 1).
    q_value_losses : list[float]
        Q-value losses for each afterstate.
    chance_losses : list[float]
        Chance distribution losses.
    commitment_losses : list[float]
        VQ-VAE commitment losses.
    commitment_weight : float
        Weight for commitment loss (β).

    Returns
    -------
    dict[str, float]
        Dictionary with individual and total losses.
    """
    policy_total = sum(policy_losses)
    value_total = sum(value_losses)
    reward_total = sum(reward_losses)
    q_value_total = sum(q_value_losses)
    chance_total = sum(chance_losses)
    commitment_total = sum(commitment_losses) * commitment_weight

    muzero_loss = policy_total + value_total + reward_total
    chance_loss = q_value_total + chance_total + commitment_total
    total_loss = muzero_loss + chance_loss

    return {
        'total': total_loss,
        'muzero': muzero_loss,
        'chance': chance_loss,
        'policy': policy_total,
        'value': value_total,
        'reward': reward_total,
        'q_value': q_value_total,
        'chance_distribution': chance_total,
        'commitment': commitment_total,
    }
