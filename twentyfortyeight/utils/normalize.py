"""Normalize rewards using logarithmic scaling for stable learning across game stages."""

from numpy import log2


def normalize_reward(reward: float, max_tile: int = 2 ** (4**2)) -> float:
    """
    Normalize the reward using logarithmic scaling.

    This function compresses the range of rewards to make the learning process more
    stable across different stages of the game.

    Parameters
    ----------
    reward : float
        The raw reward obtained from a move.
    max_tile : int, optional
        The maximum tile value for the given board size, by default 2**(4**2).

    Returns
    -------
    float
        The normalized reward value between 0 and 1.

    Notes
    -----
    The normalization is based on the maximum theoretical tile value for the given board size.
    This logarithmic normalization helps in maintaining consistent reward scales throughout
    the game progression.
    """
    if reward == 0:
        return 0.0
    return log2(reward) / log2(max_tile)
