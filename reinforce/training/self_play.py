"""
Self-play trajectory generation for Stochastic MuZero.

This module provides functions to play games using MCTS and collect trajectories for training.
Supports both single-game and batched parallel game generation.
"""

from typing import NamedTuple

import jax
import numpy as np
from tqdm import tqdm

from reinforce.game.core import encode_observation, is_done, legal_actions_mask, max_tile, next_state
from reinforce.game.env import GameState, reset
from reinforce.mcts.policy import get_policy_target, get_search_value, select_action
from reinforce.mcts.stochastic_mctx import NetworkApplyFns, NetworkParams, run_mcts_jit
from reinforce.training.config import TrainConfig
from reinforce.training.replay_buffer import Trajectory

# ##>: Type aliases.
Array = jax.Array
PRNGKey = jax.Array


class StepData(NamedTuple):
    """Data collected at each game step."""

    observation: Array
    action: int
    reward: float
    policy: Array
    value: float
    done: bool


def play_game(
    params: NetworkParams,
    apply_fns: NetworkApplyFns,
    key: PRNGKey,
    config: TrainConfig,
    training_step: int = 0,
) -> Trajectory:
    """
    Play a complete game using MCTS and return the trajectory.

    Parameters
    ----------
    params : NetworkParams
        Network parameters.
    apply_fns : NetworkApplyFns
        Network apply functions.
    key : PRNGKey
        JAX random key.
    config : TrainConfig
        Training configuration.
    training_step : int
        Current training step (for temperature scheduling).

    Returns
    -------
    Trajectory
        Complete game trajectory with MCTS statistics.
    """
    # ##>: Get temperature for current training step.
    temperature = config.get_temperature(training_step)

    # ##>: Initialize game.
    key, subkey = jax.random.split(key)
    state = reset(subkey)

    # ##>: Storage for trajectory data.
    observations = []
    actions = []
    rewards = []
    policies = []
    values = []

    step_count = 0
    total_reward = 0.0

    while not bool(state.done) and step_count < config.max_trajectory_length:
        # ##>: Get observation.
        obs = encode_observation(state.board)
        observations.append(np.array(obs))

        # ##>: Get legal actions.
        legal_mask = legal_actions_mask(state.board)

        # ##>: Run MCTS.
        key, mcts_key, action_key, step_key = jax.random.split(key, 4)

        # ##>: JIT-compiled MCTS for faster execution.
        policy_output = run_mcts_jit(
            observation=obs,
            params=params,
            apply_fns=apply_fns,
            key=mcts_key,
            num_simulations=config.num_simulations,
            num_actions=config.action_size,
            codebook_size=config.codebook_size,
            discount=config.discount,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_fraction=config.dirichlet_fraction,
            pb_c_init=config.pb_c_init,
            pb_c_base=config.pb_c_base,
        )

        # ##>: Get policy target and select action.
        policy = get_policy_target(policy_output, legal_mask, temperature=1.0)
        search_value = get_search_value(policy_output)
        action = select_action(policy_output, action_key, legal_mask, temperature)

        # ##>: Store step data.
        actions.append(int(action))
        policies.append(np.array(policy))
        values.append(float(search_value))

        # ##>: Take step in environment.
        new_board, reward = next_state(state.board, int(action), step_key)
        done = is_done(new_board)

        rewards.append(float(reward))
        total_reward += float(reward)

        # ##>: Update state.
        state = GameState(
            board=new_board,
            step_count=state.step_count + 1,
            done=done,
            total_reward=state.total_reward + reward,
        )

        step_count += 1

    # ##>: Create trajectory.
    trajectory = Trajectory(
        observations=np.stack(observations, axis=0),
        actions=np.array(actions, dtype=np.int32),
        rewards=np.array(rewards, dtype=np.float32),
        policies=np.stack(policies, axis=0),
        values=np.array(values, dtype=np.float32),
        done=bool(state.done),
        total_reward=total_reward,
        max_tile=int(max_tile(state.board)),
    )

    return trajectory


def warmup_mcts(
    params: NetworkParams,
    apply_fns: NetworkApplyFns,
    key: PRNGKey,
    config: TrainConfig,
) -> None:
    """
    Trigger JIT compilation of MCTS before game loop.

    Runs a single MCTS call to compile the JIT function, avoiding
    compilation overhead during actual self-play.

    Parameters
    ----------
    params : NetworkParams
        Network parameters.
    apply_fns : NetworkApplyFns
        Network apply functions.
    key : PRNGKey
        JAX random key (consumed).
    config : TrainConfig
        Training configuration.
    """
    # ##>: Create dummy observation.
    dummy_obs = jax.numpy.zeros(config.observation_shape)

    # ##>: Single MCTS call to trigger JIT compilation.
    _ = run_mcts_jit(
        observation=dummy_obs,
        params=params,
        apply_fns=apply_fns,
        key=key,
        num_simulations=config.num_simulations,
        num_actions=config.action_size,
        codebook_size=config.codebook_size,
        discount=config.discount,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_fraction=config.dirichlet_fraction,
        pb_c_init=config.pb_c_init,
        pb_c_base=config.pb_c_base,
    )

    # ##>: Block until compilation finishes.
    jax.block_until_ready(_)


def generate_games(
    params: NetworkParams,
    apply_fns: NetworkApplyFns,
    key: PRNGKey,
    config: TrainConfig,
    num_games: int,
    training_step: int = 0,
    show_progress: bool = False,
    warmup: bool = True,
) -> list[Trajectory]:
    """
    Generate multiple games for training.

    Parameters
    ----------
    params : NetworkParams
        Network parameters.
    apply_fns : NetworkApplyFns
        Network apply functions.
    key : PRNGKey
        JAX random key.
    config : TrainConfig
        Training configuration.
    num_games : int
        Number of games to generate.
    training_step : int
        Current training step.
    show_progress : bool
        Whether to show progress bar.
    warmup : bool
        If True, trigger JIT warmup before first game.

    Returns
    -------
    list[Trajectory]
        List of generated trajectories.
    """
    # ##>: Warmup JIT on first call to avoid compilation during gameplay.
    if warmup:
        key, warmup_key = jax.random.split(key)
        warmup_mcts(params, apply_fns, warmup_key, config)

    trajectories = []

    if show_progress:
        game_iter = tqdm(range(num_games), desc='Self-play', unit='game', leave=False)
    else:
        game_iter = range(num_games)

    for _ in game_iter:
        key, subkey = jax.random.split(key)
        traj = play_game(params, apply_fns, subkey, config, training_step)
        trajectories.append(traj)

        if show_progress:
            game_iter.set_postfix(
                reward=f'{traj.total_reward:.0f}',
                tile=traj.max_tile,
                moves=len(traj),
            )

    return trajectories


def evaluate_games(
    params: NetworkParams,
    apply_fns: NetworkApplyFns,
    key: PRNGKey,
    config: TrainConfig,
    num_games: int,
) -> dict:
    """
    Evaluate agent performance over multiple games.

    Uses greedy action selection (temperature=0) for evaluation.

    Parameters
    ----------
    params : NetworkParams
        Network parameters.
    apply_fns : NetworkApplyFns
        Network apply functions.
    key : PRNGKey
        JAX random key.
    config : TrainConfig
        Training configuration.
    num_games : int
        Number of games to play.

    Returns
    -------
    dict
        Evaluation statistics.
    """
    # ##>: Create eval config with greedy temperature.
    eval_training_step = config.training_steps + 1  # Forces temperature = 0

    trajectories = generate_games(
        params=params,
        apply_fns=apply_fns,
        key=key,
        config=config,
        num_games=num_games,
        training_step=eval_training_step,
        show_progress=False,
    )

    rewards = [t.total_reward for t in trajectories]
    max_tiles = [t.max_tile for t in trajectories]
    lengths = [len(t) for t in trajectories]

    # ##>: Count tile achievements.
    tile_counts = {}
    for tile in [2048, 4096, 8192, 16384, 32768]:
        tile_counts[f'reached_{tile}'] = sum(1 for t in max_tiles if t >= tile)

    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'max_reward': float(np.max(rewards)),
        'min_reward': float(np.min(rewards)),
        'mean_max_tile': float(np.mean(max_tiles)),
        'max_tile': int(np.max(max_tiles)),
        'mean_length': float(np.mean(lengths)),
        **tile_counts,
    }


def compute_n_step_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    discount: float,
    n_steps: int,
    td_lambda: float = 0.5,
) -> np.ndarray:
    """
    Compute n-step TD(λ) returns.

    Parameters
    ----------
    rewards : np.ndarray
        Rewards at each step, shape (T,).
    values : np.ndarray
        Value estimates at each step, shape (T,).
    discount : float
        Discount factor γ.
    n_steps : int
        Number of steps for n-step returns.
    td_lambda : float
        TD(λ) parameter for exponential averaging.

    Returns
    -------
    np.ndarray
        n-step returns for each step, shape (T,).
    """
    T = len(rewards)
    returns = np.zeros(T)

    for t in range(T):
        # ##>: Compute n-step returns with TD(λ) averaging.
        n_step_return = 0.0
        lambda_weight = 0.0

        for n in range(1, min(n_steps + 1, T - t + 1)):
            # ##>: n-step return: r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V_{t+n}
            discounted_sum = sum((discount**i) * rewards[t + i] for i in range(n) if t + i < T)

            # ##>: Add bootstrap value if not terminal.
            if t + n < T:
                discounted_sum += (discount**n) * values[t + n]

            # ##>: TD(λ) weighting.
            weight = ((1 - td_lambda) * (td_lambda ** (n - 1))) if n < n_steps else (td_lambda ** (n - 1))
            n_step_return += weight * discounted_sum
            lambda_weight += weight

        # ##>: Normalize by weight sum.
        if lambda_weight > 0:
            returns[t] = n_step_return / lambda_weight
        else:
            returns[t] = values[t]

    return returns
