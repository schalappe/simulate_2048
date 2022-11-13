# -*- coding: utf-8 -*-
"""
Self play function.
"""
from collections import Counter

from numpy import max as max_array_values

from alphazero.addons.config import StochasticAlphaZeroConfig
from alphazero.addons.types import State
from alphazero.models.network import NetworkCacher
from alphazero.module.actor import StochasticMuZeroActor
from alphazero.module.replay import ReplayBuffer


def run_self_play(config: StochasticAlphaZeroConfig, cacher: NetworkCacher, replay_buffer: ReplayBuffer) -> int:
    """
    Takes the latest network snapshot, produces an episode and makes it available to the training job by writing it
    to a replay buffer.

    Parameters
    ----------
    config: StochasticAlphaZeroConfig
        Configuration for self play
    cacher: NetworkCacher
        List of network weights
    replay_buffer: ReplayBuffer
        Buffer for experience

    """
    actor = StochasticMuZeroActor(config, cacher)

    # ##: Create a new instance of the environment.
    env = config.factory.environment_factory()

    # ##: Reset the actor.
    actor.reset()

    # ##: Play a game.
    episode = []
    while not env.is_terminal():
        # ##: Interact with environment
        obs = env.observation()
        reward = env.reward()
        action = actor.select_action(obs)

        # ##: Store state
        state = State(
            observation=obs,
            reward=reward,
            discount=config.search.bounds.discount,
            action=action,
            search_stats=actor.stats(),
        )
        episode.append(state)
        env.step(action)

    # ##: Send the episode to the replay.
    replay_buffer.save(episode)
    return max_array_values(env.observation())


def run_eval(config: StochasticAlphaZeroConfig, cacher: NetworkCacher) -> dict:
    """
    Evaluate an agent.

    Parameters
    ----------
    config: StochasticAlphaZeroConfig
        Configuration for self play
    cacher: NetworkCacher
        List of network weights
    """
    actor = StochasticMuZeroActor(config, cacher)
    score = []

    for num in range(config.self_play.evaluation):
        # ##: Create a new instance of the environment.
        env = config.factory.environment_factory()

        # ##: Reset the actor.
        actor.reset()

        # ##: Play a game.
        while not env.is_terminal():
            # ##: Interact with environment
            obs = env.observation()
            action = actor.select_action(obs, train=False)
            env.step(action)

            # ##: Log.
            print(f"Game: {num + 1} - Max: {max_array_values(env.observation())}", end="\r")

        # ##: Save max cells.
        score.append(max_array_values(env.observation()))

    # ##: Final log.
    # print("Evaluation is finished.")
    frequency = Counter(score)
    # print(f"Result: {dict(frequency)}")
    return dict(frequency)
