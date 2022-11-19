# -*- coding: utf-8 -*-
"""
Self play function.
"""
import time
from collections import Counter

from numpy import max as max_array_values
from numpy import sum as sum_array_values

from alphazero.addons.config import StochasticAlphaZeroConfig
from alphazero.addons.types import State
from alphazero.models.network import NetworkCacher
from alphazero.module.actor import StochasticMuZeroActor
from alphazero.module.replay import ReplayBuffer


def run_self_play(config: StochasticAlphaZeroConfig, cacher: NetworkCacher, replay_buffer: ReplayBuffer) -> None:
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
    epoch_start = time.time()
    max_values = 2
    for num in range(config.self_play.episodes):

        # ##: Create a new instance of the environment.
        env = config.factory.environment_factory()

        # ##: Create the actor.
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

            # ##: Log.
            print(f"Actor n°{num + 1}: - Score: {sum_array_values(env.observation())}", end="\r")
        print(f"Actor n°{num + 1}: - Score: {sum_array_values(env.observation())}")

        # ##: Send the episode to the replay.
        replay_buffer.save(episode)
        max_values = max(max_values, max_array_values(env.observation()))

    # ##: Display time
    epoch_end = time.time()
    elapsed = (epoch_end - epoch_start) / 60.0
    print(f"Max: {max_values} for the self-play ...")
    print(f"Self-play took {elapsed:.4} minutes")


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
    frequency = Counter(score)
    return dict(frequency)
