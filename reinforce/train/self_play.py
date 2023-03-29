# -*- coding: utf-8 -*-
"""
Self play function.
"""
import time
from collections import Counter

import tqdm
from numpy import max as max_array_values
from numpy import sum as sum_array_values

from reinforce.addons.config import StochasticAlphaZeroConfig
from reinforce.addons.types import State
from reinforce.models.network import Network
from reinforce.module.actor import StochasticMuZeroActor
from reinforce.module.replay import ReplayBuffer


def run_self_play(config: StochasticAlphaZeroConfig, network: Network, replay_buffer: ReplayBuffer, epochs: int):
    """
    Takes the latest network snapshot, produces an episode and makes it available to the training job by writing it
    to a replay buffer.

    Parameters
    ----------
    config: StochasticAlphaZeroConfig
        Configuration for self play
    network: Network
        Model to use
    replay_buffer: ReplayBuffer
        Buffer for experience
    epochs: int
        Training step
    """
    actor = StochasticMuZeroActor(config, network, epochs=epochs, train=True)
    epoch_start = time.time()
    max_values = 2
    with tqdm.trange(config.self_play.episodes) as period:
        for num in period:

            # ##: Create a new instance of the environment.
            env = config.factory.environment_factory()

            # ##: Create the actor.
            actor.reset()

            # ##: Play a game.
            episode = []
            while not env.is_terminal():
                # ##: Interact with environment
                obs = env.observation()
                action = actor.select_action(obs)

                # ##: Store state
                state = State(
                    observation=obs,
                    reward=env.reward(),
                    discount=config.search.bounds.discount,
                    action=action,
                    search_stats=actor.stats(),
                )
                episode.append(state)
                env.step(action)

                # ##: Log.
                period.set_description(f"Self play: {num + 1}")
                period.set_postfix(score=sum_array_values(env.observation()), max=max_array_values(env.observation()))

            # ##: Send the episode to the replay.
            replay_buffer.save(episode)
            max_values = max(max_values, max_array_values(env.observation()))

    # ##: Display time
    epoch_end = time.time()
    elapsed = (epoch_end - epoch_start) / 60.0
    print(f"Max: {max_values} for the self-play ...")
    print(f"Self-play took {elapsed:.4} minutes")


def run_eval(config: StochasticAlphaZeroConfig, network: Network) -> dict:
    """
    Evaluate an agent.

    Parameters
    ----------
    config: StochasticAlphaZeroConfig
        Configuration for self play
    network: Network
        Model to use
    """
    actor = StochasticMuZeroActor(config, network, epochs=0, train=False)
    score = []

    with tqdm.trange(config.self_play.evaluation) as period:
        for num in period:
            # ##: Create a new instance of the environment.
            env = config.factory.environment_factory()

            # ##: Reset the actor.
            actor.reset()

            # ##: Play a game.
            while not env.is_terminal():
                # ##: Interact with environment
                env.step(actor.select_action(env.observation()))

                # ##: Log.
                period.set_description(f"Evaluation: {num + 1}")
                period.set_postfix(score=sum_array_values(env.observation()), max=max_array_values(env.observation()))

            # ##: Save max cells.
            score.append(max_array_values(env.observation()))

    # ##: Final log.
    frequency = Counter(score)
    return dict(frequency)
