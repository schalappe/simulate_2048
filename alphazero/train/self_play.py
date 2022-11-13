# -*- coding: utf-8 -*-
"""
Self play function.
"""
from collections import Counter
import time

from numpy import max as max_array_values
from numpy import sum as sum_array_values

from alphazero.addons.config import StochasticAlphaZeroConfig
from alphazero.addons.types import State
from alphazero.models.network import Network
from alphazero.module.actor import StochasticMuZeroActor
from alphazero.module.replay import ReplayBuffer


def run_self_play(config: StochasticAlphaZeroConfig, network: Network, replay_buffer: ReplayBuffer, step: int) -> None:
    """
    Takes the latest network snapshot, produces an episode and makes it available to the training job by writing it
    to a replay buffer.

    Parameters
    ----------
    config: StochasticAlphaZeroConfig
        Configuration for self play
    network: Network
        The latest network.
    replay_buffer: ReplayBuffer
        Buffer for experience
    step: int
        Training step

    """
    actor = StochasticMuZeroActor(config, network)

    epoch_start = time.time()
    max_values = 2
    for num in range(config.self_play.episodes):

        # ##: Create a new instance of the environment.
        env = config.factory.environment_factory()

        # ##: Reset the actor.
        actor.reset(step)

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
            print(f"Game: {num + 1} - Score: {sum_array_values(env.observation())}", end="\r")
        print(f"Game: {num + 1} - Score: {sum_array_values(env.observation())}")

        # ##: Send the episode to the replay.
        replay_buffer.save(episode)
        max_values = max(max_values, max_array_values(env.observation()))

    # ##: Display time
    epoch_end = time.time()
    elapsed = (epoch_end - epoch_start) / 60.0
    print(f"Max: {max_values} for the self-play ...")
    print("Self-play took {:.4} minutes".format(elapsed))


def run_eval(config: StochasticAlphaZeroConfig, network: Network) -> dict:
    """
    Evaluate an agent.

    Parameters
    ----------
    config: StochasticAlphaZeroConfig
        Configuration for self play
    network: Network
        The latest network.
    """
    actor = StochasticMuZeroActor(config, network)
    score = []

    for num in range(config.self_play.evaluation):
        # ##: Create a new instance of the environment.
        env = config.factory.environment_factory()

        # ##: Reset the actor.
        actor.reset(0)

        # ##: Play a game.
        while not env.is_terminal():
            # ##: Interact with environment
            obs = env.observation()
            action = actor.select_action(obs)
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
