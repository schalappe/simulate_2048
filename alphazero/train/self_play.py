# -*- coding: utf-8 -*-
"""
Self play function.
"""
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
        Where is store the latest network.
    replay_buffer: ReplayBuffer
        Buffer for experience
    """
    actor = StochasticMuZeroActor(config, cacher)

    for num in range(config.self_play.num_actors):
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
        print(f"Actor n°{num+1} finish ...")
