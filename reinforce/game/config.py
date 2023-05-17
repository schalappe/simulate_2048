# -*- coding: utf-8 -*-
"""
Environment specific configuration.
"""
from dataclasses import dataclass
from typing import Callable

from reinforce.actor.actor import Actor
from reinforce.learner.learner import Learner
from reinforce.network.cacher import NetworkCacher
from reinforce.replay.replay import BufferReplay

from .game import Game2048

ENCODAGE_SIZE = 31


@dataclass
class LearnerConfiguration:
    """Data needed to update network weight."""

    epochs: int


@dataclass
class ActorConfiguration:
    """Data needed for self-play."""

    episodes: int
    environment_factory: Callable = lambda: Game2048(encodage_size=ENCODAGE_SIZE)


@dataclass
class Configuration:
    """Configuration necessary for reinforcement learning."""

    cycles: int
    actor_factory: Callable[[BufferReplay, NetworkCacher], Actor]
    learn_factory: Callable[[BufferReplay, NetworkCacher], Learner]
    replay_factory: Callable[[], BufferReplay]
    cacher_factory: Callable[[int], NetworkCacher]
