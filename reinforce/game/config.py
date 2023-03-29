# -*- coding: utf-8 -*-
"""
Environment specific configuration.
"""
from reinforce.addons.config import (
    ENCODAGE_SIZE,
    BufferConfig,
    Factory,
    MonteCarlosConfig,
    NoiseConfig,
    SelfPlayConfig,
    StochasticAlphaZeroConfig,
    TrainingConfig,
    UpperConfidenceBounds,
)
from reinforce.models.network import TrainNetwork

from .game import Game2048


def visit_softmax_temperature(train_steps: int) -> float:
    """
    Compute temperature during training.

    Parameters
    ----------
    train_steps: int
        Training step

    Returns
    -------
    float
        Temperature
    """
    if train_steps < 41:
        return 1.0
    if train_steps < 81:
        return 0.5
    return 0.1


def config_2048() -> StochasticAlphaZeroConfig:
    """
    Returns the config for the game of 2048.

    Returns
    -------
    StochasticAlphaZeroConfig
        Configuration for the game of 2048
    """

    # ##: Callables.
    def _environment_factory():
        return Game2048()

    def _network_factory():
        return TrainNetwork(ENCODAGE_SIZE)

    # ##: Return configuration.
    return StochasticAlphaZeroConfig(
        noise=NoiseConfig(
            root_dirichlet_alpha=0.3,
            root_dirichlet_adaptive=False,
            root_exploration_fraction=0.1,
        ),
        search=MonteCarlosConfig(
            bounds=UpperConfidenceBounds(
                discount=0.999,
                pb_c_base=19652,
                pb_c_init=1.25,
            ),
            num_simulations=50,
        ),
        replay=BufferConfig(
            td_steps=10,
            td_lambda=0.5,
            batch_size=1024,
            num_trajectories=int(125e3),
        ),
        factory=Factory(
            network_factory=_network_factory,
            environment_factory=_environment_factory,
        ),
        training=TrainingConfig(
            epochs=1,
            export=int(1e3),
            store_path="",
            learning_rate=3e-3,
            training_step=int(20e6),
            visit_softmax_temperature=visit_softmax_temperature,
        ),
        self_play=SelfPlayConfig(episodes=10, evaluation=10),
    )
