# -*- coding: utf-8 -*-
"""
Environment specific configuration.
"""
from alphazero.addons.config import (
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
from alphazero.models.network import Network

from .game import Game2048


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
        return Network(ENCODAGE_SIZE)

    def _visit_softmax_temperature(train_steps: int) -> float:
        return 1.0

    # ##: Return configuration.
    return StochasticAlphaZeroConfig(
        loop=1000,
        export=10,
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
            num_unroll_steps=5,
            num_trajectories=125000,
        ),
        factory=Factory(
            network_factory=_network_factory,
            environment_factory=_environment_factory,
        ),
        training=TrainingConfig(epochs=200, learning_rate=3e-4, store_path=""),
        self_play=SelfPlayConfig(
            episodes=10,
            evaluation=50,
            visit_softmax_temperature_fn=_visit_softmax_temperature,
        ),
    )
