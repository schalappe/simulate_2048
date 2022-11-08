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
        if train_steps < 1e5:
            return 1.0
        if train_steps < 2e5:
            return 0.5
        if train_steps < 3e5:
            return 0.1
        # ##: Greedy selection.
        return 0.0

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
            num_simulations=20,  # 100,
        ),
        replay=BufferConfig(
            td_steps=10,
            td_lambda=0.5,
            batch_size=1024,
            num_unroll_steps=5,
            num_trajectories=int(125e3),
        ),
        factory=Factory(
            network_factory=_network_factory,
            environment_factory=_environment_factory,
        ),
        training=TrainingConfig(learning_rate=3e-4, training_steps=int(20e6), store_path=""),
        self_play=SelfPlayConfig(
            num_actors=2,  # 1000,
            visit_softmax_temperature_fn=_visit_softmax_temperature,
        ),
    )
