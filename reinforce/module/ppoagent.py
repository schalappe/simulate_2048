# -*- coding: utf-8 -*-
"""
Proximal Policy Agent.
"""
from numpy import ndarray

from .agent import TrainingAgent
from reinforce.addons import AgentConfigurationPPO, GCAdam

from reinforce.models import dense_policy


class ReplayBuffer:
    pass


class AgentPPO(TrainingAgent):
    def __initialize_optimizer(self, actor_learning_rate: float, critic_learning_rate: float, batch_size: int):
        self._batch_size = batch_size
        self._optimizer_actor = GCAdam(learning_rate=actor_learning_rate)
        self._optimizer_critic = GCAdam(learning_rate=critic_learning_rate)

    def _initialize_agent(self, config: AgentConfigurationPPO, observation_type: str, reward_type: str):
        # ## ----> Initialize models.
        self._actor, self._critic = dense_policy(input_size=(4, 4, 1))

        # ## ----> Initialization optimizer.
        self.__initialize_optimizer(config.learning_rate, config.second_learning_rate, config.batch_size)

        self._name = "_".join([config.type_model, observation_type, reward_type])

    def select_action(self, state: ndarray) -> int:
        pass

    def save_model(self):
        pass

    def optimize_model(self):
        pass
