from typing import Tuple

from keras import Model, ops, random
from numpy import ndarray

from neurals import make_prediction


class PPOAgent:

    GENERATOR = random.SeedGenerator(1335)

    def __init__(self, actor: Model, critic: Model):
        self.actor = actor
        self.critic = critic

    def sample_action(self, state: ndarray) -> Tuple[ndarray, int]:
        logits = make_prediction(self.actor, state=state)
        action = ops.squeeze(random.categorical(logits, 1, seed=self.GENERATOR), axis=1)
        return logits, action

    def choose_action(self, state: ndarray) -> int:
        _, action = self.sample_action(state)
        return action
