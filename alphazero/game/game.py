# -*- coding: utf-8 -*-
"""
Environment that implements the rule of 2048.
"""
import gym

from .environment import Environment


class Game2048(Environment):
    """
    The 2048 environment.
    """

    def __init__(self):
        super().__init__()
        self.game = gym.make("GameBoard")
        self.actions = [0, 1, 2, 3]
        self.done = False
        obs, _ = self.game.reset()
        self.observations = [obs]
        self.rewards = [0]

    def step(self, action: int):
        """
        Applies an action or a chance outcome to the environment.
        """
        if action not in self.actions:
            raise ValueError(f"The action `{action}` isn't recognize.")
        observation, reward, done, _, _ = self.game.step(action)
        self.observations += [observation]
        self.rewards += [reward]
        self.done = done

    def observation(self):
        """
        Returns the observation of the environment to feed to the network.
        """
        return self.observations[-1]

    def is_terminal(self) -> bool:
        """
        Returns true if the environment is in a terminal state.
        """
        return self.done

    def reward(self):
        """
        Returns the reward of the environment.
        """
        return self.rewards[-1]
