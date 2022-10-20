"""..."""
from gym.envs.registration import register

from .envs import GameBoard
from .wrappers import FlattenLogObservation, FlattenObservation, LogObservation

register(id="GameBoard", entry_point="simulate_2048.envs:GameBoard", kwargs={"size": 4})
