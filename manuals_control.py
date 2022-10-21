# -*- coding: utf-8 -*-
"""
Play 2048 Game
"""
from typing import Any

import gym
import numpy as np

from simulate_2048.utils import WindowBoard


def redraw(window: WindowBoard, board: np.ndarray):
    """
    Redraw the game board.

    Parameters
    ----------
    window: WindowBoard
        Class to draw the game board

    board: np.ndarray
        Game board to draw
    """
    window.show_image(board)


def reset(envs: gym.Env, window: WindowBoard):
    """
    Reset and redraw the game board.

    Parameters
    ----------
    envs: gym.Env
        The Game environment

    window: WindowBoard
        Class to draw the game board
    """
    # ## ---> Reset the game.
    envs.reset()

    # ## ----> Redraw the game board.
    board = envs.board
    redraw(window, board)


def step(envs: gym.Env, window: WindowBoard, action: int):
    """
    Applied action into the game.

    Parameters
    ----------
    envs: gym.Env
        The Game environment

    window: WindowBoard
        Class to draw the game board

    action: int
        Action to apply
    """
    obs, reward, terminated, _, info = envs.step(action)
    print(f"reward={reward:.2f}")

    board = envs.board
    redraw(window, board)
    if terminated:
        print("terminated!")


def key_handler(envs: gym.Env, window: WindowBoard, event: Any):
    """
    Handle the keyboard.

    Parameters
    ----------
    envs: gym.Env
        The Game environment

    window: WindowBoard
        Class to draw the game board

    event: Any
        event to handle
    """
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset(envs, window)
        return

    if event.key == "left":
        step(envs, window, envs.LEFT)
        return
    if event.key == "right":
        step(envs, window, envs.RIGHT)
        return
    if event.key == "up":
        step(envs, window, envs.UP)
        return
    if event.key == "down":
        step(envs, window, envs.DOWN)
        return


if __name__ == "__main__":
    env = gym.make("GameBoard")
    env.seed()

    window_board = WindowBoard(title="2048 Game", size=env.size)
    window_board.register_key_handler(lambda event: key_handler(env, window_board, event))

    reset(env, window_board)

    # Blocking event loop
    window_board.show(block=True)
