# -*- coding: utf-8 -*-
"""
Play 2048 Game
"""
from typing import Any
from twentyfortyeight.envs import TwentyFortyEight
import numpy as np

from twentyfortyeight.utils import WindowBoard


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


def reset(envs: TwentyFortyEight, window: WindowBoard):
    """
    Reset and redraw the game board.

    Parameters
    ----------
    envs: TwentyFortyEight
        The Game environment

    window: WindowBoard
        Class to draw the game board
    """
    # ##: Reset the game.
    board = envs.reset()

    # ##: Redraw the game board.
    redraw(window, board)


def step(envs: TwentyFortyEight, window: WindowBoard, action: int):
    """
    Applied action into the game.

    Parameters
    ----------
    envs: TwentyFortyEight
        The Game environment

    window: WindowBoard
        Class to draw the game board

    action: int
        Action to apply
    """
    obs, reward, terminated = envs.step(action)
    print(f"reward={reward:.2f}")

    redraw(window, obs)
    if terminated:
        print("terminated!")


def key_handler(envs: TwentyFortyEight, window: WindowBoard, event: Any):
    """
    Handle the keyboard.

    Parameters
    ----------
    envs: TwentyFortyEight
        The Game environment

    window: WindowBoard
        Class to draw the game board

    event: Any
        event to handle
    """
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return None

    if event.key == "backspace":
        reset(envs, window)
        return None

    if event.key in envs.ACTIONS:
        step(envs, window, envs.ACTIONS[event.key])
        return None


if __name__ == "__main__":
    env = TwentyFortyEight()
    env.reset()

    window_board = WindowBoard(title="2048 Game", size=env.size)
    window_board.register_key_handler(lambda event: key_handler(env, window_board, event))

    reset(env, window_board)

    # Blocking event loop
    window_board.show(block=True)
