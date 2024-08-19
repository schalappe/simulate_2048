# -*- coding: utf-8 -*-
"""
Play 2048 Game
"""
from typing import Any
from simulate.envs import GameBoard
import numpy as np

from simulate.utils import WindowBoard


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


def reset(envs: GameBoard, window: WindowBoard):
    """
    Reset and redraw the game board.

    Parameters
    ----------
    envs: GameBoard
        The Game environment

    window: WindowBoard
        Class to draw the game board
    """
    # ## ---> Reset the game.
    board = envs.reset()

    # ## ----> Redraw the game board.
    redraw(window, board)


def step(envs: GameBoard, window: WindowBoard, action: int):
    """
    Applied action into the game.

    Parameters
    ----------
    envs: GameBoard
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


def key_handler(envs: GameBoard, window: WindowBoard, event: Any):
    """
    Handle the keyboard.

    Parameters
    ----------
    envs: GameBoard
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

    if event.key in envs.ACTIONS:
        step(envs, window, envs.ACTIONS[event.key])
        return


if __name__ == "__main__":
    env = GameBoard()
    env.reset()

    window_board = WindowBoard(title="2048 Game", size=env.size)
    window_board.register_key_handler(lambda event: key_handler(env, window_board, event))

    reset(env, window_board)

    # Blocking event loop
    window_board.show(block=True)
