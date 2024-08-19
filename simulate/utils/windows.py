# -*- coding: utf-8 -*-
"""
Display the environment in a window.
"""
from typing import Callable, Optional

from matplotlib import pyplot as plt
from matplotlib.backend_bases import Event
from numpy import ndarray


class WindowBoard:
    """
    Window to draw the 2048 board using Matplotlib.

    Inspired by @Farama-Foundation (Minigrid).

    This class provides a graphical representation of the 2048 game board using Matplotlib,
    allowing for visualization and updates of the game state in a window.
    """

    # ##: Colors mapping for different tile values.
    COLORS = {
        0: "#CCC0B3",
        2: "#EEE4DA",
        4: "#ECE0C8",
        8: "#ECB280",
        16: "#EC8D53",
        32: "#F57C5F",
        64: "#E95937",
        128: "#F3D96B",
        256: "#F2D04A",
        512: "#E5BF2E",
        1024: "#E2B814",
        2048: "#EBC502",
        4096: "#00A2D8",
        8192: "#9ED682",
        16384: "#9ED682",
        32768: "#9ED682",
    }

    def __init__(self, title: str, size: int):
        """
        Initialize the game board window.

        Parameters
        ----------
        title : str
            The title of the window.
        size : int
            The size of the game board.
        """
        self.fig, self.axe = plt.subplots()
        self.fig.canvas.manager.set_window_title(title)
        self._setup_axes(size)
        self.closed = False
        self.fig.canvas.mpl_connect("close_event", self._close_handler)

    def _setup_axes(self, size: int):
        """Set up the axes for the game board."""
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)
        self.axe.set_facecolor("#BBADA0")
        self.axe.axis("off")

        self.texts = []
        self.axes = [self.fig.add_subplot(size, size, r * size + c + 1) for r in range(size) for c in range(size)]
        for ax in self.axes:
            text = ax.text(0.5, 0.5, "", ha="center", va="center", fontsize="x-large", fontweight="demibold")
            self.texts.append(text)
            ax.axis("off")

    def _close_handler(self, event: Optional[Event] = None):
        """Handle the window close event."""
        self.closed = True

    def show_image(self, board: ndarray):
        """
        Show or update the game board.

        Parameters
        ----------
        board : ndarray
            The current state of the game board.
        """
        for ax, text, value in zip(self.axes, self.texts, board.flat):
            value = int(value)
            text.set_text(str(value) if value != 0 else "")
            ax.set_facecolor(self.COLORS.get(value, "#FFFFFF"))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def register_key_handler(self, key_handler: Callable):
        """
        Register a keyboard event handler.

        Parameters
        ----------
        key_handler : Callable
            A function to handle keyboard events.
        """
        self.fig.canvas.mpl_connect("key_press_event", key_handler)

    @classmethod
    def show(cls, block: bool = True):
        """
        Show the window and start the Matplotlib event loop.

        Parameters
        ----------
        block : bool
            If True, the event loop is blocking; otherwise, it is non-blocking.
        """
        if not block:
            plt.ion()
        plt.show()

    def close(self):
        """Close the window."""
        plt.close(self.fig)
        self.closed = True
