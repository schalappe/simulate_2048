# -*- coding: utf-8 -*-
"""
Display the environment in a window.
"""
import numpy as np
from matplotlib import pyplot as plt


class WindowBoard:
    """
    Window to draw the 2048 board using Matplotlib.
    Inspired by @Farama-Foundation (Minigrid).
    """

    # ##: Colors
    COLORS = {
        0: "#FFFFFF",
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
        65536: "#9ED682",
        131072: "#9ED682",
    }

    def __init__(self, title: str, size: int):
        # ## ----> Create support.
        self.fig, self.axe = plt.subplots()
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)
        self.axe.set_facecolor("#BBADA0")
        self.fig.canvas.manager.set_window_title(title)

        self.axe.xaxis.set_ticks_position("none")
        self.axe.yaxis.set_ticks_position("none")
        _ = self.axe.set_xticklabels([])
        _ = self.axe.set_yticklabels([])

        # ## ----> Add cell for board.
        self.textes = []
        self.axes = [
            self.fig.add_subplot(size, size, r * size + c) for r in range(0, size) for c in range(1, size + 1)
        ]
        for _ax in self.axes:
            text = _ax.text(
                0.5,
                0.5,
                "0",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize="x-large",
                fontweight="demibold",
            )
            self.textes.append(text)
        for _ax in self.axes:
            _ = _ax.set_xticks([])
            _ = _ax.set_yticks([])

        # ## ----> Flag indicating that the window was closed.
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect("close_event", close_handler)

    def show_image(self, board: np.ndarray):
        """
        Show an image or update the image being shown.

        Parameters
        ----------
        board: np.ndarray
            Image to show
        """
        # ## ----> Update the image data.
        values = np.reshape(board, -1)
        for _ax, text, value in zip(self.axes, self.textes, values):
            text.set_text(str(int(value)))
            _ax.set_facecolor(self.COLORS[int(value)])

        # ## ---> Request the window to be redrawn
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # ## ----> Let Matplotlib process UI events
        plt.pause(0.001)

    def register_key_handler(self, key_handler):
        """
        Register a keyboard event handler.

        Parameters
        ----------
        key_handler: Any
            Key handler
        """
        self.fig.canvas.mpl_connect("key_press_event", key_handler)

    def show(self, block: bool = True):
        """
        Show the window, and start an event loop.

        Parameters
        ----------
        block: bool
            Activate or not the interactive mode
        """
        # ## ----> If not blocking, trigger interactive mode.
        if not block:
            plt.ion()

        # ## ----> Show the plot.
        plt.show()

    def close(self):
        """
        Close the window.
        """
        plt.close()
        self.closed = True
