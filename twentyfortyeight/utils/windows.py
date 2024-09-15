# -*- coding: utf-8 -*-
"""
Graphical User Interface for 2048 Game Environment

This module provides functionality to create and manage a graphical window for displaying the
2048 game board. It utilizes Matplotlib for rendering and handling user interactions, offering
a visual representation of the game state and allowing for real-time updates as the game progresses.
"""
from typing import Callable, Optional

from matplotlib import pyplot as plt
from matplotlib.backend_bases import Event
from numpy import ndarray


class WindowBoard:
    """
    A class for rendering and managing the 2048 game board using Matplotlib.

    Methods
    -------
    show_image(board: np.ndarray)
        Update the display with the current game board state.
    register_key_handler(key_handler: Callable)
        Register a function to handle keyboard events.
    show(block: bool = True)
        Display the game window.
    close()
        Close the game window.

    Notes
    -----
    - This class uses Matplotlib for rendering, which allows for interactive
      visualization of the game board.
    - The window can be updated in real-time as the game progresses.
    - Keyboard events can be captured for user input or control.
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
            The size of the game board (e.g., 4 for a 4x4 board).
        """
        self.fig, self.axe = plt.subplots()
        self.fig.canvas.manager.set_window_title(title)
        self._setup_axes(size)
        self.closed = False
        self.fig.canvas.mpl_connect("close_event", self._close_handler)

    def _setup_axes(self, size: int):
        """
        Set up the axes for the game board.

        This method initializes the subplot structure for the game board, creating individual cells for each tile.

        Parameters
        ----------
        size : int
            The size of the game board.
        """
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.05)
        self.axe.set_facecolor("#BBADA0")

        # ##: Remove all ticks and labels for a cleaner game board appearance.
        self.axe.tick_params(axis="both", which="both", length=0)
        self.axe.set_xticklabels([])
        self.axe.set_yticklabels([])
        self.axe.set_xticks([])
        self.axe.set_yticks([])

        self.texts = []
        self.axes = [self.fig.add_subplot(size, size, r * size + c + 1) for r in range(size) for c in range(size)]
        for ax in self.axes:
            text = ax.text(0.5, 0.5, "", ha="center", va="center", fontsize="x-large", fontweight="demibold")
            self.texts.append(text)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    def _close_handler(self, event: Optional[Event] = None):
        """
        Handle the window close event.

        This method sets the closed flag to True when the window is closed.

        Parameters
        ----------
        event : Optional[Event]
            The close event (not used but required for event handling).
        """
        self.closed = True

    def show_image(self, board: ndarray):
        """
        Show or update the game board.

        Parameters
        ----------
        board : ndarray
            The current state of the game board to be displayed.

        Notes
        -----
        - This method updates the colors and text of each cell in the display.
        - It uses the COLORS dictionary to map tile values to colors.
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

        Notes
        -----
        - The registered function will be called whenever a key is pressed in the window.
        - This can be used to implement user controls or AI input for the game.
        """
        self.fig.canvas.mpl_connect("key_press_event", key_handler)

    @classmethod
    def show(cls, block: bool = True):
        """
        Show the window and start the Matplotlib event loop.

        This class method displays the game window and controls whether the Matplotlib event loop should block or not.

        Parameters
        ----------
        block : bool, optional
            If True, the event loop is blocking; otherwise, it's non-blocking (default is True).
        """
        if not block:
            plt.ion()
        plt.show()

    def close(self):
        """
        Close the window.

        This method closes the game window and sets the closed flag to True.
        """
        plt.close(self.fig)
        self.closed = True
