# -*- coding: utf-8 -*-
"""
Display the environment in a window.
"""
from matplotlib import pyplot as plt
import numpy as np


class Window:
    """
    Window to draw the 2048 board using Matplotlib.
    Inspired by @Farama-Foundation (Minigrid).
    """
    def __init__(self, title: str):
        self.no_image_to_show = True
        self.imshow = None

        # ## ----> Create support.
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(title)

        self.ax.xaxis.set_ticks_position("none")
        self.ax.yaxis.set_ticks_position("none")
        _ = self.ax.set_xticklabels([])
        _ = self.ax.set_yticklabels([])

        # ## ----> Flag indicating that the window was closed.
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect("close_event", close_handler)

    def show_image(self, img: np.ndarray):
        """
        Show an image or update the image being shown.

        Parameters
        ----------
        img: np.ndarray
            Image to show
        """
        # ## ----> Show first image of the environment.
        if self.no_image_to_show:
            self.imshow = self.ax.imshow(img, interpolation="bilinear")
            self.no_image_to_show = False

        # ## ----> Update the image data.
        self.imshow.set_data(img)

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
