# -*- coding: utf-8 -*-
"""
This module provides utility functions for managing game states and actions in a 2048-like game.

It includes functions for checking legal and illegal actions, sliding and merging tiles,
filling empty cells, generating after states and latent states, checking if the game is done,
and computing the next game state.
"""

from .gameboard import (
    after_state,
    fill_cells,
    is_done,
    latent_state,
    merge_column,
    next_state,
    slide_and_merge,
)
from .gamemove import illegal_actions, legal_actions

__all__ = [
    "legal_actions",
    "illegal_actions",
    "slide_and_merge",
    "fill_cells",
    "after_state",
    "latent_state",
    "is_done",
    "next_state",
    "merge_column",
]
