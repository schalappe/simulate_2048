# -*- coding: utf-8 -*-
"""
This module provides utilities for encoding, normalizing, and managing game boards.

It includes functions for flattening and encoding board states, normalizing rewards, and a `WindowBoard` class
for managing game boards with sliding windows.
"""

from .binary import encode_flatten
from .normalize import normalize_reward
from .windows import WindowBoard

__all__ = ["encode_flatten", "normalize_reward", "WindowBoard"]
