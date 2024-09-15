# -*- coding: utf-8 -*-
from typing import Tuple, Any
from numpy import float32, zeros


class SumTree:
    """
    A sum tree data structure for efficient sampling in prioritized experience replay.

    This tree structure allows O(log n) sampling and updating of priorities.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = zeros(2 * capacity - 1, dtype=float32)
        self.data = zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate the priority update up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find the leaf node index given a sample value s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)

        return self._retrieve(right, s - self.tree[left])

    def add(self, priority: float, data: Any) -> None:
        """Add a new experience with its priority."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float) -> None:
        """Update the priority of an experience."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get an experience using a sample value."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        """Return the total priority."""
        return self.tree[0]

    def __len__(self):
        return self.n_entries
