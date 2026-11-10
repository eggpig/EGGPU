"""Strict-finite read-only Mapping for an independently owned FP64 SSSP result."""
from collections.abc import Mapping
import math
import numpy as np


class DenseSsspDistances(Mapping):
    def __init__(self, values, reachable_count):
        self._values = np.asarray(values)
        if self._values.ndim != 1 or self._values.dtype != np.dtype('float64'):
            raise TypeError('SSSP output must be a one-dimensional FP64 array')
        self._values.setflags(write=False)
        self._reachable_count = int(reachable_count)
        self._finite_indices = None

    def __len__(self):
        return self._reachable_count

    def __iter__(self):
        if self._reachable_count == len(self._values):
            return iter(range(len(self._values)))
        if self._finite_indices is None:
            self._finite_indices = np.flatnonzero(np.isfinite(self._values))
        return (int(index) for index in self._finite_indices)

    def __getitem__(self, key):
        # Canonical integer vertex IDs; preserve equal integral numeric keys
        # as a Python dict does, while rejecting strings and fractional IDs.
        hash(key)
        try:
            index = int(key.real if isinstance(key, complex) else key)
        except (TypeError, ValueError, OverflowError):
            raise KeyError(key) from None
        if key != index or index < 0 or index >= len(self._values):
            raise KeyError(key)
        value = float(self._values[index])
        if not math.isfinite(value):
            raise KeyError(key)
        return value

    def to_dense(self, copy=False):
        return self._values.copy() if copy else self._values

    def to_numpy(self, copy=False):
        return self.to_dense(copy=copy)

    def copy(self):
        return dict(self.items())
