"""Small shared-memory ring buffer for numeric sensor samples.

This is intentionally simpler than UMI's lock-free implementation. It is
enough for low-dimensional streams such as tracker poses and encoder values,
and keeps the API shape close to UMI's ``get_last_k`` pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import multiprocessing as mp
import numbers
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ArraySpec:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


def _spec_from_example(name: str, value: object) -> ArraySpec:
    if isinstance(value, np.ndarray):
        if value.dtype == np.dtype("O"):
            raise TypeError(f"object dtype is not supported for {name!r}")
        return ArraySpec(name=name, shape=value.shape, dtype=value.dtype)
    if isinstance(value, (numbers.Number, np.bool_)):
        return ArraySpec(name=name, shape=(), dtype=np.asarray(value).dtype)
    raise TypeError(f"Unsupported ring-buffer example for {name!r}: {type(value)}")


class SharedMemoryRingBuffer:
    """A process-safe ring buffer storing dicts of numeric numpy arrays."""

    def __init__(self, examples: Mapping[str, object], capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.specs = {
            key: _spec_from_example(key, value)
            for key, value in examples.items()
        }
        self._buffers = {}
        for key, spec in self.specs.items():
            n_items = int(np.prod((self.capacity,) + spec.shape, dtype=np.int64))
            nbytes = n_items * spec.dtype.itemsize
            self._buffers[key] = mp.RawArray(ctypes.c_ubyte, nbytes)
        self._count = mp.Value("Q", 0)
        self._lock = mp.Lock()

    @property
    def count(self) -> int:
        return int(self._count.value)

    def _array(self, key: str) -> np.ndarray:
        spec = self.specs[key]
        return np.frombuffer(self._buffers[key], dtype=spec.dtype).reshape(
            (self.capacity,) + spec.shape
        )

    def put(self, data: Mapping[str, object]) -> None:
        missing = set(self.specs) - set(data)
        if missing:
            raise KeyError(f"Missing ring-buffer keys: {sorted(missing)}")
        with self._lock:
            idx = int(self._count.value % self.capacity)
            for key, spec in self.specs.items():
                self._array(key)[idx] = np.asarray(data[key], dtype=spec.dtype)
            self._count.value += 1

    def get_last_k(self, k: int) -> dict[str, np.ndarray]:
        if k <= 0:
            raise ValueError("k must be positive")
        with self._lock:
            count = int(self._count.value)
            if count <= 0:
                raise IndexError("ring buffer is empty")
            k = min(int(k), count, self.capacity)
            start = count - k
            indices = np.arange(start, count) % self.capacity
            return {
                key: self._array(key)[indices].copy()
                for key in self.specs
            }

    def get_latest(self) -> dict[str, np.ndarray]:
        return {
            key: value[0]
            for key, value in self.get_last_k(1).items()
        }

