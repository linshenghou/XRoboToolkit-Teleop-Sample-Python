"""Shared host timebase for multi-process sensor timestamping."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class Timebase:
    """Unix-like timestamps backed by a monotonic clock.

    Create this once in the parent process and pass it to sensor processes.
    Every process can then produce comparable timestamps without depending on
    later wall-clock adjustments.
    """

    wall0: float
    perf0: float

    @classmethod
    def create(cls) -> "Timebase":
        return cls(wall0=time.time(), perf0=time.perf_counter())

    def now(self) -> float:
        return self.wall0 + (time.perf_counter() - self.perf0)


def midpoint(start: float, end: float) -> float:
    """Timestamp a blocking read at the midpoint of its host-time interval."""

    return start + 0.5 * (end - start)

