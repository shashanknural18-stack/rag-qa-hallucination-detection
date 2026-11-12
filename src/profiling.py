"""Lightweight profiler recording wall time and peak process RAM per stage."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import psutil


@dataclass
class StageRecord:
    """A record of a single profiled stage."""

    name: str
    duration: float
    start_ram_mb: float
    end_ram_mb: float
    peak_ram_mb: float
    extra: dict[str, Any] = field(default_factory=dict)


class Profiler:
    """Profiler that tracks stage timings and memory usage."""

    def __init__(self) -> None:
        self.records: list[StageRecord] = []
        self._process = psutil.Process(os.getpid())

    def current_ram_mb(self) -> float:
        """Return current RSS in MB."""
        try:
            return self._process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    @contextmanager
    def stage(self, name: str, **extra: Any) -> Iterator[dict[str, Any]]:
        """Context manager to record a stage's wall time and peak RAM."""
        metadata = dict(extra)
        start_ram = self.current_ram_mb()
        t0 = time.perf_counter()
        try:
            yield metadata
        finally:
            t1 = time.perf_counter()
            end_ram = self.current_ram_mb()
            record = StageRecord(
                name=name,
                duration=t1 - t0,
                start_ram_mb=start_ram,
                end_ram_mb=end_ram,
                peak_ram_mb=max(start_ram, end_ram),
                extra=metadata,
            )
            self.records.append(record)

    def summary(self) -> list[dict[str, Any]]:
        """Return summary records."""
        return [
            {
                "stage": r.name,
                "duration_s": round(r.duration, 4),
                "peak_ram_mb": round(r.peak_ram_mb, 2),
                **r.extra,
            }
            for r in self.records
        ]

    def clear(self) -> None:
        """Clear recorded stages."""
        self.records.clear()


_GLOBAL_PROFILER = Profiler()


def get_profiler() -> Profiler:
    """Return the global profiler instance."""
    return _GLOBAL_PROFILER


@contextmanager
def stage(name: str, profiler: Profiler | None = None, **extra: Any) -> Iterator[dict[str, Any]]:
    """Record wall time and peak process RAM for stage *name*."""
    target = profiler or _GLOBAL_PROFILER
    with target.stage(name, **extra) as meta:
        yield meta

