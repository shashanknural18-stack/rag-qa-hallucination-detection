"""Shared detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.generation.schemas import GeneratedAnswer
    from src.detection.schemas import DetectorResult


def clamp_unit(value: float) -> float:
    """Clamp *value* into the closed unit interval."""
    return max(0.0, min(1.0, float(value)))


class BaseDetector(ABC):
    """A detector that scores how well an answer is supported by evidence.

    Scores are in ``[0, 1]``, where 1.0 means fully supported.
    """

    name: str

    @abstractmethod
    def detect(
        self,
        generated: GeneratedAnswer,
        sentences: list[str],
    ) -> DetectorResult:
        """Score *generated* against *sentences* (citation markers already removed)."""
