"""Data contracts for hallucination detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.generation.schemas import GeneratedAnswer


class Decision(str, Enum):
    """What the system should do with a generated answer."""

    SHOW = "SHOW"
    WARN = "WARN"
    REFUSE = "REFUSE"


@dataclass
class DetectorResult:
    """One detector's support score. 1.0 means fully supported."""

    name: str
    """``nli``, ``selfcheck``, ``factuality``, or ``citation``."""

    score: float
    """Overall support in ``[0, 1]``. 1.0 means fully supported."""

    sentence_scores: list[float]
    """Per-sentence support, in the same order as ``DetectionReport.sentences``."""

    details: dict = field(default_factory=dict)
    """Method-specific evidence (chunk ids, claims, similarities, and so on)."""

    skipped: bool = False
    """True when the detector could not run."""

    skip_reason: str = ""
    """Why the detector was skipped, when ``skipped`` is True."""


@dataclass
class DetectionReport:
    """Every detector result for one answer, plus the sentences they scored."""

    results: list[DetectorResult]
    sentences: list[str]


@dataclass
class FinalResponse:
    """The answer the user should see, plus the detection evidence."""

    query: str
    answer: str
    """Original answer, or ``REFUSAL_TEXT`` when the decision is ``REFUSE``."""

    decision: Decision
    confidence: float
    report: DetectionReport | None
    warning: str | None
    generated: GeneratedAnswer
    """The Module 4 object, left unchanged."""
