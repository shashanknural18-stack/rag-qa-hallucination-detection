"""Module 5: hallucination detection and the show / warn / refuse decision."""

from src.detection.aggregator import aggregate
from src.detection.citation_detector import CitationDetector
from src.detection.factuality_judge import FactualityJudge
from src.detection.nli_detector import NLIDetector
from src.detection.runner import DetectionRunner
from src.detection.schemas import Decision, FinalResponse
from src.detection.selfcheck_detector import SelfCheckDetector

__all__ = [
    "CitationDetector",
    "Decision",
    "DetectionRunner",
    "FactualityJudge",
    "FinalResponse",
    "NLIDetector",
    "SelfCheckDetector",
    "aggregate",
]
