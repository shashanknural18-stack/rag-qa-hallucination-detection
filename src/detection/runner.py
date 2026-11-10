"""Run every detector and turn the report into a FinalResponse."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import NO_ANSWER_TEXT, REFUSAL_TEXT, WARNING_TEXT
from src.detection.aggregator import aggregate
from src.detection.citation_detector import CitationDetector
from src.detection.factuality_judge import FactualityJudge
from src.detection.nli_detector import NLIDetector
from src.detection.schemas import Decision, DetectionReport, DetectorResult, FinalResponse
from src.detection.selfcheck_detector import SelfCheckDetector
from src.detection.sentence_splitter import split_sentences

if TYPE_CHECKING:
    from src.generation.schemas import GeneratedAnswer

logger = logging.getLogger(__name__)


class DetectionRunner:
    """Run NLI, SelfCheck, the factuality judge, and citation matching.

    Models are injected so each one is loaded once by the caller and shared.
    A detector that raises is recorded as skipped and does not stop the run.

    Parameters
    ----------
    nli:
        Shared NLI model used by the NLI and SelfCheck detectors.
    llm:
        Shared local LLM used by the factuality judge. May be ``None``.
    embedder:
        Shared embedder used by the citation detector.
    """

    def __init__(self, nli, llm, embedder) -> None:
        self.nli_detector = NLIDetector(nli)
        self.selfcheck_detector = SelfCheckDetector(nli)
        self.factuality_judge = FactualityJudge(llm)
        self.citation_detector = CitationDetector(embedder)

    def run(
        self,
        generated: GeneratedAnswer,
        skip_judge: bool = False,
    ) -> FinalResponse:
        """Detect hallucinations in *generated* and choose SHOW, WARN, or REFUSE.

        A generation refusal, or an answer with no chunks, skips every detector
        and returns ``REFUSE`` with the original no-answer text and confidence 0.
        Otherwise the answer is split into sentences, the four detectors run,
        and the aggregator picks the decision.

        ``REFUSE`` replaces the answer with ``REFUSAL_TEXT``. ``WARN`` keeps the
        answer and sets ``warning`` to ``WARNING_TEXT``. ``SHOW`` keeps the answer.

        Parameters
        ----------
        generated:
            Module 4 output, including chunks and any SelfCheck samples.
        skip_judge:
            When True, the factuality judge is marked skipped and not called.

        Returns
        -------
        FinalResponse
        """
        if generated.is_refusal or not generated.chunks:
            return FinalResponse(
                query=generated.query,
                answer=NO_ANSWER_TEXT,
                decision=Decision.REFUSE,
                confidence=0.0,
                report=None,
                warning=None,
                generated=generated,
            )

        sentences = split_sentences(generated.answer)
        detectors = [
            self.nli_detector,
            self.selfcheck_detector,
            self.factuality_judge,
            self.citation_detector,
        ]
        results: list[DetectorResult] = []
        for detector in detectors:
            if skip_judge and detector.name == "factuality":
                results.append(
                    DetectorResult(
                        name="factuality",
                        score=0.0,
                        sentence_scores=[],
                        details={},
                        skipped=True,
                        skip_reason="disabled",
                    )
                )
                continue
            results.append(self._run_one(detector, generated, sentences))

        report = DetectionReport(results=results, sentences=sentences)
        confidence, decision = aggregate(report)

        if decision is Decision.REFUSE:
            answer = REFUSAL_TEXT
            warning = None
        elif decision is Decision.WARN:
            answer = generated.answer
            warning = WARNING_TEXT
        else:
            answer = generated.answer
            warning = None

        logger.info("decision %s confidence %.3f", decision.value, confidence)
        return FinalResponse(
            query=generated.query,
            answer=answer,
            decision=decision,
            confidence=confidence,
            report=report,
            warning=warning,
            generated=generated,
        )

    def _run_one(self, detector, generated: GeneratedAnswer, sentences: list[str]) -> DetectorResult:
        """Run one detector, converting any exception into a skipped result."""
        try:
            return detector.detect(generated, sentences)
        except Exception as exc:
            logger.exception("detector %s failed", detector.name)
            return DetectorResult(
                name=detector.name,
                score=0.0,
                sentence_scores=[0.0] * len(sentences),
                details={"error": str(exc)},
                skipped=True,
                skip_reason=str(exc),
            )
