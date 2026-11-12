"""Run every detector and turn the report into a FinalResponse."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

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
        mode: str = "FULL",
        chunk_vectors: Any = None,
    ) -> FinalResponse:
        """Detect hallucinations in *generated* and choose SHOW, WARN, or REFUSE.

        A generation refusal, or an answer with no chunks, skips every detector
        and returns ``REFUSE`` with the original no-answer text and confidence 0.
        Otherwise cheap detectors (citation, NLI) run first. In FAST mode, expensive
        detectors (SelfCheck, factuality judge) run only if confidence falls in the
        WARN band; in FULL mode they run unless explicitly disabled.

        Parameters
        ----------
        generated:
            Module 4 output, including chunks and any SelfCheck samples.
        skip_judge:
            When True, the factuality judge is marked skipped and not called.
        mode:
            Execution mode ("FAST" or "FULL").
        chunk_vectors:
            Pre-computed chunk embeddings to avoid re-embedding chunks for citations.

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

        from src.config import THRESHOLD_SHOW, THRESHOLD_WARN, WEIGHTS

        sentences = split_sentences(generated.answer)

        # 1. Run cheap detectors first: citation and NLI
        cit_res = self._run_one(self.citation_detector, generated, sentences, chunk_vectors=chunk_vectors)
        nli_res = self._run_one(self.nli_detector, generated, sentences)

        # 2. Compute preliminary confidence from cheap detectors
        cheap_active = [r for r in (cit_res, nli_res) if not r.skipped and r.name in WEIGHTS]
        if cheap_active:
            w_sum = sum(WEIGHTS[r.name] for r in cheap_active)
            prelim_conf = sum(WEIGHTS[r.name] * r.score for r in cheap_active) / w_sum
        else:
            prelim_conf = 0.0

        # 3. Determine if expensive detectors (SelfCheck & Judge) should run
        is_fast = mode.upper() == "FAST"
        in_warn_band = (THRESHOLD_WARN <= prelim_conf < THRESHOLD_SHOW)

        run_expensive = (not is_fast) or in_warn_band

        # SelfCheck
        if not generated.samples:
            sc_res = DetectorResult(
                name="selfcheck",
                score=0.0,
                sentence_scores=[0.0] * len(sentences),
                details={},
                skipped=True,
                skip_reason="no samples",
            )
        elif run_expensive:
            sc_res = self._run_one(self.selfcheck_detector, generated, sentences)
        else:
            sc_res = DetectorResult(
                name="selfcheck",
                score=0.0,
                sentence_scores=[0.0] * len(sentences),
                details={},
                skipped=True,
                skip_reason="conditional_skip",
            )

        # Factuality Judge
        if skip_judge:
            judge_res = DetectorResult(
                name="factuality",
                score=0.0,
                sentence_scores=[0.0] * len(sentences),
                details={},
                skipped=True,
                skip_reason="disabled",
            )
        elif run_expensive:
            judge_res = self._run_one(self.factuality_judge, generated, sentences)
        else:
            judge_res = DetectorResult(
                name="factuality",
                score=0.0,
                sentence_scores=[0.0] * len(sentences),
                details={},
                skipped=True,
                skip_reason="conditional_skip",
            )

        results = [nli_res, sc_res, judge_res, cit_res]

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

    def _run_one(
        self,
        detector,
        generated: GeneratedAnswer,
        sentences: list[str],
        **kwargs: Any,
    ) -> DetectorResult:
        """Run one detector, converting any exception into a skipped result."""
        from src.profiling import stage

        stage_name = "judge" if detector.name == "factuality" else detector.name
        try:
            with stage(stage_name):
                # Only pass chunk_vectors if detector accepts it
                if "chunk_vectors" in kwargs and hasattr(detector, "detect"):
                    import inspect
                    sig = inspect.signature(detector.detect)
                    if "chunk_vectors" in sig.parameters:
                        return detector.detect(generated, sentences, chunk_vectors=kwargs["chunk_vectors"])
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
