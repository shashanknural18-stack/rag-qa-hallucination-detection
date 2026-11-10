"""NLI-based SelfCheckGPT: answer sentences against stochastic samples."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.detection.base import BaseDetector, clamp_unit
from src.detection.schemas import DetectorResult

if TYPE_CHECKING:
    from src.generation.schemas import GeneratedAnswer

logger = logging.getLogger(__name__)


def _consistency(probs: dict) -> float:
    """Support of one sample: entailment minus contradiction, clipped to [0, 1]."""
    entailment = float(probs.get("entailment", 0.0))
    contradiction = float(probs.get("contradiction", 0.0))
    return clamp_unit(entailment - contradiction)


class SelfCheckDetector(BaseDetector):
    """Measure whether sampled answers still entail each sentence.

    For every sentence, each sample is the NLI premise and the sentence is the
    hypothesis. Per-sample consistency is ``entailment - contradiction`` clipped
    to ``[0, 1]``. The sentence score is the mean over samples, and the detector
    score is the mean over sentences.

    If ``generated.samples`` is empty the detector is skipped with reason
    ``"no samples"``.

    Parameters
    ----------
    nli_model:
        The same shared NLI model used by :class:`NLIDetector`.
    """

    name = "selfcheck"

    def __init__(self, nli_model) -> None:
        self.nli = nli_model

    def detect(
        self,
        generated: GeneratedAnswer,
        sentences: list[str],
    ) -> DetectorResult:
        """Score each sentence against ``generated.samples``."""
        samples = [sample for sample in generated.samples if sample and sample.strip()]
        if not samples:
            return DetectorResult(
                name=self.name,
                score=0.0,
                sentence_scores=[],
                details={},
                skipped=True,
                skip_reason="no samples",
            )
        if not sentences:
            return DetectorResult(
                name=self.name,
                score=0.0,
                sentence_scores=[],
                details={"sample_consistencies": []},
            )

        pairs: list[tuple[str, str]] = []
        for sentence in sentences:
            for sample in samples:
                pairs.append((sample, sentence))

        probs = self.nli.entailment_probs_batch(pairs)
        if len(probs) != len(pairs):
            raise RuntimeError(
                f"NLI returned {len(probs)} rows for {len(pairs)} SelfCheck pairs."
            )

        n_samples = len(samples)
        sentence_scores: list[float] = []
        per_sample: list[list[float]] = []
        for sent_idx in range(len(sentences)):
            start = sent_idx * n_samples
            row_scores = [_consistency(row) for row in probs[start : start + n_samples]]
            per_sample.append(row_scores)
            sentence_scores.append(sum(row_scores) / len(row_scores))

        score = sum(sentence_scores) / len(sentence_scores)
        logger.info("selfcheck score %.3f over %d samples", score, n_samples)
        return DetectorResult(
            name=self.name,
            score=clamp_unit(score),
            sentence_scores=[clamp_unit(value) for value in sentence_scores],
            details={"sample_consistencies": per_sample},
        )
