"""Entailment check: each answer sentence against every retrieved chunk."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import NLI_ENTAIL_THRESHOLD
from src.detection.base import BaseDetector, clamp_unit
from src.detection.schemas import DetectorResult

if TYPE_CHECKING:
    from src.generation.schemas import GeneratedAnswer

logger = logging.getLogger(__name__)


class NLIDetector(BaseDetector):
    """Score sentences by the best entailment probability over retrieved chunks.

    A sentence is supported when that maximum is at least
    ``NLI_ENTAIL_THRESHOLD``. The detector score is the fraction of supported
    sentences. ``sentence_scores`` holds the max entailment itself.

    ``details["best_chunk_ids"]`` is the 1-based chunk with the highest
    entailment. ``details["contradiction_probs"]`` is the highest contradiction
    probability against any chunk, which the aggregator uses as a hard cap.

    Parameters
    ----------
    nli_model:
        Shared NLI model (real or a test double) with ``entailment_probs_batch``.
    """

    name = "nli"

    def __init__(self, nli_model) -> None:
        self.nli = nli_model

    def detect(
        self,
        generated: GeneratedAnswer,
        sentences: list[str],
    ) -> DetectorResult:
        """Compare each sentence with every chunk and keep the best entailment."""
        if not sentences:
            return DetectorResult(
                name=self.name,
                score=0.0,
                sentence_scores=[],
                details={"best_chunk_ids": [], "contradiction_probs": []},
            )
        if not generated.chunks:
            return DetectorResult(
                name=self.name,
                score=0.0,
                sentence_scores=[0.0] * len(sentences),
                details={
                    "best_chunk_ids": [0] * len(sentences),
                    "contradiction_probs": [0.0] * len(sentences),
                },
            )

        pairs: list[tuple[str, str]] = []
        for sentence in sentences:
            for chunk in generated.chunks:
                pairs.append((chunk.text, sentence))

        probs = self.nli.entailment_probs_batch(pairs)
        if len(probs) != len(pairs):
            raise RuntimeError(
                f"NLI returned {len(probs)} rows for {len(pairs)} pairs."
            )

        n_chunks = len(generated.chunks)
        sentence_scores: list[float] = []
        best_chunk_ids: list[int] = []
        contradiction_probs: list[float] = []
        supported = 0

        for sent_idx in range(len(sentences)):
            start = sent_idx * n_chunks
            rows = probs[start : start + n_chunks]
            best_entailment = -1.0
            best_chunk = 1
            max_contradiction = 0.0
            for chunk_offset, row in enumerate(rows):
                entailment = float(row.get("entailment", 0.0))
                contradiction = float(row.get("contradiction", 0.0))
                if contradiction > max_contradiction:
                    max_contradiction = contradiction
                if entailment > best_entailment:
                    best_entailment = entailment
                    best_chunk = chunk_offset + 1
            best_entailment = clamp_unit(best_entailment)
            sentence_scores.append(best_entailment)
            best_chunk_ids.append(best_chunk)
            contradiction_probs.append(clamp_unit(max_contradiction))
            if best_entailment >= NLI_ENTAIL_THRESHOLD:
                supported += 1

        score = supported / len(sentences)
        logger.info("nli score %.3f over %d sentences", score, len(sentences))
        return DetectorResult(
            name=self.name,
            score=clamp_unit(score),
            sentence_scores=sentence_scores,
            details={
                "best_chunk_ids": best_chunk_ids,
                "contradiction_probs": contradiction_probs,
            },
        )
