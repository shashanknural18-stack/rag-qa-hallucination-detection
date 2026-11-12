"""Check that every sentence cites a chunk that actually matches it."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.config import CITATION_SUPPORT_THRESHOLD
from src.detection.base import BaseDetector, clamp_unit
from src.detection.schemas import DetectorResult
from src.detection.sentence_splitter import sentence_citations

if TYPE_CHECKING:
    from src.generation.schemas import GeneratedAnswer

logger = logging.getLogger(__name__)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity of two vectors. Zero vectors compare as 0."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


class CitationDetector(BaseDetector):
    """Require every sentence to cite only chunks that are similar to it.

    A sentence with no citation is unsupported. An id outside ``1..num_chunks``
    is unsupported. A valid id is supported when the cosine similarity of the
    sentence embedding and that chunk's embedding is at least
    ``CITATION_SUPPORT_THRESHOLD``. The sentence passes only when every one of
    its citations is valid and supported. The detector score is the fraction
    of sentences that pass.

    Parameters
    ----------
    embedder:
        Shared :class:`src.ingestion.embedder.EmbeddingGenerator`, or a double
        that implements ``encode_batch``.
    """

    name = "citation"

    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def detect(
        self,
        generated: GeneratedAnswer,
        sentences: list[str],
        chunk_vectors: np.ndarray | None = None,
    ) -> DetectorResult:
        """Score citation validity and semantic support for each sentence."""
        if not sentences:
            return DetectorResult(
                name=self.name,
                score=0.0,
                sentence_scores=[],
                details={"per_sentence": []},
            )

        citations = sentence_citations(generated.answer)
        if len(citations) != len(sentences):
            citations = (citations + [[] for _ in sentences])[: len(sentences)]

        if chunk_vectors is not None and len(chunk_vectors) == len(generated.chunks):
            sentence_vectors = np.asarray(self.embedder.encode_batch(list(sentences)), dtype=np.float32)
        else:
            texts = list(sentences) + [chunk.text for chunk in generated.chunks]
            vectors = np.asarray(self.embedder.encode_batch(texts), dtype=np.float32)
            sentence_vectors = vectors[: len(sentences)]
            chunk_vectors = vectors[len(sentences) :]
        num_chunks = len(generated.chunks)

        sentence_scores: list[float] = []
        per_sentence: list[dict] = []
        passed = 0

        for index, cited_ids in enumerate(citations):
            invalid_ids = [cid for cid in cited_ids if cid < 1 or cid > num_chunks]
            valid_ids = [cid for cid in cited_ids if 1 <= cid <= num_chunks]
            similarities: list[dict] = []
            all_similar = True
            for cid in valid_ids:
                similarity = _cosine(sentence_vectors[index], chunk_vectors[cid - 1])
                similarities.append({"id": cid, "similarity": similarity})
                if similarity < CITATION_SUPPORT_THRESHOLD:
                    all_similar = False
            supported = bool(cited_ids) and not invalid_ids and all_similar
            sentence_score = 1.0 if supported else 0.0
            if supported:
                passed += 1
            sentence_scores.append(sentence_score)
            per_sentence.append(
                {
                    "cited_ids": list(cited_ids),
                    "invalid_ids": invalid_ids,
                    "similarities": similarities,
                    "supported": supported,
                }
            )

        score = passed / len(sentences)
        logger.info("citation score %.3f over %d sentences", score, len(sentences))
        return DetectorResult(
            name=self.name,
            score=clamp_unit(score),
            sentence_scores=sentence_scores,
            details={"per_sentence": per_sentence},
        )
