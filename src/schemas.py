"""Shared data contracts for query processing, retrieval, and later modules."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ProcessedQuery:
    """A user query after whitespace normalization, cleaning, and embedding."""

    raw: str
    cleaned: str
    vector: np.ndarray


@dataclass
class RetrievedChunk:
    """One retrieved passage with scores from each retrieval stage."""

    text: str
    source: str
    page: int
    chunk_id: int
    dense_score: float = 0.0
    bm25_score: float | None = None
    rerank_score: float | None = None
    rank: int = 0

    @property
    def score(self) -> float:
        """Return the rerank score when present, otherwise the dense score."""
        if self.rerank_score is not None:
            return self.rerank_score
        return self.dense_score

    @property
    def page_display(self) -> int:
        """Return the 1-based page number used in citations."""
        return self.page + 1


@dataclass
class RetrievalResult:
    """Final retrieval output consumed by generation and hallucination checks."""

    query: ProcessedQuery
    chunks: list[RetrievedChunk]
    num_candidates: int
    reranked: bool

    @property
    def is_empty(self) -> bool:
        """Return True when no chunks were kept."""
        return len(self.chunks) == 0


def chunk_from_record(
    record: dict,
    dense_score: float = 0.0,
    bm25_score: float | None = None,
) -> RetrievedChunk:
    """Build a RetrievedChunk from a Module 1 vector-store record."""
    metadata = record["metadata"]
    return RetrievedChunk(
        text=record["text"],
        source=metadata["source"],
        page=metadata["page"],
        chunk_id=metadata["chunk_id"],
        dense_score=dense_score,
        bm25_score=bm25_score,
    )
