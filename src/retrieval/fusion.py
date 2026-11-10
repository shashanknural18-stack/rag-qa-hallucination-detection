"""Reciprocal rank fusion for dense and sparse candidate lists."""

from src.config import RRF_K
from src.schemas import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], k: int = RRF_K
) -> list[RetrievedChunk]:
    """Fuse ranked lists by chunk_id without mutating the inputs."""
    if not rankings:
        return []

    fused_scores: dict[int, float] = {}
    merged: dict[int, RetrievedChunk] = {}

    for ranking in rankings:
        for position, chunk in enumerate(ranking, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (k + position)
            )
            current = merged.get(chunk.chunk_id)
            if current is None:
                merged[chunk.chunk_id] = RetrievedChunk(
                    text=chunk.text,
                    source=chunk.source,
                    page=chunk.page,
                    chunk_id=chunk.chunk_id,
                    dense_score=chunk.dense_score,
                    bm25_score=chunk.bm25_score,
                    rerank_score=chunk.rerank_score,
                    rank=0,
                )
                continue
            if chunk.dense_score != 0.0:
                current.dense_score = chunk.dense_score
            if chunk.bm25_score is not None:
                current.bm25_score = chunk.bm25_score

    ordered_ids = sorted(
        fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True
    )
    return [merged[chunk_id] for chunk_id in ordered_ids]
