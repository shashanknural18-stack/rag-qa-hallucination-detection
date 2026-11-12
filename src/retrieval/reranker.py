"""Cross-encoder reranker for the short candidate list."""

import numpy as np
from sentence_transformers import CrossEncoder

from src.config import RERANK_BATCH_SIZE, RERANK_MODEL, TOP_K_FINAL
from src.schemas import RetrievedChunk


def _identity(values):
    """Return cross-encoder outputs unchanged so the sigmoid stays explicit."""
    return values


def _predict_logits(
    model: CrossEncoder,
    pairs: list[tuple[str, str]],
    batch_size: int = RERANK_BATCH_SIZE,
) -> np.ndarray:
    """Return raw cross-encoder logits, with the built-in sigmoid disabled."""
    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        activation_fn=_identity,
    )
    return np.asarray(scores, dtype=np.float64).reshape(-1)


class CrossEncoderReranker:
    """Scores query-chunk pairs on CPU and keeps the strongest matches."""

    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        self.model = CrossEncoder(model_name, device="cpu", max_length=512)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = TOP_K_FINAL,
        batch_size: int = RERANK_BATCH_SIZE,
    ) -> list[RetrievedChunk]:
        """Rerank chunks against the raw query and assign 1-based ranks."""
        if not chunks:
            return []

        # Truncate each chunk to 256 tokens for faster scoring
        truncated_pairs = [
            (query, " ".join(chunk.text.split()[:256]))
            for chunk in chunks
        ]
        logits = _predict_logits(self.model, truncated_pairs, batch_size=batch_size)
        probabilities = 1.0 / (1.0 + np.exp(-logits))

        updated = [
            RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                dense_score=chunk.dense_score,
                bm25_score=chunk.bm25_score,
                rerank_score=float(probability),
                rank=0,
            )
            for chunk, probability in zip(chunks, probabilities)
        ]
        updated.sort(key=lambda chunk: chunk.rerank_score or 0.0, reverse=True)
        kept = updated[:top_k]
        for rank, chunk in enumerate(kept, start=1):
            chunk.rank = rank
        return kept
