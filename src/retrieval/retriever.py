"""Module 3 entry point: dense search, optional BM25 fusion, then rerank."""

import logging
from pathlib import Path

from src.config import (
    INDEX_DIR,
    MIN_RERANK_SCORE,
    TOP_K_FINAL,
    TOP_K_RETRIEVE,
    USE_BM25,
)
from src.ingestion.vector_store import VectorStore
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse import SparseRetriever
from src.schemas import ProcessedQuery, RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)


class RetrievalModule:
    """Retrieves a short, reranked evidence list for one processed query."""

    def __init__(
        self,
        store: VectorStore,
        reranker: CrossEncoderReranker | None = None,
        use_bm25: bool = USE_BM25,
        top_k_retrieve: int = TOP_K_RETRIEVE,
        top_k_final: int = TOP_K_FINAL,
        min_rerank_score: float = MIN_RERANK_SCORE,
    ) -> None:
        self.store = store
        self.reranker = reranker
        self.use_bm25 = use_bm25
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final
        self.min_rerank_score = min_rerank_score
        self.dense = DenseRetriever(store)
        self.sparse = SparseRetriever(store) if use_bm25 else None

    @classmethod
    def from_index(
        cls,
        index_dir: Path = INDEX_DIR,
        use_rerank: bool = True,
        **kwargs,
    ) -> "RetrievalModule":
        """Load a saved index and optionally attach the cross-encoder."""
        index_dir = Path(index_dir)
        if not VectorStore.exists(index_dir):
            raise FileNotFoundError(
                f"No index found at {index_dir}. Run: python -m src.ingestion.pipeline"
            )
        store = VectorStore.load(index_dir)
        reranker = CrossEncoderReranker() if use_rerank else None
        return cls(store, reranker=reranker, **kwargs)

    def retrieve(self, query: ProcessedQuery) -> RetrievalResult:
        """Run dense retrieval, optional fusion, and optional reranking."""
        if len(self.store) == 0:
            return RetrievalResult(query=query, chunks=[], num_candidates=0, reranked=False)

        if query.vector.shape != (self.store.dim,):
            raise ValueError(
                f"query vector shape {query.vector.shape} != ({self.store.dim},)"
            )

        candidates = self.dense.search(query.vector, self.top_k_retrieve)
        if self.use_bm25 and self.sparse is not None:
            sparse_hits = self.sparse.search(query.cleaned, self.top_k_retrieve)
            candidates = reciprocal_rank_fusion([candidates, sparse_hits])[
                : self.top_k_retrieve
            ]

        num_candidates = len(candidates)
        if self.reranker is not None:
            chunks = self.reranker.rerank(query.raw, candidates, self.top_k_final)
            chunks = [
                chunk
                for chunk in chunks
                if chunk.rerank_score is not None
                and chunk.rerank_score >= self.min_rerank_score
            ]
            chunks = _with_ranks(chunks)
            reranked = True
        else:
            chunks = _with_ranks(candidates[: self.top_k_final])
            reranked = False

        top_score = chunks[0].rerank_score if chunks else None
        logger.info(
            "candidates=%d kept=%d top_rerank_score=%s",
            num_candidates,
            len(chunks),
            f"{top_score:.4f}" if top_score is not None else "n/a",
        )
        return RetrievalResult(
            query=query,
            chunks=chunks,
            num_candidates=num_candidates,
            reranked=reranked,
        )


def _with_ranks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Assign 1-based ranks in the current list order."""
    for rank, chunk in enumerate(chunks, start=1):
        chunk.rank = rank
    return chunks
