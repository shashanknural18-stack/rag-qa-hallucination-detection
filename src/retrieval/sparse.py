"""Optional BM25 retriever over the same records as the vector store."""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import TOP_K_RETRIEVE
from src.ingestion.vector_store import VectorStore
from src.schemas import RetrievedChunk, chunk_from_record


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens for BM25."""
    return re.findall(r"\w+", text.lower())


class SparseRetriever:
    """BM25 search built once from the vector store records."""

    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self._bm25: BM25Okapi | None = None
        if store.records:
            corpus = [tokenize(record["text"]) for record in store.records]
            self._bm25 = BM25Okapi(corpus)

    def search(
        self, query_text: str, k: int = TOP_K_RETRIEVE
    ) -> list[RetrievedChunk]:
        """Return up to k chunks with a positive BM25 score, best first."""
        if self._bm25 is None or len(self.store) == 0:
            return []
        scores = np.asarray(self._bm25.get_scores(tokenize(query_text)), dtype=np.float64)
        order = np.argsort(-scores)
        chunks: list[RetrievedChunk] = []
        for index in order:
            score = float(scores[int(index)])
            if score <= 0.0:
                break
            chunks.append(
                chunk_from_record(
                    self.store.records[int(index)],
                    dense_score=0.0,
                    bm25_score=score,
                )
            )
            if len(chunks) >= k:
                break
        return chunks
