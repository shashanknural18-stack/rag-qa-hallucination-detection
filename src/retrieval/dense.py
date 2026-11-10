"""Dense retriever: exact inner-product search over the Module 1 store."""

import numpy as np

from src.config import TOP_K_RETRIEVE
from src.ingestion.vector_store import VectorStore
from src.schemas import RetrievedChunk, chunk_from_record


class DenseRetriever:
    """Returns the nearest stored chunks for an already-embedded query."""

    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def search(
        self, query_vec: np.ndarray, k: int = TOP_K_RETRIEVE
    ) -> list[RetrievedChunk]:
        """Return up to k chunks, best cosine similarity first."""
        if query_vec.shape != (self.store.dim,):
            raise ValueError(
                f"query vector shape {query_vec.shape} != ({self.store.dim},)"
            )
        hits = self.store.search(query_vec, k)
        return [
            chunk_from_record(record, dense_score=score) for record, score in hits
        ]
