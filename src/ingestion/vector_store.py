"""FAISS-backed vector store with JSON record persistence."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document

from src.config import EMBED_DIM, INDEX_DIR

logger = logging.getLogger(__name__)


class VectorStore:
    """In-memory FAISS index paired with a JSON record list."""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self.records: list[dict] = []

    def __len__(self) -> int:
        """Return the number of vectors in the store."""
        return self.index.ntotal

    def add(self, vectors: np.ndarray, chunks: list[Document]) -> None:
        """Add vectors and their corresponding document chunks to the store."""
        if len(chunks) == 0:
            return
        expected_shape = (len(chunks), self.dim)
        if vectors.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: got {vectors.shape}, expected {expected_shape}"
            )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add(vectors)
        for chunk in chunks:
            self.records.append({
                "text": chunk.page_content,
                "metadata": dict(chunk.metadata),
            })

    def search(
        self, query_vec: np.ndarray, k: int = 10
    ) -> list[tuple[dict, float]]:
        """Return up to *k* (record, score) pairs sorted best-first."""
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        query_vec = np.ascontiguousarray(
            query_vec.reshape(1, -1), dtype=np.float32
        )
        scores, ids = self.index.search(query_vec, k)
        results: list[tuple[dict, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append((self.records[idx], float(score)))
        return results

    def save(self, folder: Path = INDEX_DIR) -> None:
        """Persist the FAISS index and records to disk."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        index_path = folder / "faiss.index"
        records_path = folder / "records.json"

        faiss.write_index(self.index, str(index_path))
        records_path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved index (%d vectors) to %s", self.index.ntotal, folder)

    @classmethod
    def load(cls, folder: Path = INDEX_DIR) -> "VectorStore":
        """Load a previously saved store from disk."""
        folder = Path(folder)
        index_path = folder / "faiss.index"
        records_path = folder / "records.json"

        index = faiss.read_index(str(index_path))
        records: list[dict] = json.loads(
            records_path.read_text(encoding="utf-8")
        )

        if index.ntotal != len(records):
            raise RuntimeError("index/records mismatch: re-run ingestion")

        store = cls(dim=index.d)
        store.index = index
        store.records = records
        return store

    @classmethod
    def exists(cls, folder: Path = INDEX_DIR) -> bool:
        """Return True only if both index files exist on disk."""
        folder = Path(folder)
        return (folder / "faiss.index").exists() and (
            folder / "records.json"
        ).exists()
