"""Embedding generator: wraps SentenceTransformer for CPU-based encoding."""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL


class EmbeddingGenerator:
    """Generates normalized embeddings using a SentenceTransformer model."""

    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        self.model = SentenceTransformer(model_name, device="cpu")

    @property
    def dim(self) -> int:
        """Return the dimensionality of the embeddings."""
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts into a (n, dim) float32 array of unit vectors."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into a (dim,) float32 unit vector."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )
        return np.asarray(embedding, dtype=np.float32)
