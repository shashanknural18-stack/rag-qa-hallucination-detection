# src/ingestion/embedder.py
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode_batch(self, texts, batch_size=32):
        return self.model.encode(
            texts, batch_size=batch_size,
            normalize_embeddings=True,   # unit vectors, so inner product = cosine similarity
            show_progress_bar=True,
        ).astype("float32")              # FAISS requires float32