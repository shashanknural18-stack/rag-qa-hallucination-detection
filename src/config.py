"""Central configuration for the RAG QA project."""

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = ROOT / "data" / "raw"
INDEX_DIR: Path = ROOT / "data" / "index"

EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM: int = 384

CHUNK_SIZE: int = 240        # tokens (model max is 256 including special tokens)
CHUNK_OVERLAP: int = 50      # tokens

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt", ".md", ".docx"}

# Module 2 (query processing) and Module 3 (retrieval)
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_BATCH_SIZE: int = 16
TOP_K_RETRIEVE: int = 10
TOP_K_FINAL: int = 5
USE_BM25: bool = False
RRF_K: int = 60
MIN_RERANK_SCORE: float = 0.0
MAX_QUERY_CHARS: int = 1000
