"""Central configuration for the RAG QA project."""

import os
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = ROOT / "data" / "raw"
INDEX_DIR: Path = ROOT / "data" / "index"
EVAL_RESULTS_DIR: Path = ROOT / "eval" / "results"

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

# Module 4 (generation) — Mistral-7B-Instruct Q4 GGUF via llama-cpp-python
LLM_MODEL_PATH: str = os.environ.get(
    "LLM_MODEL_PATH",
    str(ROOT / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
)
LLM_N_CTX: int = 4096
LLM_N_THREADS: int = int(os.environ.get("LLM_N_THREADS", str(os.cpu_count() or 4)))
LLM_TEMPERATURE: float = 0.1
LLM_MAX_TOKENS: int = 400
SAMPLE_TEMPERATURE: float = 0.7
NUM_SAMPLES: int = 5
NO_ANSWER_TEXT: str = "I don't have enough information in the documents to answer this."

# Module 5 (hallucination detection)
NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
NLI_ENTAIL_THRESHOLD: float = 0.5
CITATION_SUPPORT_THRESHOLD: float = 0.45
JUDGE_MAX_TOKENS: int = 500
CONTRADICTION_CAP_THRESHOLD: float = 0.8
MIN_SENTENCE_WORDS: int = 3
WEIGHTS: dict[str, float] = {
    "nli": 0.35,
    "selfcheck": 0.25,
    "factuality": 0.25,
    "citation": 0.15,
}
THRESHOLD_SHOW: float = float(os.environ.get("THRESHOLD_SHOW", "0.75"))
THRESHOLD_WARN: float = float(os.environ.get("THRESHOLD_WARN", "0.50"))
REFUSAL_TEXT: str = "I can't verify this answer against the documents, so I won't show it."
WARNING_TEXT: str = "Some parts of this answer could not be fully verified. Check the sources."

# Module 6 (Streamlit UI)
UI_TITLE: str = "Document QA with Hallucination Detection"
UI_DEFAULT_TOP_K: int = 5
UI_MAX_TOP_K: int = 10
UI_MAX_UPLOAD_MB: int = 20
ALLOWED_UPLOAD_TYPES: list[str] = ["pdf", "txt", "md", "docx"]
ENABLE_JUDGE_DEFAULT: bool = True
NUM_SAMPLES_UI: int = 3


def validate_config() -> None:
    """Validate project configuration values.

    Raises
    ------
    ValueError
        If weights do not sum to 1.0, THRESHOLD_SHOW is not strictly greater
        than THRESHOLD_WARN, or the parent directory of LLM_MODEL_PATH does not exist.
    """
    weight_sum = sum(WEIGHTS.values())
    if abs(weight_sum - 1.0) > 1e-4:
        raise ValueError(f"WEIGHTS must sum to 1.0; got {weight_sum:.4f}")

    if THRESHOLD_SHOW <= THRESHOLD_WARN:
        raise ValueError(
            f"THRESHOLD_SHOW ({THRESHOLD_SHOW}) must be strictly greater than "
            f"THRESHOLD_WARN ({THRESHOLD_WARN})"
        )

    model_parent = Path(LLM_MODEL_PATH).parent
    if not model_parent.exists():
        raise ValueError(
            f"Directory for LLM_MODEL_PATH does not exist: {model_parent}"
        )
