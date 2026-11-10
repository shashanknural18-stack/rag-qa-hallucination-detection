"""Data contract for a generated answer."""

from dataclasses import dataclass, field

from src.schemas import RetrievedChunk


@dataclass
class GeneratedAnswer:
    """A generated answer with citation metadata and optional stochastic samples."""

    query: str
    """The raw (whitespace-normalized) user query."""

    answer: str
    """The text produced by the LLM."""

    cited_ids: list[int]
    """Valid ``[n]`` markers found in the answer (1-based, deduplicated)."""

    invalid_citations: list[int]
    """``[n]`` markers that do not map to any retrieved chunk."""

    chunks: list[RetrievedChunk]
    """The retrieved chunks fed into the prompt (same objects as RetrievalResult)."""

    is_refusal: bool
    """True when the model refused to answer or no context was available."""

    samples: list[str] = field(default_factory=list)
    """Stochastic re-generations used later by SelfCheckGPT (empty by default)."""

    model_name: str = ""
    """Identifier of the GGUF model file used."""

    temperature: float = 0.0
    """Temperature used for the primary generation call."""

