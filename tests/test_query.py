"""Tests for Module 2 (Query Processing)."""

import numpy as np
import pytest

from src.config import EMBED_DIM, MAX_QUERY_CHARS
from src.ingestion.embedder import EmbeddingGenerator
from src.query.processor import QueryProcessor
from src.schemas import ProcessedQuery


@pytest.fixture(scope="module")
def embedder() -> EmbeddingGenerator:
    """Load the shared embedding model once for this module."""
    return EmbeddingGenerator()


@pytest.fixture(scope="module")
def processor(embedder: EmbeddingGenerator) -> QueryProcessor:
    """Query processor bound to the shared embedder."""
    return QueryProcessor(embedder)


_CLEAN_EXAMPLES = [
    ("  What's   the GDP of INDIA in 2023??? ", "what is the gdp of india in 2023?"),
    ("Can't I return it?!", "cannot i return it?!"),
    ("What is Section 2(11) of the Act?", "what is section 2(11) of the act?"),
    ("Refund of ₹500 (18% GST)", "refund of ₹500 (18% gst)"),
    ("hello @#$ world", "hello world"),
]


def test_clean_examples():
    """Every documented cleaning example matches exactly."""
    for raw, expected in _CLEAN_EXAMPLES:
        assert QueryProcessor.clean(raw) == expected


def test_clean_preserves_legal_characters():
    """Section numbers, rupee amounts, and percentages survive cleaning."""
    cleaned = QueryProcessor.clean("section 2(11) costs ₹500 at 18%")
    assert "section 2(11)" in cleaned
    assert "₹500" in cleaned
    assert "18%" in cleaned


def test_clean_empty_raises():
    """Blank input and symbol-only input are rejected."""
    for raw in ("", "   ", "@#$%"):
        with pytest.raises(ValueError):
            QueryProcessor.clean(raw)


def test_clean_truncates_long_query():
    """Queries past the character cap are truncated to the limit."""
    cleaned = QueryProcessor.clean("word " * 600)
    assert len(cleaned) <= MAX_QUERY_CHARS


def test_process_shapes(processor: QueryProcessor):
    """process returns a unit vector and keeps raw case."""
    result = processor.process("  What's   the Return policy?  ")
    assert isinstance(result, ProcessedQuery)
    assert result.vector.shape == (EMBED_DIM,)
    assert result.vector.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(result.vector), 1.0, atol=1e-5)
    assert result.raw == "What's the Return policy?"
    assert result.cleaned == result.cleaned.lower()
    assert result.cleaned.startswith("what is")


def test_query_and_chunk_embeddings_comparable(
    processor: QueryProcessor, embedder: EmbeddingGenerator
):
    """A relevant sentence is closer to the query than an irrelevant one."""
    processed = processor.process("How fast must a return be acknowledged?")
    relevant = embedder.encode(
        "E-commerce platforms must acknowledge a return request within 48 hours."
    )
    irrelevant = embedder.encode(
        "Photosynthesis converts sunlight into chemical energy in plants."
    )
    relevant_score = float(np.dot(processed.vector, relevant))
    irrelevant_score = float(np.dot(processed.vector, irrelevant))
    assert relevant_score > irrelevant_score
