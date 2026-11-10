"""End-to-end integration tests using real local models.

All tests are marked @pytest.mark.slow and skipped automatically if models are not cached.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from src.config import LLM_MODEL_PATH, NLI_MODEL, RAW_DIR
from src.detection.schemas import Decision

# Skip if GGUF model does not exist locally
_HAS_GGUF = Path(LLM_MODEL_PATH).exists()

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _HAS_GGUF, reason=f"GGUF model not found at {LLM_MODEL_PATH}"),
]


@pytest.fixture(scope="module")
def real_pipeline(tmp_path_factory):
    """Set up a temporary ingestion index with the two sample documents."""
    temp_dir = tmp_path_factory.mktemp("e2e_data")
    raw_dir = temp_dir / "raw"
    index_dir = temp_dir / "index"
    raw_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)

    # Copy standard raw sample files
    shutil.copy2(RAW_DIR / "consumer_rights.txt", raw_dir / "consumer_rights.txt")
    shutil.copy2(RAW_DIR / "data_privacy.txt", raw_dir / "data_privacy.txt")

    # Ingest
    from src.ingestion.pipeline import ingest
    store = ingest(folder=raw_dir, index_dir=index_dir)

    # Create real pipeline
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline(store=store)
    return pipeline


def test_e2e_return_acknowledgment(real_pipeline):
    """Query 1: Grounded answer from consumer_rights.txt."""
    output = real_pipeline.ask("How fast must a return be acknowledged?", num_samples=0)
    assert output.final.decision != Decision.REFUSE
    assert len(output.final.generated.cited_ids) > 0
    assert len(output.final.generated.invalid_citations) == 0

    # Ensure source cited matches consumer_rights.txt
    sources = [c["source"] for c in output.chunks if c["rank"] in output.final.generated.cited_ids]
    assert any("consumer_rights.txt" in s for s in sources)


def test_e2e_id_number_mandatory(real_pipeline):
    """Query 2: Grounded answer from data_privacy.txt."""
    output = real_pipeline.ask("Can a platform force me to share my ID number?", num_samples=0)
    assert output.final.decision != Decision.REFUSE
    assert len(output.final.generated.invalid_citations) == 0

    sources = [c["source"] for c in output.chunks if c["rank"] in output.final.generated.cited_ids]
    assert any("data_privacy.txt" in s for s in sources)


def test_e2e_long_aadhaar_multi_document(real_pipeline):
    """Query 3: Long Aadhaar query -> top chunks should include BOTH documents."""
    q = (
        "I ordered a phone online, it arrived with a manufacturing defect, and when I tried to return it "
        "the platform said it would only process my return if I shared my Aadhaar number with their third-party "
        "pickup partner. I wasn't given a clear option to refuse. What are my rights here?"
    )
    output = real_pipeline.ask(q, num_samples=0)
    sources = {os.path.basename(c["source"]) for c in output.chunks}
    assert "consumer_rights.txt" in sources
    assert "data_privacy.txt" in sources


def test_e2e_out_of_corpus_refuses(real_pipeline):
    """Query 4: Out-of-corpus question must refuse and never show hallucinated answer."""
    output = real_pipeline.ask("Who won the 2018 FIFA World Cup?", num_samples=0)
    assert output.final.decision == Decision.REFUSE
    # Ensure answer text shown to user is the refusal text, not hallucinated text
    assert "2018" not in output.final.answer
    assert "France" not in output.final.answer


def test_e2e_planted_false_premise(real_pipeline):
    """Query 5: False premise -> WARN or REFUSE."""
    output = real_pipeline.ask(
        "Why does consumer protection law mandate a 90-day acknowledgment window and mandatory Aadhaar submission?",
        num_samples=0,
    )
    assert output.final.decision in (Decision.WARN, Decision.REFUSE)
    assert len(output.final.generated.invalid_citations) == 0

