"""Tests for Module 1 (Document Ingestion)."""

import json

import numpy as np
import pytest
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, EMBED_MODEL
from src.ingestion.loader import file_hash, load_file
from src.ingestion.chunker import chunk_documents, get_tokenizer
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.vector_store import VectorStore
from src.ingestion.pipeline import ingest


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder() -> EmbeddingGenerator:
    """Module-scoped embedding generator (loads the model once)."""
    return EmbeddingGenerator()


# ── Loader tests ─────────────────────────────────────────────────────────────

def test_loader_cleans_and_tags(tmp_path):
    """Whitespace is collapsed and metadata is set correctly."""
    f = tmp_path / "a.txt"
    f.write_text("Hello   world\n\n\nsecond   line", encoding="utf-8")
    docs = load_file(f)
    assert len(docs) == 1
    assert docs[0].page_content == "Hello world\n\nsecond line"
    assert docs[0].metadata["source"] == "a.txt"


def test_loader_skips_empty(tmp_path):
    """Whitespace-only files produce no documents."""
    f = tmp_path / "blank.txt"
    f.write_text("   \n\n   ", encoding="utf-8")
    docs = load_file(f)
    assert docs == []


def test_file_hash_stable(tmp_path):
    """Same file content gives the same hash."""
    f = tmp_path / "test.txt"
    f.write_text("deterministic content", encoding="utf-8")
    assert file_hash(f) == file_hash(f)


# ── Chunker tests ────────────────────────────────────────────────────────────

def test_chunk_size_in_tokens():
    """Every chunk must be <= CHUNK_SIZE tokens."""
    long_text = ("The quick brown fox jumps over the lazy dog. " * 200)
    docs = [Document(page_content=long_text, metadata={"source": "t.txt", "page": 0})]
    chunks = chunk_documents(docs)
    tokenizer = get_tokenizer()
    for chunk in chunks:
        n_tokens = len(tokenizer.encode(chunk.page_content, add_special_tokens=False))
        assert n_tokens <= CHUNK_SIZE, (
            f"Chunk has {n_tokens} tokens, max is {CHUNK_SIZE}"
        )


def test_chunk_ids_unique_and_metadata():
    """chunk_id values are sequential from start_id and source is preserved."""
    docs = [
        Document(
            page_content="Some text. " * 50,
            metadata={"source": "demo.txt", "page": 0},
        )
    ]
    chunks = chunk_documents(docs, start_id=10)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids == list(range(10, 10 + len(chunks)))
    for c in chunks:
        assert c.metadata["source"] == "demo.txt"


# ── Embedder tests ───────────────────────────────────────────────────────────

def test_embedding_shape_and_norm(embedder):
    """Batch encoding returns (n, 384) float32 vectors with unit norm."""
    vecs = embedder.encode_batch(["hello", "world"])
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ── VectorStore tests ────────────────────────────────────────────────────────

def test_index_matches_records(embedder):
    """index.ntotal must equal len(records) after add."""
    store = VectorStore()
    chunks = [
        Document(page_content="alpha", metadata={"source": "a.txt", "page": 0, "chunk_id": 0}),
        Document(page_content="beta",  metadata={"source": "a.txt", "page": 0, "chunk_id": 1}),
    ]
    vecs = embedder.encode_batch(["alpha", "beta"])
    store.add(vecs, chunks)
    assert store.index.ntotal == len(store.records) == 2


def test_semantic_retrieval(embedder):
    """The correct sentence surfaces as top-1 for a related query."""
    sentences = [
        "E-commerce platforms must acknowledge a return request within 48 hours.",
        "Photosynthesis converts carbon dioxide into glucose using sunlight.",
        "Consent to share personal data must be free, specific and informed.",
    ]
    chunks = [
        Document(page_content=s, metadata={"source": f"s{i}.txt", "page": 0, "chunk_id": i})
        for i, s in enumerate(sentences)
    ]
    vecs = embedder.encode_batch(sentences)
    store = VectorStore()
    store.add(vecs, chunks)

    query_vec = embedder.encode("How fast must a return be acknowledged?")
    results = store.search(query_vec, k=3)
    assert results[0][0]["text"] == sentences[0]


def test_save_load_roundtrip(tmp_path, embedder):
    """Saved and loaded store produces identical search results."""
    store = VectorStore()
    chunks = [
        Document(page_content="test sentence", metadata={"source": "t.txt", "page": 0, "chunk_id": 0}),
    ]
    vecs = embedder.encode_batch(["test sentence"])
    store.add(vecs, chunks)
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == len(store)

    query_vec = embedder.encode("test")
    r1 = store.search(query_vec, k=1)
    r2 = loaded.search(query_vec, k=1)
    assert r1[0][0]["text"] == r2[0][0]["text"]
    np.testing.assert_allclose(r1[0][1], r2[0][1], atol=1e-5)


def test_load_detects_mismatch(tmp_path, embedder):
    """Truncating records.json triggers a RuntimeError on load."""
    store = VectorStore()
    chunks = [
        Document(page_content="a", metadata={"source": "a.txt", "page": 0, "chunk_id": 0}),
        Document(page_content="b", metadata={"source": "a.txt", "page": 0, "chunk_id": 1}),
    ]
    vecs = embedder.encode_batch(["a", "b"])
    store.add(vecs, chunks)
    store.save(tmp_path)

    # Truncate records to create mismatch
    records_path = tmp_path / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records_path.write_text(
        json.dumps(records[:1], ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="mismatch"):
        VectorStore.load(tmp_path)


# ── Pipeline tests ───────────────────────────────────────────────────────────

def test_pipeline_is_incremental(tmp_path, embedder):
    """Running ingest twice on the same files adds 0 new chunks the second time."""
    raw = tmp_path / "raw"
    raw.mkdir()
    idx = tmp_path / "index"

    (raw / "sample.txt").write_text(
        "Test document for incremental pipeline check.",
        encoding="utf-8",
    )

    store1 = ingest(folder=raw, index_dir=idx, embedder=embedder)
    total1 = len(store1)
    assert total1 > 0

    store2 = ingest(folder=raw, index_dir=idx, embedder=embedder)
    total2 = len(store2)
    assert total2 == total1
