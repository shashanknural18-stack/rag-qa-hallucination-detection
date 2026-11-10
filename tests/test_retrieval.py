"""Tests for Module 3 (Retrieval)."""

import numpy as np
import pytest
from langchain_core.documents import Document

from src.config import EMBED_DIM, TOP_K_FINAL
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.pipeline import ingest
from src.ingestion.vector_store import VectorStore
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RetrievalModule
from src.retrieval.sparse import SparseRetriever
from src.schemas import ProcessedQuery, RetrievedChunk, chunk_from_record

_RETURN = "E-commerce platforms must acknowledge a return request within 48 hours."
_PHOTO = "Photosynthesis converts sunlight into chemical energy in plants."
_CONSENT = "Consent to share personal data must be free, specific, informed and unconditional."

_BM25_CORPUS = [
    "Rule 5 of the E-Commerce Rules requires a grievance to be acknowledged within 48 hours.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The Eiffel Tower was completed in 1889 in Paris.",
    "Consent to share personal data must be free, specific, informed and unconditional.",
    "A manufacturing defect qualifies as a deficiency under the Consumer Protection Act.",
    "Mount Everest is the highest mountain above sea level.",
    "Mitochondria produce most of the chemical energy needed by a cell.",
]

_CONSUMER_RIGHTS = (
    "Under the Consumer Protection Act, a manufacturing defect qualifies as a deficiency.\n"
    "The buyer is entitled to replacement, refund or repair.\n"
    "\n"
    "E-commerce platforms must acknowledge a return request within 48 hours "
    "and resolve it within 30 days.\n"
)
_DATA_PRIVACY = (
    "Consent to share personal data must be free, specific, informed and unconditional.\n"
    "\n"
    "A platform cannot make sharing an identity number a condition of processing "
    "a return unless the law requires it.\n"
)


@pytest.fixture(scope="module")
def embedder() -> EmbeddingGenerator:
    """Load the shared embedding model once for this module."""
    return EmbeddingGenerator()


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    """Load the cross-encoder once for this module."""
    return CrossEncoderReranker()


def _store_from_texts(
    embedder: EmbeddingGenerator,
    texts: list[str],
    sources: list[str] | None = None,
) -> VectorStore:
    """Build an in-memory store from sentences and real embeddings."""
    documents = [
        Document(
            page_content=text,
            metadata={
                "source": sources[index] if sources else f"s{index}.txt",
                "page": 0,
                "chunk_id": index,
            },
        )
        for index, text in enumerate(texts)
    ]
    store = VectorStore()
    store.add(embedder.encode_batch(texts), documents)
    return store


def _processed(embedder: EmbeddingGenerator, text: str) -> ProcessedQuery:
    """Build a ProcessedQuery without going through the cleaner."""
    return ProcessedQuery(raw=text, cleaned=text.lower(), vector=embedder.encode(text))


def test_chunk_from_record():
    """Record fields map onto a chunk, and score falls back to dense score."""
    record = {
        "text": "A manufacturing defect qualifies as a deficiency.",
        "metadata": {"source": "consumer_protection_act.pdf", "page": 4, "chunk_id": 17},
    }
    chunk = chunk_from_record(record, dense_score=0.42)
    assert chunk.text == record["text"]
    assert chunk.source == "consumer_protection_act.pdf"
    assert chunk.page == 4
    assert chunk.chunk_id == 17
    assert chunk.page_display == 5
    assert chunk.rerank_score is None
    assert chunk.score == pytest.approx(0.42)


def test_dense_top1(embedder: EmbeddingGenerator):
    """The 48-hour sentence is the nearest neighbor of the return query."""
    store = _store_from_texts(embedder, [_RETURN, _PHOTO, _CONSENT])
    hits = DenseRetriever(store).search(embedder.encode("How fast must a return be acknowledged?"))
    assert hits[0].text == _RETURN
    assert -1.0 <= hits[0].dense_score <= 1.0


def test_dense_wrong_dimension_raises(embedder: EmbeddingGenerator):
    """A vector with the wrong width is rejected before search."""
    store = _store_from_texts(embedder, [_RETURN])
    with pytest.raises(ValueError):
        DenseRetriever(store).search(np.zeros(10, dtype=np.float32))


def test_rrf_merges_and_orders():
    """A chunk present in both lists ranks first and keeps both scores."""
    shared = RetrievedChunk(text="shared", source="a.txt", page=0, chunk_id=1, dense_score=0.8)
    only_dense = RetrievedChunk(text="dense", source="a.txt", page=0, chunk_id=2, dense_score=0.4)
    only_sparse = RetrievedChunk(
        text="shared", source="a.txt", page=0, chunk_id=1, bm25_score=3.5
    )
    other_sparse = RetrievedChunk(
        text="sparse", source="b.txt", page=0, chunk_id=3, bm25_score=1.2
    )
    dense = [shared, only_dense]
    sparse = [only_sparse, other_sparse]
    dense_before = [(chunk.chunk_id, chunk.dense_score, chunk.bm25_score) for chunk in dense]
    sparse_before = [(chunk.chunk_id, chunk.dense_score, chunk.bm25_score) for chunk in sparse]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    assert fused[0].chunk_id == 1
    assert len({chunk.chunk_id for chunk in fused}) == len(fused)
    winner = next(chunk for chunk in fused if chunk.chunk_id == 1)
    assert winner.dense_score == pytest.approx(0.8)
    assert winner.bm25_score == pytest.approx(3.5)
    assert [(chunk.chunk_id, chunk.dense_score, chunk.bm25_score) for chunk in dense] == dense_before
    assert [(chunk.chunk_id, chunk.dense_score, chunk.bm25_score) for chunk in sparse] == sparse_before


def test_rrf_empty():
    """Fusion of no lists is an empty list."""
    assert reciprocal_rank_fusion([]) == []


def test_bm25_exact_term():
    """The Rule 5 sentence is the first positive BM25 hit."""
    documents = [
        Document(
            page_content=text,
            metadata={"source": f"s{index}.txt", "page": 0, "chunk_id": index},
        )
        for index, text in enumerate(_BM25_CORPUS)
    ]
    store = VectorStore()
    store.add(np.zeros((len(documents), EMBED_DIM), dtype=np.float32), documents)
    hits = SparseRetriever(store).search("Rule 5")
    assert hits[0].text == _BM25_CORPUS[0]
    assert all(hit.bm25_score is not None and hit.bm25_score > 0 for hit in hits)


def test_reranker_scores_sorted_and_bounded(reranker: CrossEncoderReranker):
    """Three kept chunks have bounded scores, descending order, and ranks 1..3."""
    chunks = [
        RetrievedChunk(text=text, source="s.txt", page=0, chunk_id=index, dense_score=0.1)
        for index, text in enumerate(
            [_PHOTO, _RETURN, _CONSENT, "Mount Everest is the highest mountain.", "The Eiffel Tower is in Paris."]
        )
    ]
    ranked = reranker.rerank("How fast must a return be acknowledged?", chunks, top_k=3)
    scores = [chunk.rerank_score for chunk in ranked]
    assert len(ranked) == 3
    assert scores == sorted(scores, reverse=True)
    assert all(score is not None and 0.0 <= score <= 1.0 for score in scores)
    assert [chunk.rank for chunk in ranked] == [1, 2, 3]


def test_reranker_promotes_relevant(reranker: CrossEncoderReranker):
    """A badly ordered candidate list still puts the 48-hour sentence first."""
    chunks = [
        RetrievedChunk(text=_PHOTO, source="bio.txt", page=0, chunk_id=0, dense_score=0.9),
        RetrievedChunk(text=_CONSENT, source="privacy.txt", page=0, chunk_id=1, dense_score=0.8),
        RetrievedChunk(text=_RETURN, source="returns.txt", page=0, chunk_id=2, dense_score=0.1),
    ]
    ranked = reranker.rerank("How fast must a return be acknowledged?", chunks, top_k=3)
    assert ranked[0].text == _RETURN
    assert ranked[0].rank == 1


def test_retrieve_end_to_end(tmp_path, embedder: EmbeddingGenerator):
    """Ingestion plus retrieval returns the consumer-rights file first."""
    raw = tmp_path / "raw"
    raw.mkdir()
    index_dir = tmp_path / "index"
    (raw / "consumer_rights.txt").write_text(_CONSUMER_RIGHTS, encoding="utf-8")
    (raw / "data_privacy.txt").write_text(_DATA_PRIVACY, encoding="utf-8")
    ingest(folder=raw, index_dir=index_dir, embedder=embedder)

    module = RetrievalModule.from_index(index_dir)
    result = module.retrieve(_processed(embedder, "How fast must a return be acknowledged?"))
    assert result.reranked is True
    assert result.chunks[0].source == "consumer_rights.txt"
    assert result.chunks[0].rank == 1
    assert len(result.chunks) <= TOP_K_FINAL


def test_retrieve_hybrid_matches_dense_top1(tmp_path, embedder: EmbeddingGenerator):
    """Hybrid search keeps the same rank-1 source and unique chunk ids."""
    raw = tmp_path / "raw"
    raw.mkdir()
    index_dir = tmp_path / "index"
    (raw / "consumer_rights.txt").write_text(_CONSUMER_RIGHTS, encoding="utf-8")
    (raw / "data_privacy.txt").write_text(_DATA_PRIVACY, encoding="utf-8")
    ingest(folder=raw, index_dir=index_dir, embedder=embedder)

    module = RetrievalModule.from_index(index_dir, use_bm25=True)
    result = module.retrieve(_processed(embedder, "How fast must a return be acknowledged?"))
    assert result.chunks[0].source == "consumer_rights.txt"
    assert len({chunk.chunk_id for chunk in result.chunks}) == len(result.chunks)


def test_retrieve_without_reranker(embedder: EmbeddingGenerator):
    """Without a reranker, order and ranks follow the dense stage."""
    store = _store_from_texts(embedder, [_RETURN, _PHOTO, _CONSENT])
    module = RetrievalModule(store, reranker=None, top_k_final=3)
    query = _processed(embedder, "How fast must a return be acknowledged?")
    dense_ids = [chunk.chunk_id for chunk in module.dense.search(query.vector, k=3)]
    result = module.retrieve(query)
    assert result.reranked is False
    assert [chunk.rank for chunk in result.chunks] == list(range(1, len(result.chunks) + 1))
    assert [chunk.chunk_id for chunk in result.chunks] == dense_ids


def test_retrieve_empty_store():
    """An empty store returns an empty result and does not raise."""
    module = RetrievalModule(VectorStore(), reranker=None)
    query = ProcessedQuery(
        raw="hello",
        cleaned="hello",
        vector=np.zeros(EMBED_DIM, dtype=np.float32),
    )
    result = module.retrieve(query)
    assert result.is_empty is True


def test_min_rerank_score_filters(embedder: EmbeddingGenerator, reranker: CrossEncoderReranker):
    """A very high score floor drops an irrelevant query without raising."""
    store = _store_from_texts(embedder, [_RETURN, _CONSENT])
    module = RetrievalModule(store, reranker=reranker, min_rerank_score=0.999)
    result = module.retrieve(
        _processed(embedder, "How many moons does the fictional planet Zog have?")
    )
    assert result.is_empty is True


def test_from_index_missing_raises(tmp_path):
    """A missing index names the ingestion command in the error."""
    with pytest.raises(FileNotFoundError, match=r"Run: python -m src.ingestion.pipeline"):
        RetrievalModule.from_index(tmp_path / "nope")


def test_retrieve_rejects_wrong_vector_dim(embedder: EmbeddingGenerator):
    """A processed query from another embedding width is rejected."""
    store = _store_from_texts(embedder, [_RETURN])
    module = RetrievalModule(store, reranker=None)
    query = ProcessedQuery(
        raw="hello",
        cleaned="hello",
        vector=np.zeros(10, dtype=np.float32),
    )
    with pytest.raises(ValueError):
        module.retrieve(query)
