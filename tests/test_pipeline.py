"""Unit tests for src/pipeline.py using fakes (no real models loaded)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.schemas import Decision, DetectionReport, DetectorResult, FinalResponse
from src.generation.schemas import GeneratedAnswer
from src.pipeline import PipelineOutput, RAGPipeline
from src.schemas import ProcessedQuery, RetrievalResult, RetrievedChunk


# ---------------------------------------------------------------------------
# Test Doubles
# ---------------------------------------------------------------------------
class FakeEmbedder:
    dim = 384

    def encode(self, text: str) -> np.ndarray:
        return np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)

    def encode_batch(self, texts: list[str], **kwargs) -> np.ndarray:
        n = len(texts)
        arr = np.ones((n, self.dim), dtype=np.float32)
        return arr / np.sqrt(self.dim)


class FakeLLM:
    def __init__(self, answer: str = "This is a verified answer [1].") -> None:
        self.answer = answer
        self.call_count = 0
        self.calls: list[dict] = []
        self.model_path = "models/fake-model.gguf"

    def complete(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        self.calls.append({"prompt": prompt, **kwargs})
        return self.answer


class FakeNLI:
    def entailment_probs(self, premise: str, hypothesis: str) -> dict[str, float]:
        return {"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01}

    def entailment_probs_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        return [{"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01} for _ in pairs]


class FakeReranker:
    def rerank(self, query: str, candidates: list, top_k: int = 5) -> list:
        return candidates[:top_k]


class FakeStore:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.records = []

    def __len__(self) -> int:
        return len(self.records)

    def search(self, query_vec, k=10):
        return []


def _make_chunk(text: str = "Sample text", rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source="test.txt",
        page=0,
        chunk_id=1,
        dense_score=0.9,
        rerank_score=0.85,
        rank=rank,
    )


def _make_pipeline(llm_answer="Returns take 48 hours [1]."):
    embedder = FakeEmbedder()
    llm = FakeLLM(answer=llm_answer)
    nli = FakeNLI()
    reranker = FakeReranker()
    store = FakeStore()
    pipeline = RAGPipeline(embedder=embedder, llm=llm, nli=nli, reranker=reranker, store=store)
    return pipeline, embedder, llm, nli


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAskValidation:
    def test_empty_query_raises(self):
        pipeline, _, _, _ = _make_pipeline()
        with pytest.raises(ValueError, match="must not be empty"):
            pipeline.ask("")

    def test_whitespace_query_raises(self):
        pipeline, _, _, _ = _make_pipeline()
        with pytest.raises(ValueError, match="must not be empty"):
            pipeline.ask("   \t\n  ")


class TestEarlyRefusal:
    def test_no_retrieved_chunks_refuses_early_without_llm(self):
        pipeline, _, llm, _ = _make_pipeline()
        # Mock retrieval to return empty result
        empty_result = RetrievalResult(
            query=ProcessedQuery(raw="q", cleaned="q", vector=np.zeros(384)),
            chunks=[],
            num_candidates=0,
            reranked=True,
        )
        pipeline._retrieval.retrieve = MagicMock(return_value=empty_result)

        output = pipeline.ask("What is the meaning of xyz123?")
        assert output.refused_early is True
        assert output.final.decision == Decision.REFUSE
        assert llm.call_count == 0


class TestAskHappyPath:
    def test_returns_pipeline_output_with_timings(self):
        pipeline, _, llm, _ = _make_pipeline()
        chunk = _make_chunk("Returns must be acknowledged within 48 hours.")
        ret_result = RetrievalResult(
            query=ProcessedQuery(raw="q", cleaned="q", vector=np.zeros(384)),
            chunks=[chunk],
            num_candidates=1,
            reranked=True,
        )
        pipeline._retrieval.retrieve = MagicMock(return_value=ret_result)

        output = pipeline.ask("How fast must a return be acknowledged?")
        assert isinstance(output, PipelineOutput)
        assert output.refused_early is False
        assert output.final.decision in (Decision.SHOW, Decision.WARN)

        # Check timing keys
        for key in ("retrieval", "generation", "sampling", "detection", "total"):
            assert key in output.timings
            assert output.timings[key] >= 0.0

        # Check chunks format
        assert len(output.chunks) == 1
        assert output.chunks[0]["source"] == "test.txt"
        assert output.chunks[0]["rerank_score"] == 0.85


class TestAskToggles:
    def test_use_judge_false_skips_factuality_judge(self):
        pipeline, _, _, _ = _make_pipeline()
        chunk = _make_chunk()
        ret_result = RetrievalResult(
            query=ProcessedQuery(raw="q", cleaned="q", vector=np.zeros(384)),
            chunks=[chunk],
            num_candidates=1,
            reranked=True,
        )
        pipeline._retrieval.retrieve = MagicMock(return_value=ret_result)

        output = pipeline.ask("Test query", use_judge=False, num_samples=0)
        report = output.final.report
        assert report is not None
        factuality_res = next(r for r in report.results if r.name == "factuality")
        assert factuality_res.skipped is True
        assert factuality_res.skip_reason == "disabled"

    def test_num_samples_zero_skips_sampling(self):
        pipeline, _, _, _ = _make_pipeline()
        chunk = _make_chunk()
        ret_result = RetrievalResult(
            query=ProcessedQuery(raw="q", cleaned="q", vector=np.zeros(384)),
            chunks=[chunk],
            num_candidates=1,
            reranked=True,
        )
        pipeline._retrieval.retrieve = MagicMock(return_value=ret_result)

        output = pipeline.ask("Test query", num_samples=0)
        report = output.final.report
        assert report is not None
        selfcheck_res = next(r for r in report.results if r.name == "selfcheck")
        assert selfcheck_res.skipped is True
        assert selfcheck_res.skip_reason == "no samples"


class TestIngest:
    def test_ingest_adds_suffix_if_file_exists(self, tmp_path):
        pipeline, embedder, _, _ = _make_pipeline()

        # Create dummy file to ingest
        doc1 = tmp_path / "test.txt"
        doc1.write_text("Some text for indexing.")

        # Mock RAW_DIR and INDEX_DIR to temporary paths
        fake_raw = tmp_path / "raw"
        fake_raw.mkdir()
        (fake_raw / "test.txt").write_text("Pre-existing file")

        with patch("src.pipeline.RAW_DIR", fake_raw), patch(
            "src.ingestion.pipeline.ingest"
        ) as mock_ingest:
            mock_store = MagicMock()
            mock_store.__len__ = MagicMock(return_value=5)
            mock_ingest.return_value = mock_store

            res = pipeline.ingest([doc1], rebuild=False)

            # Assert test_1.txt was created to avoid overwriting test.txt
            assert (fake_raw / "test_1.txt").exists()
            assert res["added"] == 1
            assert res["chunks"] == 5


class TestModelConstruction:
    def test_models_constructed_once(self):
        """Constructing RAGPipeline once and calling ask() three times builds models once."""
        pipeline, _, llm, _ = _make_pipeline()
        chunk = _make_chunk()
        ret_result = RetrievalResult(
            query=ProcessedQuery(raw="q", cleaned="q", vector=np.zeros(384)),
            chunks=[chunk],
            num_candidates=1,
            reranked=True,
        )
        pipeline._retrieval.retrieve = MagicMock(return_value=ret_result)

        # Call ask three times
        for _ in range(3):
            pipeline.ask("Question", num_samples=0, use_judge=False)

        # LLM complete() was called 3 times (1 for each answer generation)
        assert llm.call_count == 3
