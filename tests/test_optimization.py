"""Tests for 8 GB RAM optimizations: caching, lazy loading, presets, conditional detection, streaming, and profiling."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import THRESHOLD_SHOW, THRESHOLD_WARN
from src.detection.runner import DetectionRunner
from src.detection.schemas import Decision, DetectorResult
from src.generation.llm import LocalLLM
from src.generation.schemas import GeneratedAnswer
from src.pipeline import LazyNLIProxy, LazyRerankerProxy, PipelineOutput, RAGPipeline
from src.profiling import Profiler, stage
from src.schemas import ProcessedQuery, RetrievalResult, RetrievedChunk


# ---------------------------------------------------------------------------
# Test doubles (fakes)
# ---------------------------------------------------------------------------

class FakeEmbedder:
    dim = 384

    def encode(self, text: str) -> np.ndarray:
        return np.ones(self.dim, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [np.ones(self.dim, dtype=np.float32) for _ in texts]


class FakeLLM:
    def __init__(self, answer: str = "Returns must be acknowledged in 48 hours [1]."):
        self.answer = answer
        self.complete_calls = 0

    def complete(self, prompt: str, **kwargs) -> str:
        self.complete_calls += 1
        if "judge" in prompt.lower() or "claims" in prompt.lower():
            return '{"claims": [{"claim": "Returns must be acknowledged in 48 hours.", "verdict": "supported"}]}'
        return self.answer

    def stream(self, prompt: str, **kwargs):
        # Yield word-by-word matching complete()
        words = self.answer.split(" ")
        for i, w in enumerate(words):
            prefix = "" if i == 0 else " "
            yield prefix + w


class FakeNLI:
    def __init__(self, entailment: float = 0.95):
        self.entailment = entailment
        self.calls = 0

    def entailment_probs(self, premise: str, hypothesis: str) -> dict[str, float]:
        self.calls += 1
        return {"entailment": self.entailment, "neutral": 0.04, "contradiction": 0.01}

    def entailment_probs_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        self.calls += len(pairs)
        return [{"entailment": self.entailment, "neutral": 0.04, "contradiction": 0.01} for _ in pairs]


class FakeReranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 5, **kwargs) -> list[RetrievedChunk]:
        self.calls += 1
        for i, c in enumerate(chunks[:top_k], 1):
            c.rank = i
            c.rerank_score = 0.9
        return chunks[:top_k]


def _make_pipeline(embedder=None, llm=None, nli=None, reranker=None):
    pipe = RAGPipeline(
        embedder=embedder or FakeEmbedder(),
        llm=llm or FakeLLM(),
        nli=nli or FakeNLI(),
        reranker=reranker or FakeReranker(),
    )
    # Stub store with 1 chunk
    chunk = RetrievedChunk(
        text="Returns must be acknowledged within 48 hours.",
        source="consumer_rights.txt",
        page=0,
        chunk_id=1,
        rank=1,
        rerank_score=0.9,
    )
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = RetrievalResult(
        query=ProcessedQuery(raw="test", cleaned="test", vector=np.zeros(384, dtype=np.float32)),
        chunks=[chunk],
        num_candidates=1,
        reranked=True,
    )
    mock_retrieval.top_k_final = 5
    mock_retrieval.use_bm25 = False
    mock_retrieval.sparse = None
    pipe._retrieval = mock_retrieval
    return pipe


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cache_retrieval_same_query_called_once():
    """Cache: querying the same question twice calls the underlying retrieval only once."""
    pipe = _make_pipeline()
    query = "How fast must a return be acknowledged?"

    out1 = pipe.ask(query)
    assert not out1.refused_early
    assert pipe._retrieval.retrieve.call_count == 1

    # Second call with identical query should hit LRU cache
    out2 = pipe.ask(query)
    assert pipe._retrieval.retrieve.call_count == 1

    # Query with different top_k should miss cache and retrieve
    out3 = pipe.ask(query, top_k=2)
    assert pipe._retrieval.retrieve.call_count == 2


def test_lazy_loading_proxies():
    """Lazy loading: proxy does not call loader until first method invocation, and only once."""
    mock_loader = MagicMock(return_value=FakeNLI())
    proxy = LazyNLIProxy(mock_loader)

    assert not proxy.is_loaded
    assert mock_loader.call_count == 0

    # First invocation loads the model
    probs = proxy.entailment_probs("premise", "hypothesis")
    assert proxy.is_loaded
    assert mock_loader.call_count == 1
    assert "entailment" in probs

    # Second invocation reuses the loaded model without calling loader again
    proxy.entailment_probs("premise 2", "hypothesis 2")
    assert mock_loader.call_count == 1


def test_presets_fast_and_full():
    """Presets: FAST skips sampling and judge; FULL runs them."""
    fake_llm = FakeLLM()
    pipe = _make_pipeline(llm=fake_llm)

    # In FAST mode, num_samples is 0 (generator.sample not called) and judge is off
    out_fast = pipe.ask("How fast must a return be acknowledged?", mode="FAST")
    assert out_fast.final.report is not None
    factuality_result = next(r for r in out_fast.final.report.results if r.name == "factuality")
    assert factuality_result.skipped is True
    assert out_fast.timings["sampling"] == 0.0

    # In FULL mode, num_samples is > 0 and judge is enabled
    out_full = pipe.ask("How fast must a return be acknowledged?", mode="FULL")
    assert out_full.final.report is not None
    factuality_result_full = next(r for r in out_full.final.report.results if r.name == "factuality")
    assert factuality_result_full.skipped is False


def test_conditional_detectors():
    """Conditional detection: high and low confidence skip expensive detectors; WARN band runs them."""
    # 1. High confidence case -> skips SelfCheck and Judge in FAST mode
    high_nli = FakeNLI(entailment=0.98)
    runner = DetectionRunner(nli=high_nli, llm=FakeLLM(), embedder=FakeEmbedder())
    chunk = RetrievedChunk(text="Returns acknowledged within 48h.", source="doc.txt", page=0, chunk_id=1, rank=1)
    gen = GeneratedAnswer(
        query="test",
        answer="Returns acknowledged within 48h [1].",
        cited_ids=[1],
        invalid_citations=[],
        chunks=[chunk],
        samples=["Sample 1", "Sample 2"],
        is_refusal=False,
    )

    out_high = runner.run(gen, mode="FAST")
    sc_high = next(r for r in out_high.report.results if r.name == "selfcheck")
    judge_high = next(r for r in out_high.report.results if r.name == "factuality")
    assert sc_high.skipped is True
    assert sc_high.skip_reason == "conditional_skip"
    assert judge_high.skipped is True

    # 2. Low confidence case -> skips SelfCheck and Judge in FAST mode
    low_nli = FakeNLI(entailment=0.1)
    runner_low = DetectionRunner(nli=low_nli, llm=FakeLLM(), embedder=FakeEmbedder())
    out_low = runner_low.run(gen, mode="FAST")
    sc_low = next(r for r in out_low.report.results if r.name == "selfcheck")
    judge_low = next(r for r in out_low.report.results if r.name == "factuality")
    assert sc_low.skipped is True
    assert judge_low.skipped is True

    # 3. WARN band case -> runs SelfCheck and Judge in FAST mode
    class WarnNLI:
        def entailment_probs_batch(self, pairs):
            results = []
            for premise, hyp in pairs:
                if "48h" in hyp:
                    results.append({"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05})
                else:
                    results.append({"entailment": 0.1, "neutral": 0.85, "contradiction": 0.05})
            return results

    gen_warn = GeneratedAnswer(
        query="test",
        answer="Returns acknowledged within 48h [1]. Extra unverified sentence without citation.",
        cited_ids=[1],
        invalid_citations=[],
        chunks=[chunk],
        samples=["Sample 1", "Sample 2"],
        is_refusal=False,
    )
    runner_warn = DetectionRunner(nli=WarnNLI(), llm=FakeLLM(), embedder=FakeEmbedder())
    out_warn = runner_warn.run(gen_warn, mode="FAST")
    sc_warn = next(r for r in out_warn.report.results if r.name == "selfcheck")
    judge_warn = next(r for r in out_warn.report.results if r.name == "factuality")
    assert sc_warn.skipped is False
    assert judge_warn.skipped is False


def test_streaming_equals_complete():
    """stream() yields tokens in order and joined text equals complete()."""
    # Test on LocalLLM in mock mode
    llm = LocalLLM()
    llm._is_mock = True

    prompt = "Tell me about return policy."
    tokens = list(llm.stream(prompt))
    assert len(tokens) > 1

    complete_text = llm.complete(prompt)
    joined = "".join(tokens)
    assert joined == complete_text


def test_profiler_context_manager_records_stage():
    """Profiler stage context manager records stage name, duration, and peak RAM."""
    profiler = Profiler()
    with profiler.stage("test_stage", test_meta="val"):
        time.sleep(0.01)

    assert len(profiler.records) == 1
    rec = profiler.records[0]
    assert rec.name == "test_stage"
    assert rec.duration >= 0.008
    assert rec.peak_ram_mb > 0
    assert rec.extra.get("test_meta") == "val"

