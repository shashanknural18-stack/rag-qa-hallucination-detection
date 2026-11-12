"""RAGPipeline: one-call orchestrator that wires every module together."""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.config import (
    INDEX_DIR,
    NO_ANSWER_TEXT,
    NUM_SAMPLES_UI,
    PRESETS,
    RAW_DIR,
)
from src.detection.schemas import Decision, FinalResponse

logger = logging.getLogger(__name__)


class LazyNLIProxy:
    """Delays loading NLIModel until first attribute access or method call."""

    def __init__(self, loader: Callable[[], Any]) -> None:
        self._loader = loader
        self._instance: Any = None

    def _get_instance(self) -> Any:
        if self._instance is None:
            self._instance = self._loader()
        return self._instance

    @property
    def is_loaded(self) -> bool:
        """Return True if the underlying model is already loaded in memory."""
        return self._instance is not None

    def entailment_probs(self, premise: str, hypothesis: str) -> dict[str, float]:
        return self._get_instance().entailment_probs(premise, hypothesis)

    def entailment_probs_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        return self._get_instance().entailment_probs_batch(pairs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)


class LazyRerankerProxy:
    """Delays loading CrossEncoder until first call to rerank."""

    def __init__(self, loader: Callable[[], Any]) -> None:
        self._loader = loader
        self._instance: Any = None

    def _get_instance(self) -> Any:
        if self._instance is None:
            self._instance = self._loader()
        return self._instance

    @property
    def is_loaded(self) -> bool:
        """Return True if the underlying reranker model is already loaded in memory."""
        return self._instance is not None

    def rerank(self, query: str, chunks: list, top_k: int = 5, **kwargs: Any) -> list:
        return self._get_instance().rerank(query, chunks, top_k=top_k, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)


@dataclass
class PipelineOutput:
    """Everything the UI needs for one question."""

    final: FinalResponse
    chunks: list[dict]
    """Retrieved chunks: text, source file, rerank score."""
    timings: dict[str, float]
    """Seconds per stage: retrieval, generation, sampling, detection, total."""
    refused_early: bool
    """True if retrieval found nothing usable (LLM was never called)."""


class RAGPipeline:
    """End-to-end RAG with hallucination detection.

    Heavy models are loaded lazily or on first use and shared across
    all subsequent ``ask()`` calls.

    Parameters
    ----------
    embedder:
        An :class:`~src.ingestion.embedder.EmbeddingGenerator` instance.
        Pass one to share it across tests.  ``None`` creates a fresh one.
    llm:
        A :class:`~src.generation.llm.LocalLLM` instance (or fake).
        ``None`` creates a real one from the config model path.
    nli:
        An :class:`~src.detection.nli_model.NLIModel` instance (or fake).
        ``None`` creates a lazy-loading proxy.
    reranker:
        An :class:`~src.retrieval.reranker.CrossEncoderReranker` instance (or fake).
        ``None`` creates a lazy-loading proxy.
    store:
        A :class:`~src.ingestion.vector_store.VectorStore` instance.
    """

    def __init__(
        self,
        embedder: Any = None,
        llm: Any = None,
        nli: Any = None,
        reranker: Any = None,
        store: Any = None,
        top_k: int | None = None,
    ) -> None:
        from src.config import validate_config
        validate_config()

        # --- Embedder (shared: query, retrieval, citation detector) --------
        if embedder is None:
            from src.ingestion.embedder import EmbeddingGenerator
            embedder = EmbeddingGenerator()
        self._embedder = embedder

        # --- LLM (shared: generator + factuality judge) --------------------
        if llm is None:
            from src.generation.llm import LocalLLM
            llm = LocalLLM()
        self._llm = llm

        # --- NLI (shared: NLI detector + SelfCheck detector; lazy) ---------
        if nli is None:
            def _load_nli() -> Any:
                from src.detection.nli_model import NLIModel
                return NLIModel()

            self._nli = LazyNLIProxy(_load_nli)
        else:
            self._nli = nli

        # --- Module 2: query processor ------------------------------------
        from src.query.processor import QueryProcessor
        self._query_processor = QueryProcessor(self._embedder)

        # --- Module 3: retrieval ------------------------------------------
        from src.retrieval.retriever import RetrievalModule
        from src.ingestion.vector_store import VectorStore

        if store is None:
            if VectorStore.exists(INDEX_DIR):
                store = VectorStore.load(INDEX_DIR)
            else:
                store = VectorStore()
        self._store = store

        if reranker is None:
            def _load_reranker() -> Any:
                from src.retrieval.reranker import CrossEncoderReranker
                return CrossEncoderReranker()

            self._reranker = LazyRerankerProxy(_load_reranker)
        else:
            self._reranker = reranker

        retrieval_kwargs: dict[str, Any] = {}
        if top_k is not None:
            retrieval_kwargs["top_k_final"] = top_k
        self._retrieval = RetrievalModule(store, reranker=self._reranker, **retrieval_kwargs)

        # --- Module 4: generator ------------------------------------------
        from src.generation.generator import AnswerGenerator
        self._generator = AnswerGenerator(llm=self._llm)

        # --- Module 5: detection ------------------------------------------
        from src.detection.runner import DetectionRunner
        self._detector = DetectionRunner(
            nli=self._nli,
            llm=self._llm,
            embedder=self._embedder,
        )

        # LRU cache for full retrieval results: (cleaned_query, top_k, use_bm25) -> RetrievalResult
        self._retrieval_cache: OrderedDict[tuple[str, int, bool], Any] = OrderedDict()

    # ------------------------------------------------------------------
    # Cache management & Warmup
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear query and retrieval LRU caches."""
        self._retrieval_cache.clear()

    def warmup(self) -> None:
        """Execute tiny dummy operations to prime models for interactive use."""
        try:
            if self._embedder is not None and hasattr(self._embedder, "encode_batch"):
                self._embedder.encode_batch(["warmup"])
        except Exception as exc:
            logger.debug("Warmup embed skipped: %s", exc)

        try:
            if self._reranker is not None and hasattr(self._reranker, "rerank"):
                from src.schemas import RetrievedChunk
                dummy_chunk = RetrievedChunk(
                    text="warmup text",
                    source="warmup",
                    page=0,
                    chunk_id=1,
                    rank=1,
                )
                self._reranker.rerank("warmup", [dummy_chunk], top_k=1)
        except Exception as exc:
            logger.debug("Warmup rerank skipped: %s", exc)

        try:
            if self._llm is not None and hasattr(self._llm, "complete"):
                self._llm.complete("warmup", max_tokens=2)
        except Exception as exc:
            logger.debug("Warmup LLM skipped: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        use_bm25: bool = False,
        use_judge: bool | None = None,
        num_samples: int | None = None,
        mode: str = "FAST",
        stream_callback: Callable[[str], None] | None = None,
    ) -> PipelineOutput:
        """Answer *query* with retrieval, generation, and hallucination detection.

        Parameters
        ----------
        query:
            The user's natural-language question.
        top_k:
            Number of chunks to keep after reranking. If None, preset default is used.
        use_bm25:
            Enable hybrid BM25 fusion.
        use_judge:
            Run the factuality judge (LLM-as-judge). If None, preset default is used.
        num_samples:
            Stochastic samples for SelfCheckGPT. If None, preset default is used.
        mode:
            Execution preset: "FAST" or "FULL".
        stream_callback:
            Optional callback for streamed generation tokens.

        Returns
        -------
        PipelineOutput

        Raises
        ------
        ValueError
            If *query* is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        preset = PRESETS.get(mode.upper(), PRESETS["FAST"])
        actual_top_k = preset["TOP_K_FINAL"] if top_k is None else top_k
        actual_use_judge = preset["USE_JUDGE"] if use_judge is None else use_judge
        actual_num_samples = preset["NUM_SAMPLES"] if num_samples is None else num_samples
        max_chunk_tokens = preset.get("MAX_CHUNK_TOKENS")
        max_tokens = preset.get("LLM_MAX_TOKENS", 400)

        timings: dict[str, float] = {}

        # --- Retrieval with LRU cache ------------------------------------
        t0 = time.perf_counter()
        cleaned_query = self._query_processor.clean(query)
        cache_key = (cleaned_query, actual_top_k, use_bm25)

        if cache_key in self._retrieval_cache:
            self._retrieval_cache.move_to_end(cache_key)
            retrieval_result = self._retrieval_cache[cache_key]
        else:
            processed = self._query_processor.process(query)

            # Temporarily override top_k and bm25 on the retrieval module
            old_top_k = self._retrieval.top_k_final
            old_bm25 = self._retrieval.use_bm25
            self._retrieval.top_k_final = actual_top_k
            if use_bm25 != old_bm25:
                self._retrieval.use_bm25 = use_bm25
                if use_bm25 and self._retrieval.sparse is None:
                    from src.retrieval.sparse import SparseRetriever
                    self._retrieval.sparse = SparseRetriever(self._store)

            retrieval_result = self._retrieval.retrieve(processed)

            self._retrieval.top_k_final = old_top_k
            self._retrieval.use_bm25 = old_bm25

            self._retrieval_cache[cache_key] = retrieval_result
            if len(self._retrieval_cache) > 128:
                self._retrieval_cache.popitem(last=False)

        timings["retrieval"] = time.perf_counter() - t0

        # Build chunk dicts for the UI
        chunk_dicts = [
            {
                "text": c.text,
                "source": c.source,
                "rerank_score": c.rerank_score,
                "chunk_id": c.chunk_id,
                "rank": c.rank,
            }
            for c in retrieval_result.chunks
        ]

        # --- Early refusal (no usable chunks) -----------------------------
        if retrieval_result.is_empty:
            from src.generation.schemas import GeneratedAnswer
            refusal_answer = GeneratedAnswer(
                query=query,
                answer=NO_ANSWER_TEXT,
                cited_ids=[],
                invalid_citations=[],
                chunks=[],
                is_refusal=True,
            )
            final = FinalResponse(
                query=query,
                answer=NO_ANSWER_TEXT,
                decision=Decision.REFUSE,
                confidence=0.0,
                report=None,
                warning=None,
                generated=refusal_answer,
            )
            timings.update(generation=0.0, sampling=0.0, detection=0.0)
            timings["total"] = time.perf_counter() - t0
            logger.info(
                "refused early: no chunks  query_len=%d  total=%.2fs",
                len(query), timings["total"],
            )
            return PipelineOutput(
                final=final,
                chunks=chunk_dicts,
                timings=timings,
                refused_early=True,
            )

        # Precompute chunk vectors to reuse in citation detection
        chunk_vectors = None
        if retrieval_result.chunks and hasattr(self._embedder, "encode_batch"):
            try:
                chunk_vectors = np.asarray(
                    self._embedder.encode_batch([c.text for c in retrieval_result.chunks]),
                    dtype=np.float32,
                )
            except Exception:
                chunk_vectors = None

        # --- Generation (supports streaming callback) ----------------------
        t1 = time.perf_counter()
        if stream_callback is not None:
            token_list = []
            for token in self._generator.stream(
                retrieval_result,
                query,
                max_chunk_tokens=max_chunk_tokens,
                max_tokens=max_tokens,
            ):
                stream_callback(token)
                token_list.append(token)
            full_text = "".join(token_list)
            generated = self._generator.build_answer_from_text(full_text, retrieval_result, query)
        else:
            generated = self._generator.generate(
                retrieval_result,
                query,
                max_chunk_tokens=max_chunk_tokens,
                max_tokens=max_tokens,
            )
        timings["generation"] = time.perf_counter() - t1

        # --- Sampling (SelfCheck) -----------------------------------------
        if actual_num_samples > 0 and not generated.is_refusal:
            t2 = time.perf_counter()
            from src.profiling import stage

            with stage("sampling"):
                samples = self._generator.sample(retrieval_result, query, n=actual_num_samples)
                generated.samples = samples
            timings["sampling"] = time.perf_counter() - t2
        else:
            timings["sampling"] = 0.0

        # --- Detection ----------------------------------------------------
        t3 = time.perf_counter()
        final = self._detector.run(
            generated,
            skip_judge=not actual_use_judge,
            mode=mode,
            chunk_vectors=chunk_vectors,
        )
        timings["detection"] = time.perf_counter() - t3

        timings["total"] = time.perf_counter() - t0
        logger.info(
            "query_len=%d chunks=%d decision=%s confidence=%.3f "
            "retrieval=%.2fs gen=%.2fs sample=%.2fs detect=%.2fs total=%.2fs",
            len(query),
            len(retrieval_result.chunks),
            final.decision.value,
            final.confidence,
            timings["retrieval"],
            timings["generation"],
            timings["sampling"],
            timings["detection"],
            timings["total"],
        )
        return PipelineOutput(
            final=final,
            chunks=chunk_dicts,
            timings=timings,
            refused_early=False,
        )

    def ingest(self, paths: list[Path], rebuild: bool = False) -> dict:
        """Add documents to the index and reload retrieval.

        Files are copied into ``data/raw/``. If the name already exists a
        numeric suffix is added (``report.txt`` → ``report_1.txt``).

        Parameters
        ----------
        paths:
            File paths to ingest.
        rebuild:
            Delete the existing index first.

        Returns
        -------
        dict
            ``{"added": n_files, "chunks": total_chunks}``
        """
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        if rebuild:
            for name in ("faiss.index", "records.json", "manifest.json"):
                p = INDEX_DIR / name
                if p.exists():
                    p.unlink()

        copied = 0
        for src_path in paths:
            src_path = Path(src_path)
            dest = RAW_DIR / src_path.name
            if dest.exists():
                stem, suffix = dest.stem, dest.suffix
                counter = 1
                while dest.exists():
                    dest = RAW_DIR / f"{stem}_{counter}{suffix}"
                    counter += 1
            shutil.copy2(str(src_path), str(dest))
            copied += 1

        from src.ingestion.pipeline import ingest as run_ingest
        store = run_ingest(folder=RAW_DIR, index_dir=INDEX_DIR, embedder=self._embedder)

        # Clear retrieval cache and reload retrieval with the new store
        self.clear_cache()
        self._store = store
        self._retrieval = self._make_retrieval(store)

        # Free any garbage
        gc.collect()

        total_chunks = len(store)
        logger.info("ingested %d files, index now has %d chunks", copied, total_chunks)
        return {"added": copied, "chunks": total_chunks}

    def _make_retrieval(self, store: Any) -> Any:
        """Build a RetrievalModule from a store, reusing the shared reranker."""
        from src.retrieval.retriever import RetrievalModule
        return RetrievalModule(store, reranker=self._reranker)

    def index_stats(self) -> dict:
        """Return the number of files and chunks in the current index.

        Returns
        -------
        dict
            ``{"files": n, "chunks": n}``
        """
        manifest_path = INDEX_DIR / "manifest.json"
        n_files = 0
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                n_files = len(manifest)
            except Exception:
                pass
        return {"files": n_files, "chunks": len(self._store)}
