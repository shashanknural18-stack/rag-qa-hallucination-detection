"""RAGPipeline: one-call orchestrator that wires every module together."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import (
    INDEX_DIR,
    NO_ANSWER_TEXT,
    NUM_SAMPLES_UI,
    RAW_DIR,
)
from src.detection.schemas import Decision, FinalResponse

logger = logging.getLogger(__name__)


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

    Heavy models are loaded **once** at construction time and shared across
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
        ``None`` creates a real one.
    """

    def __init__(
        self,
        embedder: Any = None,
        llm: Any = None,
        nli: Any = None,
        reranker: Any = None,
        store: Any = None,
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

        # --- NLI (shared: NLI detector + SelfCheck detector) ---------------
        if nli is None:
            from src.detection.nli_model import NLIModel
            nli = NLIModel()
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
            from src.retrieval.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()
        self._reranker = reranker
        self._retrieval = RetrievalModule(store, reranker=self._reranker)

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        top_k: int = 5,
        use_bm25: bool = False,
        use_judge: bool = True,
        num_samples: int = NUM_SAMPLES_UI,
    ) -> PipelineOutput:
        """Answer *query* with retrieval, generation, and hallucination detection.

        Parameters
        ----------
        query:
            The user's natural-language question.
        top_k:
            Number of chunks to keep after reranking.
        use_bm25:
            Enable hybrid BM25 fusion.
        use_judge:
            Run the factuality judge (LLM-as-judge).
        num_samples:
            Stochastic samples for SelfCheckGPT (0 disables sampling).

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

        timings: dict[str, float] = {}

        # --- Retrieval ----------------------------------------------------
        t0 = time.perf_counter()
        processed = self._query_processor.process(query)

        # Temporarily override top_k and bm25 on the retrieval module
        old_top_k = self._retrieval.top_k_final
        old_bm25 = self._retrieval.use_bm25
        self._retrieval.top_k_final = top_k
        if use_bm25 != old_bm25:
            self._retrieval.use_bm25 = use_bm25
            if use_bm25 and self._retrieval.sparse is None:
                from src.retrieval.sparse import SparseRetriever
                self._retrieval.sparse = SparseRetriever(self._store)

        retrieval_result = self._retrieval.retrieve(processed)

        self._retrieval.top_k_final = old_top_k
        self._retrieval.use_bm25 = old_bm25
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

        # --- Generation ---------------------------------------------------
        t1 = time.perf_counter()
        generated = self._generator.generate(retrieval_result, query)
        timings["generation"] = time.perf_counter() - t1

        # --- Sampling (SelfCheck) -----------------------------------------
        t2 = time.perf_counter()
        if num_samples > 0 and not generated.is_refusal:
            samples = self._generator.sample(retrieval_result, query, n=num_samples)
            generated.samples = samples
        timings["sampling"] = time.perf_counter() - t2

        # --- Detection ----------------------------------------------------
        t3 = time.perf_counter()
        final = self._detector.run(generated, skip_judge=not use_judge)
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

        # Reload retrieval with the new store
        self._store = store
        self._retrieval = self._make_retrieval(store)

        total_chunks = len(store)
        logger.info("ingested %d files, index now has %d chunks", copied, total_chunks)
        return {"added": copied, "chunks": total_chunks}

    def _make_retrieval(self, store):
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
