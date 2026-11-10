"""Demo: one embedder, query processing, then retrieval."""

import argparse
import logging
import time

from src.ingestion.embedder import EmbeddingGenerator
from src.query.processor import QueryProcessor
from src.retrieval.retriever import RetrievalModule

logger = logging.getLogger(__name__)


def main() -> None:
    """Process one question and print the retrieved chunks."""
    parser = argparse.ArgumentParser(description="Retrieve evidence for a question.")
    parser.add_argument(
        "question",
        nargs="?",
        default="How fast must a return be acknowledged?",
        help="Question to retrieve evidence for",
    )
    parser.add_argument("--bm25", action="store_true", help="Enable hybrid BM25 fusion")
    parser.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder")
    parser.add_argument("--k", type=int, default=5, help="Chunks to keep after retrieval")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        embedder = EmbeddingGenerator()
        processor = QueryProcessor(embedder)
        retriever = RetrievalModule.from_index(
            use_rerank=not args.no_rerank,
            use_bm25=args.bm25,
            top_k_final=args.k,
        )
        started = time.perf_counter()
        processed = processor.process(args.question)
        result = retriever.retrieve(processed)
        elapsed = time.perf_counter() - started
        logger.info("process+retrieve took %.3f s", elapsed)
    except FileNotFoundError as exc:
        print(str(exc))
        return

    print(processed.cleaned)
    if result.is_empty:
        print("No relevant chunks found.")
        return
    for chunk in result.chunks:
        print(
            f"#{chunk.rank} | score {chunk.score:.2f} | {chunk.source} "
            f"p{chunk.page_display} | {chunk.text[:120]}..."
        )


if __name__ == "__main__":
    main()
