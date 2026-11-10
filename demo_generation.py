"""Demo: query -> retrieval -> generation with Module 4."""

import argparse
import logging
import time

from src.ingestion.embedder import EmbeddingGenerator
from src.query.processor import QueryProcessor
from src.retrieval.retriever import RetrievalModule

logger = logging.getLogger(__name__)


def main() -> None:
    """Process one question, retrieve evidence, and generate an answer."""
    parser = argparse.ArgumentParser(
        description="Generate a grounded answer for a question."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="How fast must a return be acknowledged?",
        help="Question to answer",
    )
    parser.add_argument("--samples", type=int, default=0, help="Number of stochastic samples for SelfCheckGPT")
    parser.add_argument("--bm25", action="store_true", help="Enable hybrid BM25 fusion")
    parser.add_argument("--k", type=int, default=5, help="Chunks to keep after retrieval")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        # Shared embedder across query processing and retrieval
        embedder = EmbeddingGenerator()
        processor = QueryProcessor(embedder)
        retriever = RetrievalModule.from_index(use_bm25=args.bm25, top_k_final=args.k)

        # Load the LLM (one-time cost)
        from src.generation.llm import LocalLLM
        llm = LocalLLM()

        from src.generation.generator import AnswerGenerator
        generator = AnswerGenerator(llm=llm)
    except FileNotFoundError as exc:
        print(str(exc))
        return

    # Pipeline
    started = time.perf_counter()
    processed = processor.process(args.question)
    result = retriever.retrieve(processed)
    answer = generator.generate(result, processed.raw)
    elapsed = time.perf_counter() - started

    logger.info("full pipeline took %.3f s", elapsed)

    # Display
    print("\n" + "=" * 70)
    print(f"QUERY: {answer.query}")
    print("=" * 70)
    print(f"\nANSWER:\n{answer.answer}\n")

    if answer.cited_ids:
        print("CITED CHUNKS:")
        for cid in answer.cited_ids:
            chunk = answer.chunks[cid - 1]
            print(f"  [{cid}] {chunk.source} (p{chunk.page_display}): {chunk.text[:100]}...")
    if answer.invalid_citations:
        print(f"\n⚠ INVALID CITATIONS: {answer.invalid_citations}")
    print(f"\nRefusal: {answer.is_refusal}")
    print(f"Model: {answer.model_name}")
    print(f"Temperature: {answer.temperature}")

    # Optional stochastic samples
    if args.samples > 0:
        print(f"\n{'─' * 70}")
        print(f"STOCHASTIC SAMPLES ({args.samples}):")
        samples = generator.sample(result, processed.raw, n=args.samples)
        answer.samples = samples
        for i, s in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            print(s)


if __name__ == "__main__":
    main()

