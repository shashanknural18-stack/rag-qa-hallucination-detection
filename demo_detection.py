"""Demo: query -> retrieval -> generation -> hallucination detection."""

import argparse
import logging
import time

from src.config import NUM_SAMPLES
from src.ingestion.embedder import EmbeddingGenerator
from src.query.processor import QueryProcessor
from src.retrieval.retriever import RetrievalModule

logger = logging.getLogger(__name__)


def main() -> None:
    """Answer one question and print the show / warn / refuse decision."""
    parser = argparse.ArgumentParser(
        description="Answer a question and score it for hallucinations."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="How fast must a return be acknowledged?",
        help="Question to answer",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=NUM_SAMPLES,
        help="Stochastic samples for SelfCheck (0 skips that detector)",
    )
    parser.add_argument("--bm25", action="store_true", help="Enable hybrid BM25 fusion")
    parser.add_argument("--k", type=int, default=5, help="Chunks to keep after retrieval")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the factuality judge for a faster run",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        embedder = EmbeddingGenerator()
        processor = QueryProcessor(embedder)
        retriever = RetrievalModule.from_index(use_bm25=args.bm25, top_k_final=args.k)

        from src.generation.generator import AnswerGenerator
        from src.generation.llm import LocalLLM
        from src.detection.nli_model import NLIModel
        from src.detection.runner import DetectionRunner

        llm = LocalLLM()
        nli = NLIModel()
        generator = AnswerGenerator(llm=llm)
        runner = DetectionRunner(nli=nli, llm=llm, embedder=embedder)
    except (FileNotFoundError, OSError) as exc:
        print(str(exc))
        return

    started = time.perf_counter()
    processed = processor.process(args.question)
    retrieval = retriever.retrieve(processed)
    generated = generator.generate(retrieval, processed.raw)
    if args.samples > 0 and not generated.is_refusal and generated.chunks:
        generated.samples = generator.sample(retrieval, processed.raw, n=args.samples)
    response = runner.run(generated, skip_judge=args.no_judge)
    elapsed = time.perf_counter() - started
    logger.info("full pipeline took %.3f s", elapsed)

    print("\n" + "=" * 70)
    print(f"QUERY: {response.query}")
    print("=" * 70)
    print(f"\nDECISION: {response.decision.value}    confidence {response.confidence:.3f}")
    if response.warning:
        print(f"WARNING: {response.warning}")
    print(f"\nANSWER:\n{response.answer}\n")

    report = response.report
    if report is None:
        print("Detection skipped (the model refused or retrieval returned nothing).")
    else:
        print(f"{'detector':<14} {'score':>8}  status")
        print("-" * 70)
        for result in report.results:
            if result.skipped:
                print(f"{result.name:<14} {'—':>8}  skipped: {result.skip_reason}")
            else:
                print(f"{result.name:<14} {result.score:8.3f}")
        if report.sentences:
            print(f"\n{'─' * 70}")
            print("PER-SENTENCE SCORES:")
            for index, sentence in enumerate(report.sentences):
                parts: list[str] = []
                for result in report.results:
                    if result.skipped or index >= len(result.sentence_scores):
                        parts.append(f"{result.name}=skip")
                    else:
                        parts.append(f"{result.name}={result.sentence_scores[index]:.2f}")
                print(f"\n[{index + 1}] {sentence}")
                print("    " + "  ".join(parts))

    cited = response.generated
    if cited.cited_ids:
        print(f"\n{'─' * 70}")
        print("CITED SOURCES:")
        for cid in cited.cited_ids:
            if cid < 1 or cid > len(cited.chunks):
                continue
            chunk = cited.chunks[cid - 1]
            print(f"  [{cid}] {chunk.source} (p{chunk.page_display}): {chunk.text[:120]}...")
    if cited.invalid_citations:
        print(f"\nInvalid citations: {cited.invalid_citations}")
    print(f"\nGeneration refusal: {cited.is_refusal}")


if __name__ == "__main__":
    main()
