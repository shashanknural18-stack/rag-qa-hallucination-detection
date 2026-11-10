"""Quick sanity check: search the index for a query and print top results."""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import INDEX_DIR
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.vector_store import VectorStore


def main() -> None:
    """Run a quick search against the stored index."""
    parser = argparse.ArgumentParser(description="Quick index search")
    parser.add_argument(
        "question",
        nargs="?",
        default="What counts as a deficiency in goods?",
        help="Query string to search",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=3,
        help="Number of results (default: 3)",
    )
    args = parser.parse_args()

    if not VectorStore.exists(INDEX_DIR):
        print(
            "Index not found. Run the ingestion pipeline first:\n"
            "  python -m src.ingestion.pipeline"
        )
        sys.exit(1)

    store = VectorStore.load(INDEX_DIR)
    embedder = EmbeddingGenerator()
    query_vec = embedder.encode(args.question)

    results = store.search(query_vec, k=args.k)

    print(f"\nQuery: {args.question}\n")
    for record, score in results:
        source = record["metadata"].get("source", "?")
        page = record["metadata"].get("page", 0)
        text = record["text"][:120]
        print(f"  {score:.2f} | {source} p{page} | {text}...")


if __name__ == "__main__":
    main()
