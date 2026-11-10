"""Ingestion pipeline: loads, chunks, embeds, and indexes documents."""

import argparse
import json
import logging
from pathlib import Path

from src.config import RAW_DIR, INDEX_DIR
from src.ingestion.loader import file_hash, list_supported_files, load_file
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest(
    folder: Path = RAW_DIR,
    index_dir: Path = INDEX_DIR,
    embedder: EmbeddingGenerator | None = None,
) -> VectorStore:
    """Run the full ingestion pipeline on *folder* and return the vector store."""
    folder = Path(folder)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Load or create the store
    if VectorStore.exists(index_dir):
        store = VectorStore.load(index_dir)
        logger.info("Loaded existing index with %d vectors", len(store))
    else:
        store = VectorStore()
        logger.info("Starting fresh index")

    # Load manifest
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists():
        manifest: dict[str, str] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    else:
        manifest = {}

    # List files
    files = list_supported_files(folder)
    if not files:
        logger.warning("No supported files found in %s", folder)

    # Lazy-init embedder
    if embedder is None:
        embedder = EmbeddingGenerator()

    new_pages = 0
    new_chunks = 0

    for filepath in files:
        try:
            fhash = file_hash(filepath)
            if fhash in manifest:
                logger.info("Skip (unchanged): %s", filepath.name)
                continue

            docs = load_file(filepath)
            if not docs:
                logger.warning("No documents extracted from %s", filepath.name)
                manifest[fhash] = filepath.name
                continue

            new_pages += len(docs)
            chunks = chunk_documents(docs, start_id=len(store))
            if chunks:
                vectors = embedder.encode_batch(
                    [c.page_content for c in chunks]
                )
                store.add(vectors, chunks)
                new_chunks += len(chunks)

            manifest[fhash] = filepath.name
            logger.info(
                "Indexed %s: %d pages -> %d chunks",
                filepath.name,
                len(docs),
                len(chunks),
            )
        except Exception:
            logger.exception("Error processing %s", filepath.name)

    # Save
    store.save(index_dir)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"{new_pages} new pages -> {new_chunks} new chunks "
        f"| total in index: {len(store)}"
    )
    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into the FAISS vector store."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=RAW_DIR,
        help="Folder containing raw documents (default: data/raw)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing index and rebuild from scratch",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.rebuild:
        for name in ("faiss.index", "records.json", "manifest.json"):
            p = INDEX_DIR / name
            if p.exists():
                p.unlink()
                logger.info("Deleted %s", p)

    ingest(folder=args.folder)
