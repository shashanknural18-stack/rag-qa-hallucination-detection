"""Document loader: reads PDF, TXT, MD, DOCX files into LangChain Documents."""

import hashlib
import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

from src.config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1_048_576)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def list_supported_files(folder: Path) -> list[Path]:
    """Return a sorted list of files with supported extensions under *folder*."""
    folder = Path(folder)
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _clean_text(text: str) -> str:
    """Collapse whitespace while preserving paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_file(path: Path) -> list[Document]:
    """Load a single file and return cleaned LangChain Document objects."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    else:
        logger.warning("Unsupported file type: %s", path.name)
        return []

    raw_docs = loader.load()
    cleaned: list[Document] = []

    for doc in raw_docs:
        text = _clean_text(doc.page_content)
        if not text:
            logger.warning("Skipping empty page in %s (scanned PDF?)", path.name)
            continue
        doc.page_content = text
        doc.metadata["source"] = path.name
        doc.metadata.setdefault("page", 0)
        cleaned.append(doc)

    if not cleaned:
        logger.warning("No extractable text from %s (scanned PDF?)", path.name)

    return cleaned
