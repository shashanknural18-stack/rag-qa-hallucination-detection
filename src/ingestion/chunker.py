"""Text chunker: splits Documents into token-sized chunks."""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from src.config import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


@lru_cache(maxsize=1)
def get_tokenizer() -> AutoTokenizer:
    """Lazily load and cache the tokenizer for the embedding model."""
    return AutoTokenizer.from_pretrained(EMBED_MODEL)


def build_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Return a token-aware recursive text splitter."""
    tokenizer = get_tokenizer()
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(docs: list[Document], start_id: int = 0) -> list[Document]:
    """Split documents into chunks with sequential chunk_id metadata."""
    splitter = build_splitter()
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = start_id + i
    return chunks
