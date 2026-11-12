"""Module 2: clean a user query and embed it with the shared embedder."""

import logging
import re
import unicodedata

import numpy as np

from src.config import MAX_QUERY_CHARS
from src.ingestion.embedder import EmbeddingGenerator
from src.schemas import ProcessedQuery

logger = logging.getLogger(__name__)

_CONTRACTIONS: dict[str, str] = {
    "what's": "what is",
    "that's": "that is",
    "it's": "it is",
    "there's": "there is",
    "who's": "who is",
    "how's": "how is",
    "where's": "where is",
    "let's": "let us",
    "i'm": "i am",
    "i've": "i have",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "can't": "cannot",
    "won't": "will not",
}

_CONTRACTION_RE = re.compile(
    r"\b("
    + "|".join(re.escape(key) for key in sorted(_CONTRACTIONS, key=len, reverse=True))
    + r")\b"
)
_DISALLOWED_RE = re.compile(r"[^\w\s?!.,()\-/%&§₹:]", flags=re.UNICODE)
_PUNCT_RUN_RE = re.compile(r"([?!.])\1+")
_WHITESPACE_RE = re.compile(r"\s+")


class QueryProcessor:
    """Cleans a query and embeds the cleaned text with a shared embedder."""

    def __init__(self, embedder: EmbeddingGenerator) -> None:
        self.embedder = embedder

    @staticmethod
    def clean(query: str) -> str:
        """Normalize case, expand contractions, and keep legal punctuation."""
        if query is None or not str(query).strip():
            raise ValueError("empty query")

        text = unicodedata.normalize("NFKC", query)
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        text = text.lower()
        text = _CONTRACTION_RE.sub(lambda match: _CONTRACTIONS[match.group(0)], text)
        text = _DISALLOWED_RE.sub(" ", text)
        text = _PUNCT_RUN_RE.sub(r"\1", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()

        if len(text) > MAX_QUERY_CHARS:
            truncated = text[:MAX_QUERY_CHARS]
            cut = truncated.rfind(" ")
            text = truncated[:cut] if cut > 0 else truncated
            logger.warning(
                "Query truncated to %d characters", MAX_QUERY_CHARS
            )

        if not text or not re.search(r"\w", text):
            raise ValueError("query contains no usable text")
        return text

    def embed(self, cleaned: str) -> np.ndarray:
        """Embed a cleaned query into one L2-normalized float32 vector."""
        from src.profiling import stage

        with stage("query_embed"):
            vector = self.embedder.encode(cleaned)
        assert vector.shape == (self.embedder.dim,)
        return vector

    def process(self, query: str) -> ProcessedQuery:
        """Return the raw, cleaned, and embedded forms of a query."""
        raw = " ".join(query.split())
        cleaned = self.clean(query)
        return ProcessedQuery(raw=raw, cleaned=cleaned, vector=self.embed(cleaned))
