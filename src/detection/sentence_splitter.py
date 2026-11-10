"""Split an answer into sentences and recover the citations on each one."""

from __future__ import annotations

import re

# [1], [1][2], and [1, 2]
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_LEADING_CITATIONS_RE = re.compile(
    r"^((?:\[\d+(?:\s*,\s*\d+)*\]\s*)+)"
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_MIN_WORDS = 3


def _citation_ids(text: str) -> list[int]:
    """Return citation ids in *text*, deduplicated, first-seen order preserved."""
    seen: dict[int, None] = {}
    for match in _CITATION_RE.finditer(text):
        for part in match.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            seen.setdefault(int(part), None)
    return list(seen)


def _strip_citations(text: str) -> str:
    """Remove citation markers and tidy the whitespace they leave behind."""
    text = _CITATION_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.!?])", r"\1", text)
    return text.strip()


def _raw_segments(answer: str) -> list[str]:
    """Split on sentence boundaries and keep a trailing ``[n]`` with its sentence.

    A marker that sits after the period (``"... hours. [1] Next"``) is attached
    to the sentence it follows, which is the citation style the generator asks for.
    """
    text = re.sub(r"\s+", " ", answer).strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        leading = _LEADING_CITATIONS_RE.match(part)
        if leading and merged:
            merged[-1] = f"{merged[-1]} {leading.group(1).strip()}".strip()
            rest = part[leading.end() :].strip()
            if rest:
                merged.append(rest)
        else:
            merged.append(part)
    return merged


def _kept_segments(answer: str) -> list[tuple[str, list[int]]]:
    """Return ``(cleaned sentence, citation ids)`` for sentences of at least 3 words."""
    kept: list[tuple[str, list[int]]] = []
    for segment in _raw_segments(answer):
        citation_ids = _citation_ids(segment)
        cleaned = _strip_citations(segment)
        if len(_WORD_RE.findall(cleaned)) < _MIN_WORDS:
            continue
        kept.append((cleaned, citation_ids))
    return kept


def split_sentences(answer: str) -> list[str]:
    """Split *answer* into sentences with citation markers removed.

    Splitting is regex-based and does not download tokenizer data. Fragments
    shorter than three words are dropped.

    Parameters
    ----------
    answer:
        Generated answer, possibly containing ``[n]`` citation markers.

    Returns
    -------
    list[str]
        Cleaned sentences, in order.
    """
    return [text for text, _ids in _kept_segments(answer)]


def sentence_citations(answer: str) -> list[list[int]]:
    """Return the citation ids attached to each sentence from :func:`split_sentences`.

    Ids keep first-seen order, duplicates are removed, and ids that do not map
    to a chunk are kept so the citation detector can flag them. The outer list
    lines up with :func:`split_sentences`.

    Parameters
    ----------
    answer:
        Generated answer, possibly containing ``[n]`` citation markers.

    Returns
    -------
    list[list[int]]
        Citation ids per kept sentence.
    """
    return [ids for _text, ids in _kept_segments(answer)]
