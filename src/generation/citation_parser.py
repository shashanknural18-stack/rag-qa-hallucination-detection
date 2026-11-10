"""Extract ``[n]`` citation markers from generated text."""

from __future__ import annotations

import re

# Matches [1], [12], etc.  Also handles [1, 2] and [1][2] patterns.
_BRACKET_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_citations(
    answer: str,
    num_chunks: int,
) -> tuple[list[int], list[int]]:
    """Return ``(valid_ids, invalid_ids)`` found in *answer*.

    Handles single markers like ``[1]``, adjacent markers like ``[1][2]``,
    and comma-separated markers like ``[1, 2]``.

    Both returned lists are **deduplicated** and preserve first-occurrence order.

    Parameters
    ----------
    answer:
        The generated answer text to scan.
    num_chunks:
        How many chunks were provided (valid ids are ``1..num_chunks``).

    Returns
    -------
    tuple[list[int], list[int]]
        ``(valid_ids, invalid_ids)``
    """
    seen_valid: dict[int, None] = {}
    seen_invalid: dict[int, None] = {}

    for match in _BRACKET_RE.finditer(answer):
        ids_str = match.group(1)
        for part in ids_str.split(","):
            part = part.strip()
            if not part:
                continue
            cid = int(part)
            if 1 <= cid <= num_chunks:
                seen_valid.setdefault(cid, None)
            else:
                seen_invalid.setdefault(cid, None)

    return list(seen_valid), list(seen_invalid)

