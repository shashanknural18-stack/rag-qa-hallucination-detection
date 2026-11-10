"""Pure functions for the Streamlit UI — no Streamlit imports, fully testable."""

from __future__ import annotations

import os
import re
from typing import Any

from src.config import ALLOWED_UPLOAD_TYPES, UI_MAX_UPLOAD_MB


def decision_style(decision: str) -> dict[str, str]:
    """Return display metadata for a decision value.

    Parameters
    ----------
    decision:
        One of ``"SHOW"``, ``"WARN"``, or ``"REFUSE"``.

    Returns
    -------
    dict
        Keys ``label``, ``emoji``, and ``color``.
    """
    mapping: dict[str, dict[str, str]] = {
        "SHOW": {"label": "Verified", "emoji": "✅", "color": "green"},
        "WARN": {"label": "Warning", "emoji": "⚠️", "color": "orange"},
        "REFUSE": {"label": "Refused", "emoji": "🚫", "color": "red"},
    }
    return mapping.get(decision, mapping["REFUSE"])


def confidence_label(confidence: float) -> str:
    """Map a 0–1 confidence to a human-readable label.

    Parameters
    ----------
    confidence:
        The aggregated confidence score (0–1).

    Returns
    -------
    str
        ``"High"`` (≥ 0.75), ``"Medium"`` (≥ 0.50), or ``"Low"``.
    """
    if confidence >= 0.75:
        return "High"
    if confidence >= 0.50:
        return "Medium"
    return "Low"


def format_detector_table(report: Any) -> list[dict]:
    """Build a list of row dicts for the detector results table.

    Parameters
    ----------
    report:
        A :class:`~src.detection.schemas.DetectionReport` or ``None``.

    Returns
    -------
    list[dict]
        Each dict has keys ``name``, ``score``, and ``status``.
    """
    if report is None:
        return []
    rows: list[dict] = []
    for result in report.results:
        if result.skipped:
            status = f"skipped — {result.skip_reason}" if result.skip_reason else "skipped"
            score_str = "—"
        else:
            status = "ok"
            score_str = f"{result.score * 100:.0f}%"
        rows.append({"name": result.name, "score": score_str, "status": status})
    return rows


def format_sources(
    chunks: list[dict],
    cited_ids: list[int],
    max_preview: int = 200,
) -> list[dict]:
    """Build source-chunk rows for display.

    Parameters
    ----------
    chunks:
        Chunk dicts (from ``PipelineOutput.chunks``).
    cited_ids:
        1-based chunk ids that appear as ``[n]`` in the answer.
    max_preview:
        Truncate ``text`` to this many characters for display.

    Returns
    -------
    list[dict]
        Keys: ``id``, ``filename``, ``score``, ``text``, ``is_cited``.
    """
    rows: list[dict] = []
    for idx, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")
        preview = text[:max_preview] + "…" if len(text) > max_preview else text
        rows.append(
            {
                "id": idx,
                "filename": os.path.basename(chunk.get("source", "")),
                "score": chunk.get("rerank_score"),
                "text": preview,
                "is_cited": idx in cited_ids,
            }
        )
    return rows


def highlight_citations(answer: str) -> str:
    """Wrap ``[n]`` citation markers in bold for Streamlit markdown display.

    Parameters
    ----------
    answer:
        The answer text.

    Returns
    -------
    str
        Text with ``[n]`` replaced by ``**[n]**``.
    """
    return re.sub(r"\[(\d+)\]", r"**[\1]**", answer)


def validate_upload(filename: str, size_bytes: int) -> tuple[bool, str]:
    """Check whether a file upload should be accepted.

    Parameters
    ----------
    filename:
        Original filename including extension.
    size_bytes:
        File size in bytes.

    Returns
    -------
    tuple[bool, str]
        ``(accepted, message)``.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_UPLOAD_TYPES:
        return False, f"Unsupported file type '.{ext}'. Allowed: {', '.join(ALLOWED_UPLOAD_TYPES)}"
    max_bytes = UI_MAX_UPLOAD_MB * 1024 * 1024
    if size_bytes > max_bytes:
        return False, f"File too large ({size_bytes / (1024*1024):.1f} MB). Max: {UI_MAX_UPLOAD_MB} MB"
    return True, "ok"

