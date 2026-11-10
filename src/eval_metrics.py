"""Pure metric calculation functions and error injection for evaluation."""

from __future__ import annotations

import re
from typing import Any

_DECISIONS = ("SHOW", "WARN", "REFUSE")


def compute_confusion_matrix(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Compute an expected vs actual confusion matrix for decisions.

    Parameters
    ----------
    records:
        List of dicts containing ``expected_decision`` and ``actual_decision``.

    Returns
    -------
    dict[str, dict[str, int]]
        Mapping: expected -> {actual: count}.
    """
    matrix: dict[str, dict[str, int]] = {
        exp: {act: 0 for act in _DECISIONS} for exp in _DECISIONS
    }
    for rec in records:
        exp = rec.get("expected_decision", "")
        act = rec.get("actual_decision", "")
        if exp in matrix and act in matrix[exp]:
            matrix[exp][act] += 1
    return matrix


def answer_rate_answerable(records: list[dict[str, Any]]) -> float:
    """Fraction of answerable questions that resulted in SHOW or WARN."""
    ans_records = [r for r in records if r.get("category") == "answerable"]
    if not ans_records:
        return 0.0
    accepted = sum(1 for r in ans_records if r.get("actual_decision") in ("SHOW", "WARN"))
    return accepted / len(ans_records)


def refusal_rate_unanswerable(records: list[dict[str, Any]]) -> float:
    """Fraction of unanswerable questions that resulted in REFUSE."""
    unans_records = [r for r in records if r.get("category") == "unanswerable"]
    if not unans_records:
        return 0.0
    refused = sum(1 for r in unans_records if r.get("actual_decision") == "REFUSE")
    return refused / len(unans_records)


def false_accept_rate(records: list[dict[str, Any]]) -> float:
    """Fraction of unanswerable questions that incorrectly received SHOW."""
    unans_records = [r for r in records if r.get("category") == "unanswerable"]
    if not unans_records:
        return 0.0
    false_accepts = sum(1 for r in unans_records if r.get("actual_decision") == "SHOW")
    return false_accepts / len(unans_records)


def false_refuse_rate(records: list[dict[str, Any]]) -> float:
    """Fraction of answerable questions that incorrectly received REFUSE."""
    ans_records = [r for r in records if r.get("category") == "answerable"]
    if not ans_records:
        return 0.0
    false_refuses = sum(1 for r in ans_records if r.get("actual_decision") == "REFUSE")
    return false_refuses / len(ans_records)


def mean_confidence_by_category(records: list[dict[str, Any]]) -> dict[str, float]:
    """Compute mean confidence score grouped by question category."""
    totals: dict[str, list[float]] = {}
    for r in records:
        cat = r.get("category", "unknown")
        conf = r.get("confidence", 0.0)
        totals.setdefault(cat, []).append(float(conf))
    return {cat: sum(vals) / len(vals) for cat, vals in totals.items()}


def mean_detector_scores_by_category(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute average detector scores per category."""
    groups: dict[str, dict[str, list[float]]] = {}
    for r in records:
        cat = r.get("category", "unknown")
        d_scores = r.get("detector_scores", {})
        if cat not in groups:
            groups[cat] = {}
        for dname, score in d_scores.items():
            if score is not None:
                groups[cat].setdefault(dname, []).append(float(score))

    summary: dict[str, dict[str, float]] = {}
    for cat, d_map in groups.items():
        summary[cat] = {
            dname: (sum(scores) / len(scores) if scores else 0.0)
            for dname, scores in d_map.items()
        }
    return summary


def mean_latency_by_stage(records: list[dict[str, Any]]) -> dict[str, float]:
    """Compute average execution latency in seconds per pipeline stage."""
    totals: dict[str, list[float]] = {}
    for r in records:
        timings = r.get("timings", {})
        for stage, secs in timings.items():
            totals.setdefault(stage, []).append(float(secs))
    return {stage: sum(vals) / len(vals) for stage, vals in totals.items()}


_NUMBER_SUBS = [
    (r"\b48\s*(?:hours|hrs)\b", "96 hours"),
    (r"\b30\s*(?:days)\b", "60 days"),
    (r"\b48\b", "99"),
    (r"\b30\b", "75"),
    (r"\bfree\b", "mandatory"),
    (r"\bunconditional\b", "conditional"),
    (r"\breplacement\b", "confiscation"),
    (r"\brefund\b", "forfeiture"),
]


def inject_error(text: str) -> str:
    """Alter exactly one factual number, duration, or keyword in the text.

    If no known rule matches, changes the first digit found in the text.
    """
    for pattern, replacement in _NUMBER_SUBS:
        new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        if count > 0:
            return new_text

    # Fallback: modify the first standalone digit
    digit_match = re.search(r"\b\d+\b", text)
    if digit_match:
        val = int(digit_match.group(0))
        return text[: digit_match.start()] + str(val + 99) + text[digit_match.end() :]

    return text + " [Factually contradicted detail: not applicable]."
