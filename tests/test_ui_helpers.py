"""Unit tests for src/ui_helpers.py (pure functions, no Streamlit needed)."""

from __future__ import annotations

import pytest

from src.detection.schemas import DetectionReport, DetectorResult
from src.ui_helpers import (
    confidence_label,
    decision_style,
    format_detector_table,
    format_sources,
    highlight_citations,
    validate_upload,
)


class TestDecisionStyle:
    def test_show_style(self):
        style = decision_style("SHOW")
        assert style["label"] == "Verified"
        assert style["emoji"] == "✅"
        assert style["color"] == "green"

    def test_warn_style(self):
        style = decision_style("WARN")
        assert style["label"] == "Warning"
        assert style["emoji"] == "⚠️"
        assert style["color"] == "orange"

    def test_refuse_style(self):
        style = decision_style("REFUSE")
        assert style["label"] == "Refused"
        assert style["emoji"] == "🚫"
        assert style["color"] == "red"

    def test_unknown_defaults_to_refuse(self):
        style = decision_style("UNKNOWN")
        assert style["label"] == "Refused"


class TestConfidenceLabel:
    def test_high(self):
        assert confidence_label(0.75) == "High"
        assert confidence_label(0.99) == "High"
        assert confidence_label(1.0) == "High"

    def test_medium(self):
        assert confidence_label(0.50) == "Medium"
        assert confidence_label(0.749) == "Medium"

    def test_low(self):
        assert confidence_label(0.499) == "Low"
        assert confidence_label(0.0) == "Low"


class TestFormatDetectorTable:
    def test_none_report(self):
        assert format_detector_table(None) == []

    def test_active_and_skipped(self):
        report = DetectionReport(
            results=[
                DetectorResult(
                    name="nli",
                    score=0.92,
                    sentence_scores=[0.92],
                ),
                DetectorResult(
                    name="factuality",
                    score=0.0,
                    sentence_scores=[],
                    skipped=True,
                    skip_reason="disabled",
                ),
            ],
            sentences=["A sentence."],
        )
        rows = format_detector_table(report)
        assert len(rows) == 2
        assert rows[0] == {"name": "nli", "score": "92%", "status": "ok"}
        assert rows[1] == {
            "name": "factuality",
            "score": "—",
            "status": "skipped — disabled",
        }


class TestFormatSources:
    def test_marks_cited_chunks(self):
        chunks = [
            {"text": "First chunk text", "source": "path/to/doc_a.txt", "rerank_score": 0.85},
            {"text": "Second chunk text", "source": "doc_b.pdf", "rerank_score": 0.42},
        ]
        rows = format_sources(chunks, cited_ids=[1])
        assert len(rows) == 2
        assert rows[0]["id"] == 1
        assert rows[0]["filename"] == "doc_a.txt"
        assert rows[0]["is_cited"] is True
        assert rows[1]["id"] == 2
        assert rows[1]["filename"] == "doc_b.pdf"
        assert rows[1]["is_cited"] is False

    def test_truncates_long_text(self):
        long_text = "x" * 300
        chunks = [{"text": long_text, "source": "test.txt", "rerank_score": 0.5}]
        rows = format_sources(chunks, cited_ids=[], max_preview=100)
        assert len(rows[0]["text"]) == 101  # 100 + ellipsis
        assert rows[0]["text"].endswith("…")


class TestHighlightCitations:
    def test_wraps_markers(self):
        result = highlight_citations("According to [1] and [2], returns take 48 hours.")
        assert result == "According to **[1]** and **[2]**, returns take 48 hours."

    def test_no_markers_unchanged(self):
        text = "No citations here."
        assert highlight_citations(text) == text


class TestValidateUpload:
    def test_accepts_valid_types(self):
        for ext in ("pdf", "txt", "md", "docx"):
            ok, msg = validate_upload(f"test.{ext}", 1024)
            assert ok is True
            assert msg == "ok"

    def test_rejects_disallowed_types(self):
        ok, msg = validate_upload("malware.exe", 1024)
        assert ok is False
        assert "Unsupported file type" in msg

    def test_rejects_oversized_files(self):
        twenty_one_mb = 21 * 1024 * 1024
        ok, msg = validate_upload("big.pdf", twenty_one_mb)
        assert ok is False
        assert "File too large" in msg

