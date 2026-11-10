"""Unit tests for evaluation metrics and error injection (no real models needed)."""

from __future__ import annotations

import pytest

from src.eval_metrics import (
    answer_rate_answerable,
    compute_confusion_matrix,
    false_accept_rate,
    false_refuse_rate,
    inject_error,
    mean_confidence_by_category,
    mean_detector_scores_by_category,
    mean_latency_by_stage,
    refusal_rate_unanswerable,
)


@pytest.fixture
def sample_records():
    return [
        {
            "id": "1",
            "category": "answerable",
            "expected_decision": "SHOW",
            "actual_decision": "SHOW",
            "confidence": 0.90,
            "detector_scores": {"nli": 0.95, "citation": 1.0},
            "timings": {"retrieval": 0.1, "total": 0.5},
        },
        {
            "id": "2",
            "category": "answerable",
            "expected_decision": "SHOW",
            "actual_decision": "WARN",
            "confidence": 0.65,
            "detector_scores": {"nli": 0.70, "citation": 0.5},
            "timings": {"retrieval": 0.2, "total": 0.6},
        },
        {
            "id": "3",
            "category": "answerable",
            "expected_decision": "SHOW",
            "actual_decision": "REFUSE",
            "confidence": 0.20,
            "detector_scores": {"nli": 0.10, "citation": 0.0},
            "timings": {"retrieval": 0.1, "total": 0.2},
        },
        {
            "id": "4",
            "category": "unanswerable",
            "expected_decision": "REFUSE",
            "actual_decision": "REFUSE",
            "confidence": 0.0,
            "detector_scores": {},
            "timings": {"retrieval": 0.05, "total": 0.05},
        },
        {
            "id": "5",
            "category": "unanswerable",
            "expected_decision": "REFUSE",
            "actual_decision": "SHOW",
            "confidence": 0.85,
            "detector_scores": {"nli": 0.8},
            "timings": {"retrieval": 0.1, "total": 0.4},
        },
    ]


class TestMetrics:
    def test_confusion_matrix(self, sample_records):
        matrix = compute_confusion_matrix(sample_records)
        assert matrix["SHOW"]["SHOW"] == 1
        assert matrix["SHOW"]["WARN"] == 1
        assert matrix["SHOW"]["REFUSE"] == 1
        assert matrix["REFUSE"]["REFUSE"] == 1
        assert matrix["REFUSE"]["SHOW"] == 1

    def test_answer_rate_answerable(self, sample_records):
        # 2 out of 3 answerable got SHOW or WARN -> 2/3 ≈ 0.6667
        rate = answer_rate_answerable(sample_records)
        assert pytest.approx(rate, rel=1e-2) == 2 / 3

    def test_refusal_rate_unanswerable(self, sample_records):
        # 1 out of 2 unanswerable got REFUSE -> 0.50
        rate = refusal_rate_unanswerable(sample_records)
        assert rate == 0.5

    def test_false_accept_rate(self, sample_records):
        # 1 out of 2 unanswerable got SHOW -> 0.50
        rate = false_accept_rate(sample_records)
        assert rate == 0.5

    def test_false_refuse_rate(self, sample_records):
        # 1 out of 3 answerable got REFUSE -> 1/3 ≈ 0.3333
        rate = false_refuse_rate(sample_records)
        assert pytest.approx(rate, rel=1e-2) == 1 / 3

    def test_mean_confidence_by_category(self, sample_records):
        confs = mean_confidence_by_category(sample_records)
        assert pytest.approx(confs["answerable"], rel=1e-2) == (0.90 + 0.65 + 0.20) / 3
        assert pytest.approx(confs["unanswerable"], rel=1e-2) == (0.0 + 0.85) / 2

    def test_mean_detector_scores(self, sample_records):
        scores = mean_detector_scores_by_category(sample_records)
        assert "nli" in scores["answerable"]
        assert pytest.approx(scores["answerable"]["nli"], rel=1e-2) == (0.95 + 0.70 + 0.10) / 3

    def test_mean_latency_by_stage(self, sample_records):
        latencies = mean_latency_by_stage(sample_records)
        assert "retrieval" in latencies
        assert "total" in latencies
        assert latencies["total"] > 0


class TestErrorInjection:
    def test_inject_error_hours(self):
        text = "Returns must be acknowledged within 48 hours."
        injected = inject_error(text)
        assert "96 hours" in injected
        assert "48 hours" not in injected
        # Assert rest of text remains unchanged
        assert injected.startswith("Returns must be acknowledged within")

    def test_inject_error_days(self):
        text = "Resolve requests within 30 days."
        injected = inject_error(text)
        assert "60 days" in injected
        assert "30 days" not in injected

    def test_inject_error_fallback_number(self):
        text = "Article 12 guarantees this."
        injected = inject_error(text)
        assert "111" in injected
        assert "guarantees this" in injected
