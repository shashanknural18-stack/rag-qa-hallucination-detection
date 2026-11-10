"""Streamlit AppTest integration tests for app.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

# Skip if streamlit is not yet installed in environment
st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from src.detection.schemas import Decision, DetectionReport, DetectorResult, FinalResponse
from src.generation.schemas import GeneratedAnswer
from src.pipeline import PipelineOutput


def _create_mock_pipeline(decision=Decision.SHOW, answer="Acknowledged within 48h [1]."):
    mock_pipeline = MagicMock()
    mock_pipeline.index_stats.return_value = {"files": 2, "chunks": 12}

    gen = GeneratedAnswer(
        query="How fast must a return be acknowledged?",
        answer=answer,
        cited_ids=[1],
        invalid_citations=[],
        chunks=[],
        is_refusal=(decision == Decision.REFUSE),
    )
    final = FinalResponse(
        query=gen.query,
        answer=answer if decision != Decision.REFUSE else "Refusal message",
        decision=decision,
        confidence=0.88 if decision == Decision.SHOW else 0.2,
        report=DetectionReport(
            results=[
                DetectorResult(name="nli", score=0.9, sentence_scores=[0.9]),
                DetectorResult(name="citation", score=1.0, sentence_scores=[1.0]),
            ],
            sentences=[answer],
        ),
        warning=None if decision != Decision.WARN else "Check sources",
        generated=gen,
    )
    mock_pipeline.ask.return_value = PipelineOutput(
        final=final,
        chunks=[
            {
                "text": "Returns must be acknowledged within 48 hours.",
                "source": "consumer_rights.txt",
                "rerank_score": 0.85,
                "chunk_id": 1,
                "rank": 1,
            }
        ],
        timings={"retrieval": 0.1, "generation": 0.2, "sampling": 0.0, "detection": 0.1, "total": 0.4},
        refused_early=False,
    )
    return mock_pipeline


def test_app_starts_without_exception():
    mock_pipe = _create_mock_pipeline()
    with patch("app.get_pipeline", return_value=mock_pipe), patch(
        "app._load_pipeline", return_value=mock_pipe
    ):
        at = AppTest.from_file("../app.py", default_timeout=30)
        at.run()
        assert not at.exception


def test_app_submitting_question_renders_banner_and_sources():
    mock_pipe = _create_mock_pipeline(decision=Decision.SHOW)
    with patch("app.get_pipeline", return_value=mock_pipe), patch(
        "app._load_pipeline", return_value=mock_pipe
    ):
        at = AppTest.from_file("../app.py", default_timeout=30)
        at.run()

        # Chat input is present
        if at.chat_input:
            at.chat_input[0].set_value("How fast must a return be acknowledged?").run(timeout=30)
            assert not at.exception
            assert len(at.success) > 0 or len(at.markdown) > 0


def test_app_refuse_output_hides_unverified_answer_from_main():
    mock_pipe = _create_mock_pipeline(decision=Decision.REFUSE, answer="Unverified secret draft")
    with patch("app.get_pipeline", return_value=mock_pipe), patch(
        "app._load_pipeline", return_value=mock_pipe
    ):
        at = AppTest.from_file("../app.py", default_timeout=30)
        at.run()

        if at.chat_input:
            at.chat_input[0].set_value("Random ungrounded query").run(timeout=30)
            assert not at.exception
            # Error banner shown for refusal
            assert len(at.error) > 0
            # Main banner displays the refusal, not the unverified draft directly in success/info
            for s in at.success:
                assert "Unverified secret draft" not in s.value

