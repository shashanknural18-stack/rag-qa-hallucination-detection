"""Tests for Module 5 (Hallucination Detection). No real models are loaded."""

import json
import re

import numpy as np
import pytest

from src.config import (
    JUDGE_MAX_TOKENS,
    NO_ANSWER_TEXT,
    REFUSAL_TEXT,
    THRESHOLD_SHOW,
    THRESHOLD_WARN,
    WARNING_TEXT,
    WEIGHTS,
)
from src.detection.aggregator import aggregate
from src.detection.citation_detector import CitationDetector
from src.detection.factuality_judge import FactualityJudge
from src.detection.nli_detector import NLIDetector
from src.detection.runner import DetectionRunner
from src.detection.schemas import Decision, DetectionReport, DetectorResult
from src.detection.selfcheck_detector import SelfCheckDetector
from src.detection.sentence_splitter import sentence_citations, split_sentences
from src.generation.schemas import GeneratedAnswer
from src.schemas import RetrievedChunk

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CHUNK_A = "E-commerce platforms must acknowledge a return request within 48 hours."
_CHUNK_B = "The buyer is entitled to replacement, refund or repair."


def _chunk(text: str, source: str, chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source=source,
        page=0,
        chunk_id=chunk_id,
        rerank_score=0.9,
        rank=chunk_id + 1,
    )


def _generated(
    answer: str,
    chunks: list[RetrievedChunk] | None = None,
    samples: list[str] | None = None,
    is_refusal: bool = False,
    cited_ids: list[int] | None = None,
    invalid: list[int] | None = None,
) -> GeneratedAnswer:
    return GeneratedAnswer(
        query="How fast must a return be acknowledged?",
        answer=answer,
        cited_ids=cited_ids or [],
        invalid_citations=invalid or [],
        chunks=list(chunks or []),
        is_refusal=is_refusal,
        samples=list(samples or []),
        model_name="fake-model.gguf",
        temperature=0.1,
    )


def _two_chunks() -> list[RetrievedChunk]:
    return [
        _chunk(_CHUNK_A, "consumer_rights.txt", 0),
        _chunk(_CHUNK_B, "consumer_rights.txt", 1),
    ]


def _grounded_answer() -> str:
    return f"{_CHUNK_A} [1] {_CHUNK_B} [2]"


class FakeNLI:
    """Scripted NLI probabilities. No model file."""

    def __init__(self, resolver=None, default=None) -> None:
        self.resolver = resolver
        self.default = default or {
            "entailment": 0.1,
            "neutral": 0.8,
            "contradiction": 0.1,
        }
        self.calls: list[tuple[str, str]] = []
        self.error: str | None = None

    def entailment_probs(self, premise: str, hypothesis: str) -> dict[str, float]:
        return self.entailment_probs_batch([(premise, hypothesis)])[0]

    def entailment_probs_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        if self.error:
            raise RuntimeError(self.error)
        rows = []
        for premise, hypothesis in pairs:
            self.calls.append((premise, hypothesis))
            if self.resolver is not None:
                rows.append(self.resolver(premise, hypothesis))
            else:
                rows.append(dict(self.default))
        return rows


class FakeLLM:
    """Returns scripted judge strings and records the call settings."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 400,
        seed: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
            }
        )
        if not self.responses:
            return ""
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


class FakeEmbedder:
    """Maps exact strings to vectors. Unknown strings use ``default``."""

    def __init__(self, table: dict[str, np.ndarray] | None = None, default: np.ndarray | None = None) -> None:
        self.table = table or {}
        self.default = default if default is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.calls: list[list[str]] = []

    def encode_batch(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        self.calls.append(list(texts))
        rows = [np.asarray(self.table.get(text, self.default), dtype=np.float32) for text in texts]
        return np.stack(rows)


def _unit(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def _probs(entailment: float, contradiction: float = 0.02, neutral: float = 0.0) -> dict[str, float]:
    if neutral == 0.0:
        neutral = max(0.0, 1.0 - entailment - contradiction)
    return {"entailment": entailment, "neutral": neutral, "contradiction": contradiction}


def _result(
    name: str,
    score: float,
    skipped: bool = False,
    contradiction: list[float] | None = None,
) -> DetectorResult:
    details = {}
    if contradiction is not None:
        details["contradiction_probs"] = contradiction
    return DetectorResult(
        name=name,
        score=score,
        sentence_scores=[score],
        details=details,
        skipped=skipped,
    )


def _report(*results: DetectorResult) -> DetectionReport:
    return DetectionReport(results=list(results), sentences=["Platforms acknowledge returns quickly."])


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

class TestSentenceSplitter:
    """split_sentences and sentence_citations."""

    def test_strips_markers_and_splits_on_punctuation(self):
        answer = (
            "Platforms must acknowledge a return request within 48 hours. [1] "
            "Buyers may seek a refund or a repair. [2][3]"
        )
        sentences = split_sentences(answer)
        assert sentences == [
            "Platforms must acknowledge a return request within 48 hours.",
            "Buyers may seek a refund or a repair.",
        ]
        assert "[1]" not in sentences[0]
        assert "[" not in sentences[1]

    def test_sentence_citations_keep_ids_in_order(self):
        answer = (
            "Platforms must acknowledge a return request within 48 hours. [1] "
            "Buyers may seek a refund or a repair. [2][3]"
        )
        assert sentence_citations(answer) == [[1], [2, 3]]

    def test_comma_separated_citations_stay_with_the_sentence(self):
        answer = "See the policy and the statute for details. [1, 3]"
        assert split_sentences(answer) == ["See the policy and the statute for details."]
        assert sentence_citations(answer) == [[1, 3]]

    def test_inline_citation_is_stripped_and_recorded(self):
        answer = "The rule in [1] requires a fast acknowledgement of returns."
        assert split_sentences(answer) == ["The rule in requires a fast acknowledgement of returns."]
        assert sentence_citations(answer) == [[1]]

    def test_duplicate_ids_are_collapsed(self):
        answer = "Platforms must acknowledge a return request within 48 hours. [1][1]"
        assert sentence_citations(answer) == [[1]]

    def test_short_fragments_are_dropped(self):
        answer = "No. Platforms must acknowledge a return request within 48 hours. [1]"
        sentences = split_sentences(answer)
        assert sentences == ["Platforms must acknowledge a return request within 48 hours."]
        assert sentence_citations(answer) == [[1]]
        assert all(len(re.findall(r"[A-Za-z0-9]+", sentence)) >= 3 for sentence in sentences)

    def test_citations_align_with_kept_sentences(self):
        answer = _grounded_answer()
        assert len(split_sentences(answer)) == len(sentence_citations(answer))


# ---------------------------------------------------------------------------
# NLI detector
# ---------------------------------------------------------------------------

class TestNLIDetector:
    """NLIDetector with a scripted model."""

    def test_grounded_sentences_score_near_one(self):
        def resolver(premise: str, hypothesis: str) -> dict[str, float]:
            return _probs(0.92, 0.03)

        generated = _generated(_grounded_answer(), _two_chunks())
        result = NLIDetector(FakeNLI(resolver)).detect(generated, split_sentences(generated.answer))
        assert result.score == pytest.approx(1.0)
        assert all(score >= 0.9 for score in result.sentence_scores)
        assert result.skipped is False

    def test_unsupported_sentences_score_near_zero(self):
        generated = _generated(_grounded_answer(), _two_chunks())
        result = NLIDetector(FakeNLI(default=_probs(0.08, 0.2))).detect(
            generated, split_sentences(generated.answer)
        )
        assert result.score == pytest.approx(0.0)
        assert all(score < 0.5 for score in result.sentence_scores)

    def test_mixed_sentences_score_the_supported_fraction(self):
        def resolver(premise: str, hypothesis: str) -> dict[str, float]:
            if hypothesis.startswith("E-commerce"):
                return _probs(0.9, 0.02)
            return _probs(0.1, 0.2)

        generated = _generated(_grounded_answer(), _two_chunks())
        sentences = split_sentences(generated.answer)
        result = NLIDetector(FakeNLI(resolver)).detect(generated, sentences)
        assert result.score == pytest.approx(0.5)
        assert result.sentence_scores[0] == pytest.approx(0.9)
        assert result.sentence_scores[1] == pytest.approx(0.1)

    def test_picks_the_max_entailment_across_chunks(self):
        weak = "Shipping delays depend on the weather and the carrier."
        strong = "A return request must be acknowledged within 48 hours."

        def resolver(premise: str, hypothesis: str) -> dict[str, float]:
            if premise == strong:
                return _probs(0.91, 0.04)
            return _probs(0.15, 0.70)

        chunks = [_chunk(weak, "other.txt", 0), _chunk(strong, "consumer_rights.txt", 1)]
        answer = "A return request must be acknowledged within 48 hours."
        generated = _generated(answer, chunks)
        result = NLIDetector(FakeNLI(resolver)).detect(generated, split_sentences(answer))
        assert result.sentence_scores == [pytest.approx(0.91)]
        assert result.details["best_chunk_ids"] == [2]
        assert result.details["contradiction_probs"] == [pytest.approx(0.70)]
        assert result.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SelfCheck
# ---------------------------------------------------------------------------

class TestSelfCheck:
    """SelfCheckDetector with a scripted NLI model."""

    def _generated(self, samples: list[str]) -> tuple[GeneratedAnswer, list[str]]:
        answer = "Platforms must acknowledge a return request within 48 hours. [1]"
        generated = _generated(answer, [_chunk(_CHUNK_A, "consumer_rights.txt", 0)], samples=samples)
        return generated, split_sentences(answer)

    def test_consistent_samples_score_high(self):
        def resolver(premise: str, hypothesis: str) -> dict[str, float]:
            return _probs(0.9, 0.0)

        generated, sentences = self._generated(["yes the documents agree", "yes they agree again"])
        result = SelfCheckDetector(FakeNLI(resolver)).detect(generated, sentences)
        assert result.score == pytest.approx(0.9)
        assert result.sentence_scores == [pytest.approx(0.9)]
        assert result.skipped is False

    def test_contradicting_samples_score_low(self):
        def resolver(premise: str, hypothesis: str) -> dict[str, float]:
            return _probs(0.1, 0.85)

        generated, sentences = self._generated(["no that is false here", "no the sample disagrees"])
        result = SelfCheckDetector(FakeNLI(resolver)).detect(generated, sentences)
        assert result.score == pytest.approx(0.0)

    def test_sentence_score_is_the_mean_over_samples(self):
        def resolver(premise: str, hypothesis: str) -> dict[str, float]:
            if premise.startswith("yes"):
                return _probs(0.9, 0.0)
            return _probs(0.1, 0.85)

        generated, sentences = self._generated(["yes this sample supports it", "no this sample contradicts it"])
        result = SelfCheckDetector(FakeNLI(resolver)).detect(generated, sentences)
        assert result.sentence_scores == [pytest.approx(0.45)]
        assert result.score == pytest.approx(0.45)

    def test_no_samples_is_skipped(self):
        fake = FakeNLI()
        generated, sentences = self._generated([])
        result = SelfCheckDetector(fake).detect(generated, sentences)
        assert result.skipped is True
        assert result.skip_reason == "no samples"
        assert fake.calls == []


# ---------------------------------------------------------------------------
# Factuality judge
# ---------------------------------------------------------------------------

class TestFactualityJudge:
    """JSON parsing and the supported-claim ratio."""

    def _sentences(self) -> list[str]:
        return split_sentences(_grounded_answer())

    def _generated(self) -> GeneratedAnswer:
        return _generated(_grounded_answer(), _two_chunks())

    def test_parses_clean_json_and_uses_judge_settings(self):
        payload = {
            "claims": [
                {"claim": _CHUNK_A, "verdict": "supported"},
                {"claim": _CHUNK_B, "verdict": "unsupported"},
            ]
        }
        llm = FakeLLM([json.dumps(payload)])
        result = FactualityJudge(llm).detect(self._generated(), self._sentences())
        assert result.skipped is False
        assert result.score == pytest.approx(0.5)
        assert result.sentence_scores == [pytest.approx(1.0), pytest.approx(0.0)]
        assert len(llm.calls) == 1
        assert llm.calls[0]["temperature"] == 0.0
        assert llm.calls[0]["max_tokens"] == JUDGE_MAX_TOKENS
        assert llm.calls[0]["prompt"].startswith("<s>[INST]")
        assert llm.calls[0]["prompt"].endswith("[/INST]")
        assert "consumer_rights.txt" in llm.calls[0]["prompt"]

    def test_parses_json_inside_code_fences(self):
        body = json.dumps({"claims": [{"claim": "anything at all here", "verdict": "supported"}]})
        llm = FakeLLM([f"```json\n{body}\n```"])
        result = FactualityJudge(llm).detect(self._generated(), self._sentences())
        assert result.score == pytest.approx(1.0)
        assert result.sentence_scores == [pytest.approx(1.0), pytest.approx(1.0)]
        assert len(llm.calls) == 1

    def test_parses_json_with_trailing_text(self):
        body = json.dumps(
            {"claims": [{"claim": _CHUNK_A, "verdict": "Supported"}, {"claim": _CHUNK_B, "verdict": "unsupported"}]}
        )
        llm = FakeLLM([f"Here is the result:\n{body}\nThanks, that is all."])
        result = FactualityJudge(llm).detect(self._generated(), self._sentences())
        assert result.score == pytest.approx(0.5)
        assert len(llm.calls) == 1

    def test_unparseable_output_twice_is_skipped(self):
        llm = FakeLLM(["I will not use JSON.", "Still no JSON here."])
        result = FactualityJudge(llm).detect(self._generated(), self._sentences())
        assert result.skipped is True
        assert result.skip_reason == "judge output not parseable"
        assert len(llm.calls) == 2

    def test_retries_once_then_accepts_json(self):
        payload = json.dumps({"claims": [{"claim": _CHUNK_A, "verdict": "supported"}]})
        llm = FakeLLM(["not json", payload])
        result = FactualityJudge(llm).detect(self._generated(), self._sentences())
        assert result.skipped is False
        assert result.score == pytest.approx(1.0)
        assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# Citation detector
# ---------------------------------------------------------------------------

class TestCitationDetector:
    """Citation validity plus cosine support."""

    def test_valid_and_similar_passes(self):
        answer = f"{_CHUNK_A} [1]"
        sentence = split_sentences(answer)[0]
        embedder = FakeEmbedder({sentence: _unit(1, 0, 0), _CHUNK_A: _unit(1, 0, 0)})
        generated = _generated(answer, [_chunk(_CHUNK_A, "consumer_rights.txt", 0)])
        result = CitationDetector(embedder).detect(generated, split_sentences(answer))
        assert result.score == pytest.approx(1.0)
        assert result.details["per_sentence"][0]["supported"] is True

    def test_invalid_id_fails(self):
        answer = "Platforms must acknowledge a return request within 48 hours. [9]"
        generated = _generated(answer, [_chunk(_CHUNK_A, "consumer_rights.txt", 0)])
        result = CitationDetector(FakeEmbedder()).detect(generated, split_sentences(answer))
        assert result.score == pytest.approx(0.0)
        assert result.details["per_sentence"][0]["invalid_ids"] == [9]

    def test_missing_citation_fails(self):
        answer = "Platforms must acknowledge a return request within 48 hours."
        generated = _generated(answer, [_chunk(_CHUNK_A, "consumer_rights.txt", 0)])
        result = CitationDetector(FakeEmbedder()).detect(generated, split_sentences(answer))
        assert result.score == pytest.approx(0.0)
        assert result.details["per_sentence"][0]["cited_ids"] == []

    def test_low_similarity_fails(self):
        answer = "The courier left the parcel at the wrong address yesterday. [1]"
        sentence = split_sentences(answer)[0]
        embedder = FakeEmbedder(
            {sentence: _unit(1, 0, 0), _CHUNK_A: _unit(0, 1, 0)},
            default=_unit(0, 0, 1),
        )
        generated = _generated(answer, [_chunk(_CHUNK_A, "consumer_rights.txt", 0)])
        result = CitationDetector(embedder).detect(generated, split_sentences(answer))
        assert result.score == pytest.approx(0.0)
        similarity = result.details["per_sentence"][0]["similarities"][0]["similarity"]
        assert similarity == pytest.approx(0.0)

    def test_one_supported_sentence_of_two_is_half(self):
        second = "Buyers can ask for a replacement refund or repair."
        answer = f"{_CHUNK_A} [1] {second} [2]"
        sentences = split_sentences(answer)
        embedder = FakeEmbedder(
            {
                sentences[0]: _unit(1, 0, 0),
                _CHUNK_A: _unit(1, 0, 0),
                sentences[1]: _unit(0, 1, 0),
                _CHUNK_B: _unit(1, 0, 0),
            }
        )
        result = CitationDetector(embedder).detect(_generated(answer, _two_chunks()), sentences)
        assert result.score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class TestAggregator:
    """Weights, thresholds, and the contradiction cap."""

    def test_weighted_average_uses_config_weights(self):
        report = _report(
            _result("nli", 1.0),
            _result("selfcheck", 0.0),
            _result("factuality", 0.5),
            _result("citation", 1.0),
        )
        confidence, _decision = aggregate(report)
        expected = (
            WEIGHTS["nli"] * 1.0
            + WEIGHTS["selfcheck"] * 0.0
            + WEIGHTS["factuality"] * 0.5
            + WEIGHTS["citation"] * 1.0
        )
        assert confidence == pytest.approx(expected)

    def test_skipped_detectors_are_renormalized(self):
        report = _report(
            _result("nli", 1.0, skipped=True),
            _result("selfcheck", 1.0),
            _result("factuality", 0.0),
            _result("citation", 1.0),
        )
        confidence, _decision = aggregate(report)
        remaining = WEIGHTS["selfcheck"] + WEIGHTS["factuality"] + WEIGHTS["citation"]
        expected = (WEIGHTS["selfcheck"] * 1.0 + WEIGHTS["citation"] * 1.0) / remaining
        assert confidence == pytest.approx(expected)

    def test_boundary_show(self):
        confidence, decision = aggregate(_report(_result("nli", 0.75, contradiction=[0.1])))
        assert confidence == pytest.approx(0.75)
        assert decision is Decision.SHOW

    def test_boundary_warn(self):
        _confidence, decision = aggregate(_report(_result("nli", 0.50)))
        assert decision is Decision.WARN

    def test_boundary_refuse(self):
        _confidence, decision = aggregate(_report(_result("nli", 0.49)))
        assert decision is Decision.REFUSE

    def test_all_skipped_refuses(self):
        report = _report(
            _result("nli", 1.0, skipped=True),
            _result("selfcheck", 1.0, skipped=True),
            _result("factuality", 1.0, skipped=True),
            _result("citation", 1.0, skipped=True),
        )
        confidence, decision = aggregate(report)
        assert confidence == pytest.approx(0.0)
        assert decision is Decision.REFUSE

    def test_contradiction_above_0_8_caps_show_at_warn(self):
        report = _report(
            _result("nli", 1.0, contradiction=[0.81]),
            _result("selfcheck", 1.0),
            _result("factuality", 1.0),
            _result("citation", 1.0),
        )
        confidence, decision = aggregate(report)
        assert confidence == pytest.approx(1.0)
        assert decision is Decision.WARN

    def test_contradiction_at_0_8_does_not_cap(self):
        report = _report(_result("nli", 1.0, contradiction=[0.8]))
        _confidence, decision = aggregate(report)
        assert decision is Decision.SHOW


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TestRunner:
    """DetectionRunner orchestration with fakes."""

    def test_refusal_input_skips_every_detector(self):
        nli = FakeNLI()
        llm = FakeLLM([json.dumps({"claims": []})])
        embedder = FakeEmbedder()
        generated = _generated(NO_ANSWER_TEXT, _two_chunks(), is_refusal=True)
        response = DetectionRunner(nli, llm, embedder).run(generated)
        assert response.decision is Decision.REFUSE
        assert response.answer == NO_ANSWER_TEXT
        assert response.confidence == pytest.approx(0.0)
        assert response.report is None
        assert response.generated is generated
        assert nli.calls == []
        assert llm.calls == []
        assert embedder.calls == []

    def test_empty_chunks_skip_detectors(self):
        nli = FakeNLI()
        generated = _generated("Some answer that should not be shown here.", chunks=[])
        response = DetectionRunner(nli, FakeLLM([]), FakeEmbedder()).run(generated)
        assert response.decision is Decision.REFUSE
        assert response.answer == NO_ANSWER_TEXT
        assert nli.calls == []

    def test_detector_exception_is_skipped_and_run_completes(self):
        nli = FakeNLI()
        nli.error = "nli exploded"
        llm = FakeLLM([json.dumps({"claims": [{"claim": _CHUNK_A, "verdict": "supported"}]})])
        answer = f"{_CHUNK_A} [1]"
        generated = _generated(
            answer,
            [_chunk(_CHUNK_A, "consumer_rights.txt", 0)],
            samples=["A long sample that should trigger selfcheck."],
        )
        response = DetectionRunner(nli, llm, FakeEmbedder()).run(generated)
        assert response.report is not None
        by_name = {result.name: result for result in response.report.results}
        assert by_name["nli"].skipped is True
        assert "nli exploded" in by_name["nli"].skip_reason
        assert by_name["selfcheck"].skipped is True
        assert by_name["factuality"].skipped is False
        assert by_name["citation"].skipped is False
        assert response.decision in {Decision.SHOW, Decision.WARN, Decision.REFUSE}

    def test_low_confidence_replaces_the_answer(self):
        nli = FakeNLI(default=_probs(0.05, 0.1))
        llm = FakeLLM(
            [json.dumps({"claims": [{"claim": "totally invented claim text", "verdict": "unsupported"}]})]
        )
        answer = _grounded_answer()
        sentences = split_sentences(answer)
        embedder = FakeEmbedder(
            {
                sentences[0]: _unit(1, 0, 0),
                _CHUNK_A: _unit(0, 1, 0),
                sentences[1]: _unit(1, 0, 0),
                _CHUNK_B: _unit(0, 1, 0),
            }
        )
        generated = _generated(answer, _two_chunks())
        response = DetectionRunner(nli, llm, embedder).run(generated)
        assert response.decision is Decision.REFUSE
        assert response.answer == REFUSAL_TEXT
        assert response.warning is None
        assert generated.answer == answer
        assert response.generated is generated

    def test_warn_keeps_the_answer_and_sets_warning(self):
        def resolver(premise: str, hypothesis: str) -> dict[str, float]:
            if hypothesis.startswith("E-commerce"):
                return _probs(0.9, 0.02)
            return _probs(0.1, 0.1)

        second = "Buyers can ask for a replacement refund or repair."
        answer = f"{_CHUNK_A} [1] {second} [2]"
        sentences = split_sentences(answer)
        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "claims": [
                            {"claim": _CHUNK_A, "verdict": "supported"},
                            {"claim": second, "verdict": "unsupported"},
                        ]
                    }
                )
            ]
        )
        embedder = FakeEmbedder(
            {
                sentences[0]: _unit(1, 0, 0),
                _CHUNK_A: _unit(1, 0, 0),
                sentences[1]: _unit(0, 1, 0),
                _CHUNK_B: _unit(1, 0, 0),
            }
        )
        generated = _generated(answer, _two_chunks())
        response = DetectionRunner(FakeNLI(resolver), llm, embedder).run(generated)
        assert response.decision is Decision.WARN
        assert response.confidence == pytest.approx(0.5)
        assert THRESHOLD_WARN <= response.confidence < THRESHOLD_SHOW
        assert response.answer == answer
        assert response.warning == WARNING_TEXT

    def test_skip_judge_does_not_call_the_llm(self):
        llm = FakeLLM([json.dumps({"claims": [{"claim": _CHUNK_A, "verdict": "supported"}]})])
        answer = f"{_CHUNK_A} [1]"
        generated = _generated(answer, [_chunk(_CHUNK_A, "consumer_rights.txt", 0)])
        response = DetectionRunner(FakeNLI(default=_probs(0.95, 0.02)), llm, FakeEmbedder()).run(
            generated,
            skip_judge=True,
        )
        assert llm.calls == []
        factuality = next(result for result in response.report.results if result.name == "factuality")
        assert factuality.skipped is True
        assert factuality.skip_reason == "disabled"


# ---------------------------------------------------------------------------
# Hand-made grounded vs hallucinated answers
# ---------------------------------------------------------------------------

def _overlap_nli() -> FakeNLI:
    def resolver(premise: str, hypothesis: str) -> dict[str, float]:
        premise_words = set(re.findall(r"[a-z0-9]+", premise.lower()))
        hypothesis_words = set(re.findall(r"[a-z0-9]+", hypothesis.lower()))
        overlap = len(premise_words & hypothesis_words) / max(len(hypothesis_words), 1)
        if overlap >= 0.6:
            return _probs(0.95, 0.02)
        return _probs(0.05, 0.85)

    return FakeNLI(resolver)


def _bow_embedder() -> FakeEmbedder:
    def encode_batch(texts, batch_size=32, show_progress=False):
        rows = []
        for text in texts:
            vector = np.zeros(128, dtype=np.float32)
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vector[hash(word) % 128] += 1.0
            norm = float(np.linalg.norm(vector))
            if norm:
                vector /= norm
            rows.append(vector)
        return np.stack(rows)

    embedder = FakeEmbedder()
    embedder.encode_batch = encode_batch  # type: ignore[method-assign]
    return embedder


class _BranchingLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__([])

    def complete(self, prompt: str, temperature: float = 0.1, max_tokens: int = 400, seed: int | None = None) -> str:
        if "Paris" in prompt or "50 lakh" in prompt:
            payload = {
                "claims": [
                    {
                        "claim": "The platform must pay a penalty of 50 lakh rupees within one day.",
                        "verdict": "unsupported",
                    },
                    {
                        "claim": "The consumer court sits in Paris every Tuesday afternoon.",
                        "verdict": "unsupported",
                    },
                ]
            }
        else:
            payload = {
                "claims": [
                    {"claim": _CHUNK_A, "verdict": "supported"},
                    {"claim": _CHUNK_B, "verdict": "supported"},
                ]
            }
        self.responses = [json.dumps(payload)]
        return super().complete(prompt, temperature, max_tokens, seed)


class TestHandMadeAnswers:
    """A fully grounded answer should pass; an invented one should be refused."""

    def test_grounded_answer_is_shown(self):
        answer = _grounded_answer()
        sample = f"{_CHUNK_A} {_CHUNK_B}"
        generated = _generated(answer, _two_chunks(), samples=[sample], cited_ids=[1, 2])
        response = DetectionRunner(_overlap_nli(), _BranchingLLM(), _bow_embedder()).run(generated)
        assert response.decision in {Decision.SHOW, Decision.WARN}
        assert response.confidence >= 0.7
        if response.decision is Decision.WARN:
            assert response.confidence >= THRESHOLD_WARN
        else:
            assert response.confidence >= THRESHOLD_SHOW
        assert response.answer == answer
        for result in response.report.results:
            if not result.skipped:
                assert 0.0 <= result.score <= 1.0

    def test_hallucinated_answer_is_refused(self):
        answer = (
            "The platform must pay a penalty of 50 lakh rupees within one day. [1] "
            "The consumer court sits in Paris every Tuesday afternoon. [2]"
        )
        sample = f"{_CHUNK_A} {_CHUNK_B}"
        generated = _generated(answer, _two_chunks(), samples=[sample], cited_ids=[1, 2])
        response = DetectionRunner(_overlap_nli(), _BranchingLLM(), _bow_embedder()).run(generated)
        assert response.decision is Decision.REFUSE
        assert response.answer == REFUSAL_TEXT
        assert response.confidence < THRESHOLD_WARN
        assert generated.answer == answer
        for result in response.report.results:
            if not result.skipped:
                assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Real NLI model (slow; skipped unless the weights are already cached)
# ---------------------------------------------------------------------------

def _nli_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    from src.config import NLI_MODEL

    resolved = try_to_load_from_cache(NLI_MODEL, "config.json")
    return isinstance(resolved, str)


@pytest.mark.slow
def test_real_nli_model_smoke():
    """Entailment should outrank contradiction on one clear pair, and the reverse on another."""
    if not _nli_cached():
        pytest.skip("NLI model is not in the local Hugging Face cache")

    from src.detection.nli_model import NLIModel

    model = NLIModel()
    entailed = model.entailment_probs(
        "E-commerce platforms must acknowledge a return request within 48 hours.",
        "A return request must be acknowledged within 48 hours.",
    )
    contradicted = model.entailment_probs(
        "E-commerce platforms must acknowledge a return request within 48 hours.",
        "Platforms are forbidden from acknowledging return requests.",
    )
    assert entailed["entailment"] > entailed["contradiction"]
    assert contradicted["contradiction"] > contradicted["entailment"]
    assert entailed["entailment"] > contradicted["entailment"]
