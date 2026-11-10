"""Tests for Module 4 (Generation)."""

import os
from pathlib import Path

import pytest

from src.config import LLM_MODEL_PATH, LLM_N_CTX, LLM_MAX_TOKENS, NO_ANSWER_TEXT, SAMPLE_TEMPERATURE
from src.generation.citation_parser import parse_citations
from src.generation.generator import AnswerGenerator
from src.generation.prompt_builder import build_prompt
from src.generation.schemas import GeneratedAnswer
from src.schemas import ProcessedQuery, RetrievalResult, RetrievedChunk

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNKS = [
    RetrievedChunk(
        text="E-commerce platforms must acknowledge a return request within 48 hours.",
        source="consumer_rights.txt",
        page=0,
        chunk_id=0,
        rerank_score=0.95,
        rank=1,
    ),
    RetrievedChunk(
        text="The buyer is entitled to replacement, refund or repair.",
        source="consumer_rights.txt",
        page=0,
        chunk_id=1,
        rerank_score=0.85,
        rank=2,
    ),
    RetrievedChunk(
        text="Consent to share personal data must be free, specific, informed and unconditional.",
        source="data_privacy.txt",
        page=0,
        chunk_id=2,
        rerank_score=0.70,
        rank=3,
    ),
    RetrievedChunk(
        text="A platform cannot make sharing an identity number a condition of processing a return.",
        source="data_privacy.txt",
        page=0,
        chunk_id=3,
        rerank_score=0.65,
        rank=4,
    ),
    RetrievedChunk(
        text="A manufacturing defect qualifies as a deficiency under the Consumer Protection Act.",
        source="consumer_rights.txt",
        page=1,
        chunk_id=4,
        rerank_score=0.60,
        rank=5,
    ),
]

_DUMMY_VECTOR = np.zeros(384, dtype=np.float32)

_DUMMY_QUERY = ProcessedQuery(
    raw="How fast must a return be acknowledged?",
    cleaned="how fast must a return be acknowledged?",
    vector=_DUMMY_VECTOR,
)

_RETRIEVAL = RetrievalResult(
    query=_DUMMY_QUERY,
    chunks=_CHUNKS,
    num_candidates=10,
    reranked=True,
)

_EMPTY_RETRIEVAL = RetrievalResult(
    query=_DUMMY_QUERY,
    chunks=[],
    num_candidates=0,
    reranked=False,
)


class FakeLLM:
    """Deterministic stand-in for LocalLLM — no model file required."""

    def __init__(self, canned_response: str = "The answer is 48 hours. [1][3]") -> None:
        self.canned_response = canned_response
        self.calls: list[dict] = []
        self.model_path = "fake-model.gguf"

    def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 400,
        seed: int | None = None,
    ) -> str:
        self.calls.append(
            {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens, "seed": seed}
        )
        return self.canned_response


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    """Tests for build_prompt."""

    def test_prompt_contains_chunk_labels(self):
        """Every chunk is labelled [1]..[n] with its source filename."""
        prompt = build_prompt(_DUMMY_QUERY.raw, _CHUNKS)
        for i in range(1, len(_CHUNKS) + 1):
            assert f"[{i}]" in prompt

    def test_prompt_contains_source_filenames(self):
        """Source filenames appear in the prompt."""
        prompt = build_prompt(_DUMMY_QUERY.raw, _CHUNKS)
        assert "consumer_rights.txt" in prompt
        assert "data_privacy.txt" in prompt

    def test_prompt_uses_raw_query(self):
        """The raw query (not the cleaned one) is embedded in the prompt."""
        prompt = build_prompt(_DUMMY_QUERY.raw, _CHUNKS)
        assert _DUMMY_QUERY.raw in prompt

    def test_prompt_contains_no_answer_instruction(self):
        """The NO_ANSWER_TEXT instruction is present in the prompt."""
        prompt = build_prompt(_DUMMY_QUERY.raw, _CHUNKS)
        assert NO_ANSWER_TEXT in prompt

    def test_prompt_has_mistral_wrapper(self):
        """The Mistral [INST] wrapper is present."""
        prompt = build_prompt(_DUMMY_QUERY.raw, _CHUNKS)
        assert prompt.startswith("<s>[INST]")
        assert prompt.endswith("[/INST]")

    def test_prompt_truncation_stays_within_budget(self):
        """When chunks are very long the prompt stays under the token budget."""
        big_chunks = [
            RetrievedChunk(
                text="word " * 2000,
                source=f"big_{i}.txt",
                page=0,
                chunk_id=i,
                rank=i + 1,
            )
            for i in range(5)
        ]
        prompt = build_prompt("test?", big_chunks, n_ctx=LLM_N_CTX, max_tokens=LLM_MAX_TOKENS)
        estimated_tokens = len(prompt) // 4 + 1
        assert estimated_tokens < LLM_N_CTX - LLM_MAX_TOKENS + 100  # small margin for heuristic


# ---------------------------------------------------------------------------
# Citation parser tests
# ---------------------------------------------------------------------------

class TestCitationParser:
    """Tests for parse_citations."""

    def test_single_citation(self):
        valid, invalid = parse_citations("Returns take 48 hours. [1]", num_chunks=5)
        assert valid == [1]
        assert invalid == []

    def test_adjacent_citations(self):
        valid, invalid = parse_citations("Both policies apply. [1][2]", num_chunks=5)
        assert valid == [1, 2]
        assert invalid == []

    def test_comma_separated_citations(self):
        valid, invalid = parse_citations("See [1, 3] for details.", num_chunks=5)
        assert valid == [1, 3]
        assert invalid == []

    def test_duplicate_citations(self):
        valid, invalid = parse_citations("See [1] and also [1].", num_chunks=5)
        assert valid == [1]
        assert invalid == []

    def test_invalid_citation_id(self):
        valid, invalid = parse_citations("Claim [9] is wrong.", num_chunks=5)
        assert valid == []
        assert invalid == [9]

    def test_mixed_valid_and_invalid(self):
        valid, invalid = parse_citations("See [1][9][3].", num_chunks=5)
        assert valid == [1, 3]
        assert invalid == [9]

    def test_no_citations(self):
        valid, invalid = parse_citations("No citations here.", num_chunks=5)
        assert valid == []
        assert invalid == []


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

class TestGenerator:
    """Tests for AnswerGenerator with FakeLLM."""

    def test_empty_chunks_returns_refusal_without_llm_call(self):
        """Empty retrieval returns refusal and FakeLLM is NOT called."""
        fake = FakeLLM()
        gen = AnswerGenerator(llm=fake)
        result = gen.generate(_EMPTY_RETRIEVAL, _DUMMY_QUERY.raw)
        assert result.is_refusal is True
        assert result.answer == NO_ANSWER_TEXT
        assert fake.calls == []  # LLM was never called

    def test_generate_parses_cited_ids(self):
        """generate() with a fake answer '... [1][3].' sets cited_ids == [1, 3]."""
        fake = FakeLLM(canned_response="The answer is 48 hours. [1][3].")
        gen = AnswerGenerator(llm=fake)
        result = gen.generate(_RETRIEVAL, _DUMMY_QUERY.raw)
        assert result.cited_ids == [1, 3]
        assert result.is_refusal is False

    def test_generate_flags_invalid_citation(self):
        """generate() flags invalid citation [9] when only 5 chunks exist."""
        fake = FakeLLM(canned_response="Per the rules [9], returns take 48 hours.")
        gen = AnswerGenerator(llm=fake)
        result = gen.generate(_RETRIEVAL, _DUMMY_QUERY.raw)
        assert 9 in result.invalid_citations

    def test_generate_sets_refusal_on_no_answer_text(self):
        """generate() sets is_refusal when the fake LLM returns NO_ANSWER_TEXT."""
        fake = FakeLLM(canned_response=NO_ANSWER_TEXT)
        gen = AnswerGenerator(llm=fake)
        result = gen.generate(_RETRIEVAL, _DUMMY_QUERY.raw)
        assert result.is_refusal is True

    def test_generate_preserves_chunks(self):
        """The result carries the same chunk objects from the retrieval result."""
        fake = FakeLLM()
        gen = AnswerGenerator(llm=fake)
        result = gen.generate(_RETRIEVAL, _DUMMY_QUERY.raw)
        assert result.chunks == list(_RETRIEVAL.chunks)

    def test_sample_returns_n_strings(self):
        """sample() returns n strings and uses SAMPLE_TEMPERATURE."""
        fake = FakeLLM(canned_response="Sample answer.")
        gen = AnswerGenerator(llm=fake)
        samples = gen.sample(_RETRIEVAL, _DUMMY_QUERY.raw, n=3)
        assert len(samples) == 3
        assert all(isinstance(s, str) for s in samples)
        # Verify temperature used
        for call in fake.calls:
            assert call["temperature"] == SAMPLE_TEMPERATURE

    def test_sample_empty_retrieval_returns_empty_list(self):
        """sample() with empty retrieval returns [] without calling the LLM."""
        fake = FakeLLM()
        gen = AnswerGenerator(llm=fake)
        samples = gen.sample(_EMPTY_RETRIEVAL, _DUMMY_QUERY.raw)
        assert samples == []
        assert fake.calls == []

    def test_sample_uses_different_seeds(self):
        """Each sample call uses a different seed."""
        fake = FakeLLM()
        gen = AnswerGenerator(llm=fake)
        gen.sample(_RETRIEVAL, _DUMMY_QUERY.raw, n=3)
        seeds = [call["seed"] for call in fake.calls]
        assert len(set(seeds)) == 3  # all different

    def test_generate_returns_generatedanswer_dataclass(self):
        """generate() returns a GeneratedAnswer with expected fields."""
        fake = FakeLLM()
        gen = AnswerGenerator(llm=fake)
        result = gen.generate(_RETRIEVAL, _DUMMY_QUERY.raw)
        assert isinstance(result, GeneratedAnswer)
        assert result.query == _DUMMY_QUERY.raw
        assert result.model_name == "fake-model.gguf"


# ---------------------------------------------------------------------------
# Smoke test — real model (slow, skipped if model missing)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_model_smoke():
    """Load the real Mistral GGUF model and answer one question.

    Skipped automatically if the model file is not present.
    """
    model_path = Path(LLM_MODEL_PATH)
    if not model_path.exists():
        pytest.skip(f"Model file not found at {model_path}")

    from src.generation.llm import LocalLLM

    llm = LocalLLM(model_path=model_path)
    gen = AnswerGenerator(llm=llm)

    # Build a minimal retrieval result
    chunks = [
        RetrievedChunk(
            text="E-commerce platforms must acknowledge a return request within 48 hours and resolve it within 30 days.",
            source="consumer_rights.txt",
            page=0,
            chunk_id=0,
            rerank_score=0.95,
            rank=1,
        ),
    ]
    retrieval = RetrievalResult(
        query=_DUMMY_QUERY,
        chunks=chunks,
        num_candidates=1,
        reranked=True,
    )

    result = gen.generate(retrieval, "How fast must a return be acknowledged?")
    assert isinstance(result.answer, str)
    assert len(result.answer) > 10
    # The answer should mention 48 hours
    assert "48" in result.answer or "forty-eight" in result.answer.lower()

