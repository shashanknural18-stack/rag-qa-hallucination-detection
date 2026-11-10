"""Orchestrates prompt building, LLM completion, and citation parsing."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from src.config import LLM_TEMPERATURE, NO_ANSWER_TEXT, NUM_SAMPLES, SAMPLE_TEMPERATURE
from src.generation.citation_parser import parse_citations
from src.generation.prompt_builder import build_prompt
from src.generation.schemas import GeneratedAnswer

if TYPE_CHECKING:
    from src.generation.llm import LocalLLM
    from src.schemas import RetrievalResult

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates a grounded answer from retrieved evidence.

    Parameters
    ----------
    llm:
        A :class:`LocalLLM` instance.  Accepts ``None`` for unit-testing with
        a mock / ``FakeLLM`` that is later injected via the attribute.
    """

    def __init__(self, llm: "LocalLLM | None" = None) -> None:
        self.llm = llm

    def generate(
        self,
        retrieval_result: "RetrievalResult",
        raw_query: str,
    ) -> GeneratedAnswer:
        """Produce a single grounded answer.

        If *retrieval_result* contains no chunks the method returns a refusal
        **without** calling the LLM.

        Parameters
        ----------
        retrieval_result:
            The output of :class:`RetrievalModule.retrieve`.
        raw_query:
            The whitespace-normalized user query.

        Returns
        -------
        GeneratedAnswer
        """
        chunks = retrieval_result.chunks

        # Fast-path: nothing retrieved → refuse without touching the LLM
        if not chunks:
            return GeneratedAnswer(
                query=raw_query,
                answer=NO_ANSWER_TEXT,
                cited_ids=[],
                invalid_citations=[],
                chunks=[],
                is_refusal=True,
                model_name=self._model_name,
                temperature=LLM_TEMPERATURE,
            )

        prompt = build_prompt(raw_query, chunks)
        assert self.llm is not None, "LLM must be set before calling generate()"
        answer = self.llm.complete(prompt, temperature=LLM_TEMPERATURE)

        valid_ids, invalid_ids = parse_citations(answer, len(chunks))
        is_refusal = NO_ANSWER_TEXT.lower() in answer.lower()

        logger.info(
            "generated answer: %d chars, cited=%s, invalid=%s, refusal=%s",
            len(answer),
            valid_ids,
            invalid_ids,
            is_refusal,
        )

        return GeneratedAnswer(
            query=raw_query,
            answer=answer,
            cited_ids=valid_ids,
            invalid_citations=invalid_ids,
            chunks=list(chunks),
            is_refusal=is_refusal,
            model_name=self._model_name,
            temperature=LLM_TEMPERATURE,
        )

    def sample(
        self,
        retrieval_result: "RetrievalResult",
        raw_query: str,
        n: int = NUM_SAMPLES,
    ) -> list[str]:
        """Generate *n* stochastic answers at ``SAMPLE_TEMPERATURE``.

        These are used downstream by SelfCheckGPT.  If retrieval is empty the
        method returns an empty list without calling the LLM.

        Parameters
        ----------
        retrieval_result:
            The output of :class:`RetrievalModule.retrieve`.
        raw_query:
            The whitespace-normalized user query.
        n:
            Number of samples to draw.

        Returns
        -------
        list[str]
        """
        chunks = retrieval_result.chunks
        if not chunks:
            return []

        prompt = build_prompt(raw_query, chunks)
        assert self.llm is not None, "LLM must be set before calling sample()"

        samples: list[str] = []
        for i in range(n):
            text = self.llm.complete(
                prompt,
                temperature=SAMPLE_TEMPERATURE,
                seed=42 + i,
            )
            samples.append(text)

        logger.info("sampled %d answers at temperature %.2f", len(samples), SAMPLE_TEMPERATURE)
        return samples

    @property
    def _model_name(self) -> str:
        """Return the model basename, or an empty string if no LLM is set."""
        if self.llm is not None and hasattr(self.llm, "model_path"):
            return os.path.basename(self.llm.model_path)
        return ""

