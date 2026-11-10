"""LLM-as-judge: atomic claims labelled supported or unsupported."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING

from src.config import JUDGE_MAX_TOKENS
from src.detection.base import BaseDetector, clamp_unit
from src.detection.schemas import DetectorResult

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

if TYPE_CHECKING:
    from src.generation.schemas import GeneratedAnswer

logger = logging.getLogger(__name__)

_SUPPORTED = {"supported", "support", "yes", "true"}


def build_judge_prompt(generated: GeneratedAnswer) -> str:
    """Build a Mistral ``[INST]`` prompt that asks for strict JSON claims.

    Parameters
    ----------
    generated:
        The answer and the chunks the judge is allowed to use.

    Returns
    -------
    str
        ``<s>[INST] ... [/INST]`` prompt.
    """
    blocks: list[str] = []
    for index, chunk in enumerate(generated.chunks, start=1):
        source = os.path.basename(chunk.source)
        blocks.append(f"[{index}] (source: {source})\n{chunk.text}")
    context = "\n\n".join(blocks)
    inner = (
        "You are a factuality judge. Use ONLY the numbered context below. "
        "Do not use outside knowledge.\n"
        "List each atomic claim in the answer and label it supported or unsupported.\n"
        "Return STRICT JSON and nothing else, in this form:\n"
        '{"claims":[{"claim":"...","verdict":"supported"},'
        '{"claim":"...","verdict":"unsupported"}]}\n'
        'Each verdict must be exactly "supported" or "unsupported".\n\n'
        f"Context:\n{context}\n\n"
        f"Answer:\n{generated.answer}"
    )
    return f"<s>[INST] {inner} [/INST]"


def _extract_claims(text: str) -> list[dict] | None:
    """Pull the first JSON object out of *text* and normalise its claims.

    Returns ``None`` when no usable object is present. An object with an empty
    ``claims`` list is valid and returns ``[]``.
    """
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            obj, _end = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "claims" not in obj:
            continue
        raw_claims = obj["claims"]
        if not isinstance(raw_claims, list):
            continue
        claims: list[dict] = []
        for item in raw_claims:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            verdict = str(item.get("verdict", "")).strip().lower()
            claims.append(
                {
                    "claim": claim,
                    "verdict": "supported" if verdict in _SUPPORTED else "unsupported",
                }
            )
        return claims
    return None


def _normalise(text: str) -> str:
    text = _CITATION_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _sentence_scores(sentences: list[str], claims: list[dict], overall: float) -> list[float]:
    """Map claim verdicts onto sentences. Unmatched sentences get *overall*."""
    if not sentences:
        return []
    scores: list[float] = []
    normalised_sentences = [_normalise(sentence) for sentence in sentences]
    for sentence in normalised_sentences:
        matched: list[float] = []
        for claim in claims:
            claim_text = _normalise(claim["claim"])
            if not claim_text:
                continue
            if claim_text in sentence or sentence in claim_text:
                matched.append(1.0 if claim["verdict"] == "supported" else 0.0)
        if matched:
            scores.append(sum(matched) / len(matched))
        else:
            scores.append(overall)
    return [clamp_unit(value) for value in scores]


class FactualityJudge(BaseDetector):
    """Ask the shared local LLM to label atomic claims as supported or not.

    The prompt uses the Mistral ``[INST]`` format. Generation uses temperature
    0 and ``JUDGE_MAX_TOKENS``. The first JSON object is parsed, including JSON
    wrapped in code fences or followed by extra text. A failed parse is retried
    once; a second failure skips the detector with reason
    ``"judge output not parseable"``.

    The score is supported claims divided by total claims. Sentence scores
    follow claims that overlap a sentence; other sentences repeat the overall
    score.

    Parameters
    ----------
    llm:
        Shared :class:`src.generation.llm.LocalLLM`, or a test double.
    """

    name = "factuality"

    def __init__(self, llm) -> None:
        self.llm = llm

    def detect(
        self,
        generated: GeneratedAnswer,
        sentences: list[str],
    ) -> DetectorResult:
        """Judge *generated.answer* against its retrieved chunks."""
        if self.llm is None:
            return DetectorResult(
                name=self.name,
                score=0.0,
                sentence_scores=[],
                details={},
                skipped=True,
                skip_reason="no llm",
            )

        prompt = build_judge_prompt(generated)
        claims: list[dict] | None = None
        last_text = ""
        for _attempt in range(2):
            last_text = self.llm.complete(
                prompt,
                temperature=0.0,
                max_tokens=JUDGE_MAX_TOKENS,
            )
            claims = _extract_claims(last_text)
            if claims is not None:
                break

        if claims is None:
            logger.info("factuality judge output was not parseable")
            return DetectorResult(
                name=self.name,
                score=0.0,
                sentence_scores=[0.0] * len(sentences),
                details={"raw": last_text},
                skipped=True,
                skip_reason="judge output not parseable",
            )

        if not claims:
            overall = 0.0
        else:
            supported = sum(1 for claim in claims if claim["verdict"] == "supported")
            overall = supported / len(claims)
        overall = clamp_unit(overall)
        logger.info("factuality score %.3f over %d claims", overall, len(claims))
        return DetectorResult(
            name=self.name,
            score=overall,
            sentence_scores=_sentence_scores(sentences, claims, overall),
            details={"claims": claims},
        )
