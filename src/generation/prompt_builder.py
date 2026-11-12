"""Builds the Mistral [INST] prompt from retrieved chunks and a raw query."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.config import LLM_MAX_TOKENS, LLM_N_CTX, NO_ANSWER_TEXT

if TYPE_CHECKING:
    from src.schemas import RetrievedChunk

_SYSTEM = (
    "You are a helpful legal-information assistant. "
    "Answer the user's question ONLY from the numbered context below. "
    "Cite every claim with the chunk number in square brackets, e.g. [1], "
    "immediately after the sentence that uses that information. "
    "If the context does not contain the answer, reply EXACTLY with: "
    f'"{NO_ANSWER_TEXT}" '
    "Do not use any outside knowledge. Be concise."
)


def _chunk_label(index: int, chunk: "RetrievedChunk") -> str:
    """Format one numbered chunk for the prompt."""
    source = os.path.basename(chunk.source)
    return f"[{index}] (source: {source})\n{chunk.text}"


def _count_tokens(text: str) -> int:
    """Estimate token count.

    Uses a fast heuristic (≈ 1 token per 4 chars) that avoids importing
    the llama tokenizer at module level.  The prompt builder trims
    aggressively so a small over-count is safe.
    """
    return len(text) // 4 + 1


def build_prompt(
    raw_query: str,
    chunks: list["RetrievedChunk"],
    n_ctx: int = LLM_N_CTX,
    max_tokens: int = LLM_MAX_TOKENS,
    max_chunk_tokens: int | None = None,
) -> str:
    """Return a Mistral-instruct formatted prompt.

    Parameters
    ----------
    raw_query:
        The whitespace-normalized user query (not the lowercased/cleaned one).
    chunks:
        Retrieved evidence chunks, numbered starting at 1.
    n_ctx:
        Model context window in tokens.
    max_tokens:
        Tokens reserved for the generated answer.
    max_chunk_tokens:
        Optional per-chunk token limit (e.g. 300 for FAST mode).

    Returns
    -------
    str
        ``<s>[INST] ... [/INST]`` formatted prompt.
    """
    budget = n_ctx - max_tokens

    header = f"{_SYSTEM}\n\nContext:\n"
    footer = f"\n\nQuestion: {raw_query}"
    overhead_tokens = _count_tokens(header) + _count_tokens(footer) + 20  # wrapper

    available = budget - overhead_tokens
    context_parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_obj = chunk
        if max_chunk_tokens is not None:
            words = chunk.text.split()
            if len(words) > max_chunk_tokens:
                from src.schemas import RetrievedChunk
                chunk_obj = RetrievedChunk(
                    text=" ".join(words[:max_chunk_tokens]) + " …",
                    source=chunk.source,
                    page=chunk.page,
                    chunk_id=chunk.chunk_id,
                    dense_score=chunk.dense_score,
                    bm25_score=chunk.bm25_score,
                    rerank_score=chunk.rerank_score,
                    rank=chunk.rank,
                )

        label = _chunk_label(idx, chunk_obj)
        label_tokens = _count_tokens(label)
        if available - label_tokens < 0:
            # Truncate the chunk text so it fits
            safe_chars = max((available - 10) * 4, 0)
            if safe_chars > 40:
                truncated_text = chunk_obj.text[:safe_chars].rsplit(" ", 1)[0] + " …"
                label = _chunk_label(idx, chunk_obj).split("\n", 1)[0] + "\n" + truncated_text
                context_parts.append(label)
            break
        context_parts.append(label)
        available -= label_tokens

    context_block = "\n\n".join(context_parts)
    inner = f"{header}{context_block}{footer}"
    return f"<s>[INST] {inner} [/INST]"


def build_chat_messages(
    raw_query: str,
    chunks: list["RetrievedChunk"],
    max_chunk_tokens: int | None = None,
) -> list[dict[str, str]]:
    """Build chat messages structure suitable for chat templates."""
    context_parts = []
    for idx, chunk in enumerate(chunks, start=1):
        text = chunk.text
        if max_chunk_tokens is not None:
            words = text.split()
            if len(words) > max_chunk_tokens:
                text = " ".join(words[:max_chunk_tokens]) + " …"
        source = os.path.basename(chunk.source)
        context_parts.append(f"[{idx}] (source: {source})\n{text}")

    context_block = "\n\n".join(context_parts)
    user_content = f"Context:\n{context_block}\n\nQuestion: {raw_query}"

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]

