"""Thin wrapper around llama-cpp-python: loads the GGUF model once."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.config import LLM_MAX_TOKENS, LLM_MODEL_PATH, LLM_N_CTX, LLM_N_THREADS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


class LocalLLM:
    """CPU-based LLM backed by a quantised GGUF model via ``llama-cpp-python``.

    The model is loaded **once** at construction time and reused for every
    ``complete()`` call.

    Parameters
    ----------
    model_path:
        Path to the ``.gguf`` file.  Defaults to :pydata:`LLM_MODEL_PATH`.
    n_ctx:
        Context window size in tokens.
    n_threads:
        Number of CPU threads for inference.
    """

    def __init__(
        self,
        model_path: str | Path = LLM_MODEL_PATH,
        n_ctx: int = LLM_N_CTX,
        n_threads: int = LLM_N_THREADS,
    ) -> None:
        self.model_path = str(model_path)
        self._is_mock = os.environ.get("USE_MOCK_LLM", "").lower() in ("1", "true", "yes")

        if self._is_mock:
            logger.warning("Running LocalLLM in MOCK mode (USE_MOCK_LLM=1)")
            return

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Download it with either:\n"
                "  .\\venv\\Scripts\\hf.exe download TheBloke/Mistral-7B-Instruct-v0.2-GGUF "
                "mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models\n"
                "or via python:\n"
                "  .\\venv\\Scripts\\python.exe -c \"from huggingface_hub import hf_hub_download; "
                "hf_hub_download('TheBloke/Mistral-7B-Instruct-v0.2-GGUF', 'mistral-7b-instruct-v0.2.Q4_K_M.gguf', local_dir='models')\"\n\n"
                "Or run in mock mode for instant UI testing:\n"
                "  $env:USE_MOCK_LLM=\"1\"; streamlit run app.py"
            )

        try:
            from llama_cpp import Llama  # type: ignore[import-not-found,import-untyped]
        except ImportError as exc:
            if os.environ.get("USE_MOCK_LLM", "").lower() in ("1", "true", "yes", "fallback"):
                logger.warning("llama-cpp-python is not installed. Operating in fallback mock mode.")
                self._is_mock = True
                return
            raise ImportError(
                "llama-cpp-python is not installed in the current environment.\n"
                "Install it with:\n"
                "  .\\venv\\Scripts\\pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n\n"
                "Or run in mock mode for instant UI testing:\n"
                "  $env:USE_MOCK_LLM=\"1\"; .\\venv\\Scripts\\streamlit.exe run app.py"
            ) from exc

        logger.info("Loading GGUF model from %s (n_ctx=%d, threads=%d)", model_path, n_ctx, n_threads)
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=256,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        self.model_path = str(model_path)

    @staticmethod
    def _clean_prompt_for_model(prompt: str) -> str:
        """Strip explicit <s> prefix because llama-cpp-python automatically prepends BOS token."""
        cleaned = prompt.strip()
        if cleaned.startswith("<s>"):
            cleaned = cleaned[3:].lstrip()
        return cleaned

    def stream(
        self,
        prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        seed: int | None = None,
    ):
        """Stream tokens for *prompt* as they are generated.

        Yields
        ------
        str
            Token chunks in generation order. Joined tokens match complete().
        """
        if getattr(self, "_is_mock", False):
            text = self.complete(prompt, temperature=temperature, max_tokens=max_tokens, seed=seed)
            # Split preserving spaces so "".join(tokens) == text
            import re
            tokens = re.findall(r"\S+|\s+", text)
            for token in tokens:
                yield token
            return

        kwargs: dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "echo": False,
        }
        if seed is not None:
            kwargs["seed"] = seed

        prompt_to_send = self._clean_prompt_for_model(prompt)
        for chunk in self._model(prompt_to_send, **kwargs):
            text_chunk = chunk["choices"][0]["text"]
            yield text_chunk

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        """Create completion using native chat template if supported, else fallback."""
        if getattr(self, "_is_mock", False):
            last_msg = messages[-1]["content"] if messages else ""
            return self.complete(last_msg)

        if hasattr(self, "_model") and hasattr(self._model, "create_chat_completion"):
            try:
                res = self._model.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return res["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                logger.warning("create_chat_completion failed (%s); using prompt fallback", exc)

        flat_prompt = "\n".join(m["content"] for m in messages)
        return self.complete(flat_prompt, temperature=temperature, max_tokens=max_tokens)

    def complete(
        self,
        prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        seed: int | None = None,
    ) -> str:
        """Generate a completion for *prompt* and return the text.

        Parameters
        ----------
        prompt:
            Full ``[INST]``-wrapped prompt.
        temperature:
            Sampling temperature (0 = greedy).
        max_tokens:
            Maximum tokens to generate.
        seed:
            Random seed for reproducibility.  ``None`` lets the backend choose.

        Returns
        -------
        str
            The generated text, stripped of leading/trailing whitespace.
        """
        if getattr(self, "_is_mock", False):
            # Check if this is the factuality judge prompt requesting JSON
            if "factuality judge" in prompt.lower() or "claims" in prompt.lower():
                return '{"claims": [{"claim": "The response is grounded in the retrieved documents.", "verdict": "supported"}]}'

            import re
            from src.config import NO_ANSWER_TEXT

            context_match = re.search(r"Context:\s*(.*?)\s*Question:\s*(.*)", prompt, re.DOTALL)
            if context_match:
                context_str = context_match.group(1).strip()
                question_str = context_match.group(2).replace("[/INST]", "").strip().lower()

                # Parse chunk blocks: [1] (source: ...)\n<text>
                chunk_blocks = re.findall(
                    r"\[(\d+)\]\s*(?:\(source:[^\)]*\))?\s*\n(.*?)(?=(?:\[\d+\]|\Z))",
                    context_str,
                    re.DOTALL,
                )

                q_words = set(re.findall(r"\w{3,}", question_str))
                scored_sentences: list[tuple[int, int, str]] = []

                for cid_str, ctext in chunk_blocks:
                    cid = int(cid_str)
                    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", ctext) if len(s.strip()) > 10]
                    for s in sents:
                        s_lower = s.lower()
                        s_words = set(re.findall(r"\w{3,}", s_lower))
                        overlap = len(q_words & s_words)
                        if overlap > 0:
                            scored_sentences.append((overlap, cid, s))

                if scored_sentences:
                    scored_sentences.sort(key=lambda x: x[0], reverse=True)
                    chosen = scored_sentences[:2]
                    parts = []
                    for _, cid, text in chosen:
                        clean_sent = text.rstrip(".")
                        parts.append(f"{clean_sent} [{cid}].")
                    return " ".join(parts)
                else:
                    return NO_ANSWER_TEXT

            return (
                "Based on the provided documents [1], returns must be acknowledged within 48 hours "
                "and resolved within 30 days [1]. Consent to share personal data must be free and informed [2]."
            )

        kwargs: dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "echo": False,
        }
        if seed is not None:
            kwargs["seed"] = seed

        prompt_to_send = self._clean_prompt_for_model(prompt)
        output = self._model(prompt_to_send, **kwargs)
        text: str = output["choices"][0]["text"]  # type: ignore[index]
        return text.strip()

