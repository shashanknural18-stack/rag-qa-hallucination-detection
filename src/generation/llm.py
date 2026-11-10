"""Thin wrapper around llama-cpp-python: loads the GGUF model once."""

from __future__ import annotations

import logging
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
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Download it with:\n"
                "  huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF "
                "mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models"
            )

        try:
            from llama_cpp import Llama  # type: ignore[import-not-found,import-untyped]
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed in the current environment.\n"
                "Install it with:\n"
                "  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
            ) from exc

        logger.info("Loading GGUF model from %s (n_ctx=%d, threads=%d)", model_path, n_ctx, n_threads)
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        self.model_path = str(model_path)

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
        kwargs: dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "echo": False,
        }
        if seed is not None:
            kwargs["seed"] = seed

        output = self._model(prompt, **kwargs)
        text: str = output["choices"][0]["text"]  # type: ignore[index]
        return text.strip()

