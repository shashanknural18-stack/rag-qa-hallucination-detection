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
        if getattr(self, "_is_mock", False):
            # Check if this is the factuality judge prompt requesting JSON
            if "factuality judge" in prompt.lower():
                return '{"claims": [{"claim": "The response is grounded in the retrieved documents.", "verdict": "supported"}]}'
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

        output = self._model(prompt, **kwargs)
        text: str = output["choices"][0]["text"]  # type: ignore[index]
        return text.strip()

