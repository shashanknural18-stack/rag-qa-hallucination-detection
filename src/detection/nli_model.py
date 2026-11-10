"""Shared NLI cross-encoder. Loaded once and reused by NLI and SelfCheck."""

from __future__ import annotations

import logging

from src.config import NLI_MODEL

logger = logging.getLogger(__name__)

_LABELS = ("entailment", "neutral", "contradiction")


class NLIModel:
    """CPU NLI model that returns entailment, neutral, and contradiction probabilities.

    The model is loaded once at construction time from the local Hugging Face
    cache. Label indices come from ``model.config.id2label``.

    Parameters
    ----------
    model_name:
        Hugging Face model id. Defaults to :pydata:`NLI_MODEL`.
    """

    def __init__(self, model_name: str = NLI_MODEL) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=True,
            )
        except OSError as exc:
            raise FileNotFoundError(
                f"NLI model '{model_name}' is not in the local cache.\n"
                "Download it once with:\n"
                "  python -c \"from transformers import AutoTokenizer, "
                "AutoModelForSequenceClassification as M; "
                f"n='{model_name}'; AutoTokenizer.from_pretrained(n); M.from_pretrained(n)\""
            ) from exc

        self.model.to("cpu")
        self.model.eval()
        self._torch = torch
        self._label_index = self._read_label_index()
        logger.info("Loaded NLI model %s", model_name)

    def _read_label_index(self) -> dict[str, int]:
        """Map entailment / neutral / contradiction onto logit indices."""
        id2label = getattr(self.model.config, "id2label", {}) or {}
        mapping: dict[str, int] = {}
        for idx, label in id2label.items():
            name = str(label).lower()
            for key in _LABELS:
                if key in name:
                    mapping[key] = int(idx)
        missing = [key for key in _LABELS if key not in mapping]
        if missing:
            raise RuntimeError(
                f"NLI model '{self.model_name}' labels {id2label} are missing {missing}."
            )
        return mapping

    def _max_length(self) -> int:
        """Return a usable token budget. Some tokenizers report a huge max length."""
        max_len = getattr(self.tokenizer, "model_max_length", 512)
        if not isinstance(max_len, int) or max_len > 4096 or max_len < 8:
            return 512
        return max_len

    def _truncate_premise(self, premise: str, hypothesis: str) -> str:
        """Shorten *premise* so the pair fits in the model window."""
        max_len = self._max_length()
        hyp_ids = self.tokenizer.encode(hypothesis, add_special_tokens=False)
        budget = max_len - len(hyp_ids) - 3
        if budget < 8:
            budget = 8
        prem_ids = self.tokenizer.encode(premise, add_special_tokens=False)[:budget]
        return self.tokenizer.decode(prem_ids, skip_special_tokens=True)

    def entailment_probs(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Return softmax probabilities for one premise/hypothesis pair.

        Parameters
        ----------
        premise:
            The evidence text (a chunk or a sampled answer).
        hypothesis:
            The claim being checked (one answer sentence).

        Returns
        -------
        dict[str, float]
            Keys ``entailment``, ``neutral``, and ``contradiction``.
        """
        return self.entailment_probs_batch([(premise, hypothesis)])[0]

    def entailment_probs_batch(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[dict[str, float]]:
        """Score many ``(premise, hypothesis)`` pairs in one forward pass.

        Parameters
        ----------
        pairs:
            Premise/hypothesis pairs. Premises are truncated to the model window.

        Returns
        -------
        list[dict[str, float]]
            One probability dict per pair, in the same order.
        """
        if not pairs:
            return []

        premises: list[str] = []
        hypotheses: list[str] = []
        for premise, hypothesis in pairs:
            premises.append(self._truncate_premise(premise, hypothesis))
            hypotheses.append(hypothesis)

        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self._max_length(),
            return_tensors="pt",
        )
        torch = self._torch
        with torch.no_grad():
            logits = self.model(**inputs).logits
            matrix = torch.softmax(logits, dim=-1).cpu().tolist()

        results: list[dict[str, float]] = []
        for row in matrix:
            item = {key: 0.0 for key in _LABELS}
            for key, idx in self._label_index.items():
                if 0 <= idx < len(row):
                    item[key] = float(row[idx])
            results.append(item)
        return results
