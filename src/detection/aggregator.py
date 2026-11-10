"""Turn detector scores into one confidence value and a decision."""

from __future__ import annotations

from src.config import THRESHOLD_SHOW, THRESHOLD_WARN, WEIGHTS
from src.detection.schemas import Decision, DetectionReport


def aggregate(report: DetectionReport) -> tuple[float, Decision]:
    """Combine detector scores with ``WEIGHTS`` and apply the decision thresholds.

    Skipped detectors are left out and the remaining weights are renormalized
    so they sum to 1. If every detector was skipped, confidence is 0 and the
    decision is ``REFUSE``.

    ``confidence >= THRESHOLD_SHOW`` is ``SHOW``, ``>= THRESHOLD_WARN`` is
    ``WARN``, and anything lower is ``REFUSE``. If the NLI detector recorded a
    contradiction probability above 0.8 for any sentence, a ``SHOW`` is capped
    at ``WARN``. The confidence value itself is not changed.

    Parameters
    ----------
    report:
        Sentences and one :class:`DetectorResult` per detector.

    Returns
    -------
    tuple[float, Decision]
        ``(confidence, decision)``. Confidence is in ``[0, 1]``.
    """
    active = [result for result in report.results if not result.skipped and result.name in WEIGHTS]
    if not active:
        return 0.0, Decision.REFUSE

    weight_sum = sum(WEIGHTS[result.name] for result in active)
    if weight_sum <= 0:
        return 0.0, Decision.REFUSE

    confidence = sum(WEIGHTS[result.name] * result.score for result in active) / weight_sum
    confidence = max(0.0, min(1.0, float(confidence)))

    # A 1e-9 cushion keeps a score of 0.75 on the SHOW side of the cut
    # after dividing binary weights that are not exact.
    if confidence + 1e-9 >= THRESHOLD_SHOW:
        decision = Decision.SHOW
    elif confidence + 1e-9 >= THRESHOLD_WARN:
        decision = Decision.WARN
    else:
        decision = Decision.REFUSE

    nli = next((result for result in report.results if result.name == "nli" and not result.skipped), None)
    if nli is not None and decision is Decision.SHOW:
        contradictions = nli.details.get("contradiction_probs", [])
        if any(float(prob) > 0.8 for prob in contradictions):
            decision = Decision.WARN

    return confidence, decision
