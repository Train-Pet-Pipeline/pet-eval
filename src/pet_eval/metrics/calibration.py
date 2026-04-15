"""Calibration metric (Expected Calibration Error) for pet-eval.

Computes ECE as an informational-only MetricResult — never gated,
always passes.  ECE measures alignment between model confidence and
empirical accuracy across equal-width confidence bins.
"""
from __future__ import annotations

import logging

from pet_eval.metrics.types import MetricResult

logger = logging.getLogger(__name__)


def compute_ece(
    confidences: list[float],
    correct: list[bool],
    *,
    n_bins: int = 10,
) -> MetricResult:
    """Compute Expected Calibration Error (ECE).

    Divides the [0, 1] confidence range into ``n_bins`` equal-width bins.
    For each bin the average confidence and empirical accuracy are computed
    for all samples whose confidence falls in that bin.  ECE is the
    sample-count-weighted mean of |accuracy - avg_conf| across bins.

    This metric is informational only (threshold=None) and always passes.
    Empty inputs return value=0.0.

    Args:
        confidences: Predicted confidence scores, each in [0, 1].
        correct: Boolean correctness label for each prediction.
        n_bins: Number of equal-width bins to partition [0, 1].

    Returns:
        A single :class:`MetricResult` with name ``"calibration_ece"``,
        ``threshold=None``, ``passed=True``, and a ``"bins"`` list in
        ``details``.
    """
    if not confidences:
        logger.info("compute_ece: empty inputs, returning ECE=0.0")
        return MetricResult.create(
            "calibration_ece",
            value=0.0,
            threshold=None,
            details={"bins": []},
        )

    bin_width = 1.0 / n_bins

    # Accumulate per-bin stats
    bin_confs: list[list[float]] = [[] for _ in range(n_bins)]
    bin_correct: list[list[bool]] = [[] for _ in range(n_bins)]

    for conf, is_correct in zip(confidences, correct):
        # Clamp to last bin for conf == 1.0
        idx = min(int(conf / bin_width), n_bins - 1)
        bin_confs[idx].append(conf)
        bin_correct[idx].append(is_correct)

    total = len(confidences)
    ece = 0.0
    bins_details: list[dict] = []

    for i in range(n_bins):
        lo = i * bin_width
        hi = (i + 1) * bin_width
        count = len(bin_confs[i])

        if count == 0:
            bins_details.append(
                {
                    "range": [lo, hi],
                    "count": 0,
                    "avg_conf": None,
                    "accuracy": None,
                }
            )
            continue

        avg_conf = sum(bin_confs[i]) / count
        accuracy = sum(bin_correct[i]) / count
        weight = count / total
        ece += weight * abs(accuracy - avg_conf)

        bins_details.append(
            {
                "range": [lo, hi],
                "count": count,
                "avg_conf": avg_conf,
                "accuracy": accuracy,
            }
        )

    logger.info(
        "compute_ece",
        extra={"n_samples": total, "n_bins": n_bins, "ece": ece},
    )

    return MetricResult.create(
        "calibration_ece",
        value=ece,
        threshold=None,
        details={"bins": bins_details},
    )
