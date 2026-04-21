"""Anomaly recall and false positive rate metrics for pet-eval.

Computes two MetricResults from binary anomaly predictions:
- ``anomaly_recall``: TP / (TP + FN), gated with ``"gte"`` operator.
- ``anomaly_false_positive``: FP / (FP + TN), gated with ``"lte"`` operator.
"""

from __future__ import annotations

import logging

from pet_eval.metrics.types import MetricResult

logger = logging.getLogger(__name__)


def compute_anomaly_recall(
    predicted: list[bool],
    actual: list[bool],
    *,
    recall_threshold: float | None = None,
    fpr_threshold: float | None = None,
) -> list[MetricResult]:
    """Compute anomaly recall and false positive rate.

    Given parallel lists of predicted and ground-truth boolean labels,
    computes:

    - **anomaly_recall** = TP / (TP + FN). Returns 0.0 when there are no
      positive labels in ``actual``.
    - **anomaly_false_positive** = FP / (FP + TN). Returns 0.0 when there
      are no negative labels in ``actual``.

    Args:
        predicted: Model-predicted anomaly flags (True = anomaly detected).
        actual: Ground-truth anomaly flags (True = anomaly present).
        recall_threshold: Optional gate threshold for recall (``"gte"``).
            Pass ``None`` for an informational-only metric.
        fpr_threshold: Optional gate threshold for false positive rate
            (``"lte"``). Pass ``None`` for an informational-only metric.

    Returns:
        A list of two :class:`MetricResult` instances:
        ``[anomaly_recall, anomaly_false_positive]``.
    """
    tp = sum(p and a for p, a in zip(predicted, actual))
    fn = sum(not p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    tn = sum(not p and not a for p, a in zip(predicted, actual))

    total_positive = tp + fn
    total_negative = fp + tn

    recall = tp / total_positive if total_positive > 0 else 0.0
    fpr = fp / total_negative if total_negative > 0 else 0.0

    logger.info(
        "compute_anomaly_recall",
        extra={
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "recall": recall,
            "fpr": fpr,
        },
    )

    recall_result = MetricResult.create(
        "anomaly_recall",
        value=recall,
        threshold=recall_threshold,
        operator="gte",
        details={
            "tp": tp,
            "fn": fn,
            "total_positive": total_positive,
        },
    )

    fpr_result = MetricResult.create(
        "anomaly_false_positive",
        value=fpr,
        threshold=fpr_threshold,
        operator="lte",
        details={
            "fp": fp,
            "tn": tn,
            "total_negative": total_negative,
        },
    )

    return [recall_result, fpr_result]
