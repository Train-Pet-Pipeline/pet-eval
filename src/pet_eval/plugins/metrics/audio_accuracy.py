"""Audio classification accuracy metrics for pet-eval.

Computes two MetricResults from multi-class audio predictions:
- ``audio_overall_accuracy``: correct / total, gated with ``"gte"`` operator.
- ``audio_vomit_recall``: per-class recall for the "vomiting" class, gated with ``"gte"`` operator.
"""

from __future__ import annotations

import logging
from collections import Counter

from pet_infra.registry import METRICS

from pet_eval.plugins.metrics.types import MetricResult

logger = logging.getLogger(__name__)


def compute_audio_accuracy(
    predicted: list[str],
    actual: list[str],
    classes: list[str],
    *,
    accuracy_threshold: float | None = None,
    vomit_recall_threshold: float | None = None,
) -> list[MetricResult]:
    """Compute audio classification accuracy and vomit recall.

    Given parallel lists of predicted and ground-truth string class labels,
    computes:

    - **audio_overall_accuracy** = correct_count / len(predicted). Returns 0.0
      on empty input.
    - **audio_vomit_recall** = recall of the ``"vomiting"`` class (TP / (TP + FN)).
      Returns 0.0 when the vomiting class is absent from ``actual`` or the
      input is empty.

    Per-class precision, recall, and F1 are built from a confusion matrix
    ``{actual_class: Counter(predicted_class)}``.

    Args:
        predicted: Model-predicted audio class labels.
        actual: Ground-truth audio class labels.
        classes: Full list of class names used by the model.
        accuracy_threshold: Optional gate threshold for overall accuracy
            (``"gte"``). Pass ``None`` for an informational-only metric.
        vomit_recall_threshold: Optional gate threshold for vomit recall
            (``"gte"``). Pass ``None`` for an informational-only metric.

    Returns:
        A list of two :class:`MetricResult` instances:
        ``[audio_overall_accuracy, audio_vomit_recall]``.
    """
    n_samples = len(predicted)

    if n_samples == 0:
        logger.info("compute_audio_accuracy called with empty input")
        return [
            MetricResult.create(
                "audio_overall_accuracy",
                value=0.0,
                threshold=accuracy_threshold,
                operator="gte",
                details={"per_class": {}, "confusion_matrix": {}, "n_samples": 0},
            ),
            MetricResult.create(
                "audio_vomit_recall",
                value=0.0,
                threshold=vomit_recall_threshold,
                operator="gte",
            ),
        ]

    # Build confusion matrix: {actual_class: Counter({predicted_class: count})}
    confusion_matrix: dict[str, Counter[str]] = {}
    correct_count = 0
    for pred, gt in zip(predicted, actual):
        if gt not in confusion_matrix:
            confusion_matrix[gt] = Counter()
        confusion_matrix[gt][pred] += 1
        if pred == gt:
            correct_count += 1

    accuracy = correct_count / n_samples

    # Per-class precision, recall, F1
    per_class: dict[str, dict[str, float]] = {}
    for cls in classes:
        # True positives: predicted cls when actual is cls
        tp = confusion_matrix.get(cls, Counter())[cls]
        # False negatives: actual is cls but predicted something else
        fn = sum(confusion_matrix.get(cls, Counter()).values()) - tp
        # False positives: predicted cls when actual is NOT cls
        fp = sum(
            confusion_matrix.get(other_cls, Counter())[cls]
            for other_cls in confusion_matrix
            if other_cls != cls
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}

    # Vomit recall — check "vomiting" first (canonical), then "vomit" (alias); fall back to 0.0
    vomit_recall = per_class.get("vomiting", per_class.get("vomit", {})).get("recall", 0.0)

    # Serialise confusion matrix to plain dicts for JSON-safe details
    cm_serialisable = {k: dict(v) for k, v in confusion_matrix.items()}

    logger.info(
        "compute_audio_accuracy",
        extra={
            "n_samples": n_samples,
            "accuracy": accuracy,
            "vomit_recall": vomit_recall,
        },
    )

    acc_result = MetricResult.create(
        "audio_overall_accuracy",
        value=accuracy,
        threshold=accuracy_threshold,
        operator="gte",
        details={
            "per_class": per_class,
            "confusion_matrix": cm_serialisable,
            "n_samples": n_samples,
        },
    )

    vomit_result = MetricResult.create(
        "audio_vomit_recall",
        value=vomit_recall,
        threshold=vomit_recall_threshold,
        operator="gte",
    )

    return [acc_result, vomit_result]


# ---- Registry adapter (P2-B) ----


@METRICS.register_module(name="audio_accuracy")
class AudioAccuracyMetric:
    """Registry adapter wrapping compute_audio_accuracy."""

    def __init__(self, **kwargs) -> None:
        """Store kwargs to forward to compute_audio_accuracy."""
        self._kwargs = kwargs

    def __call__(self, *args, **call_kwargs) -> list[MetricResult]:
        """Delegate to compute_audio_accuracy with merged kwargs."""
        return compute_audio_accuracy(*args, **{**self._kwargs, **call_kwargs})
