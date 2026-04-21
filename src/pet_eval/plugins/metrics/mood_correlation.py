"""Mood correlation metric using Spearman rank correlation for pet-eval.

Computes the mean Spearman rank correlation across three mood dimensions
(alertness, anxiety, engagement) between model predictions and teacher labels.
"""

from __future__ import annotations

import logging
import math

from pet_infra.registry import METRICS
from scipy.stats import spearmanr

from pet_eval.plugins.metrics.types import MetricResult

logger = logging.getLogger(__name__)

MOOD_DIMENSIONS: list[str] = ["alertness", "anxiety", "engagement"]


def compute_mood_correlation(
    model_moods: list[dict[str, float]],
    teacher_moods: list[dict[str, float]],
    *,
    threshold: float | None = None,
) -> MetricResult:
    """Compute mean Spearman correlation across mood dimensions.

    For each of the three mood dimensions (alertness, anxiety, engagement),
    extracts the per-sample values from model and teacher dicts, computes
    scipy.stats.spearmanr, then averages across dimensions.

    Args:
        model_moods: List of dicts mapping dimension name → predicted score.
        teacher_moods: List of dicts mapping dimension name → teacher score.
        threshold: Optional gate threshold for the ``"gte"`` operator.
            Pass ``None`` for an informational-only metric.

    Returns:
        A single :class:`MetricResult` with ``name="mood_spearman"`` and
        ``operator="gte"``. ``details["per_dimension"]`` contains the
        individual Spearman correlation for each dimension.
    """
    n = len(model_moods)

    if n < 2:
        logger.warning(
            "compute_mood_correlation: need >=2 samples for Spearman, got %d; returning 0.0",
            n,
        )
        per_dimension = {dim: 0.0 for dim in MOOD_DIMENSIONS}
        return MetricResult.create(
            "mood_spearman",
            value=0.0,
            threshold=threshold,
            operator="gte",
            details={"per_dimension": per_dimension},
        )

    per_dimension = {}
    for dim in MOOD_DIMENSIONS:
        model_vals = [entry[dim] for entry in model_moods]
        teacher_vals = [entry[dim] for entry in teacher_moods]

        result = spearmanr(model_vals, teacher_vals)
        corr = float(result.statistic)

        if math.isnan(corr):
            # Constant arrays (zero variance) → treat as 0.0
            corr = 0.0

        per_dimension[dim] = corr

    mean_corr = sum(per_dimension.values()) / len(MOOD_DIMENSIONS)

    logger.info(
        "compute_mood_correlation",
        extra={"per_dimension": per_dimension, "mean_corr": mean_corr},
    )

    return MetricResult.create(
        "mood_spearman",
        value=mean_corr,
        threshold=threshold,
        operator="gte",
        details={"per_dimension": per_dimension},
    )


# ---- Registry adapter (P2-B) ----


@METRICS.register_module(name="mood_correlation")
class MoodCorrelationMetric:
    """Registry adapter wrapping compute_mood_correlation."""

    def __init__(self, **kwargs) -> None:
        """Store kwargs to forward to compute_mood_correlation."""
        self._kwargs = kwargs

    def __call__(self, *args, **call_kwargs) -> MetricResult:
        """Delegate to compute_mood_correlation with merged kwargs."""
        return compute_mood_correlation(*args, **{**self._kwargs, **call_kwargs})
