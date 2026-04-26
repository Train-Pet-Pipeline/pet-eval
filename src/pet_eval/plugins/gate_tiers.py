"""Preset gate threshold tiers (handoff task #30).

Recipe `gate_tier: <name>` lets eval recipes pick a complete threshold preset
without hand-rolling every key. Explicit `thresholds:` in the same config
override individual tier keys (preset is the floor, override is the choice).

Supported tier names:
  - ``strict``: production-quality bar; matches ``params.yaml gates.vlm`` defaults
  - ``balanced``: relaxed for mid-training checkpoints / smaller datasets
  - ``permissive``: smoke-test bar; only catches gross regressions

Tier values cover standard VLM + audio metrics. Recipes whose metrics are
outside the standard set must still pass them via explicit ``thresholds:``.
"""

from __future__ import annotations

# Tier presets. Keys follow the ``min_<metric>`` / ``max_<metric>`` convention
# consumed by :func:`pet_eval.plugins.gate.apply_gate`.
TIERS: dict[str, dict[str, float]] = {
    "strict": {
        "min_schema_compliance": 0.99,
        "min_distribution_sum_error": 0.0,
        "max_distribution_sum_error": 0.01,
        "min_anomaly_recall": 0.85,
        "max_anomaly_false_positive": 0.15,
        "min_mood_spearman": 0.75,
        "min_narrative_bertscore": 0.80,
        "max_latency_p95_ms": 4000,
        "max_kl_divergence": 0.02,
        "min_overall_accuracy": 0.80,
        "min_vomit_recall": 0.70,
    },
    "balanced": {
        "min_schema_compliance": 0.95,
        "max_distribution_sum_error": 0.05,
        "min_anomaly_recall": 0.70,
        "max_anomaly_false_positive": 0.25,
        "min_mood_spearman": 0.60,
        "min_narrative_bertscore": 0.70,
        "max_latency_p95_ms": 6000,
        "max_kl_divergence": 0.05,
        "min_overall_accuracy": 0.65,
        "min_vomit_recall": 0.55,
    },
    "permissive": {
        "min_schema_compliance": 0.85,
        "max_distribution_sum_error": 0.15,
        "min_anomaly_recall": 0.50,
        "max_anomaly_false_positive": 0.40,
        "min_mood_spearman": 0.40,
        "min_narrative_bertscore": 0.55,
        "max_latency_p95_ms": 10000,
        "max_kl_divergence": 0.15,
        "min_overall_accuracy": 0.50,
        "min_vomit_recall": 0.40,
    },
}


def resolve_tier(name: str) -> dict[str, float]:
    """Return the threshold preset for the named tier.

    Args:
        name: Tier name; one of ``strict`` / ``balanced`` / ``permissive``.

    Returns:
        A new copy of the tier's threshold dict (mutating the return value
        does not affect other callers).

    Raises:
        ValueError: If ``name`` is not a known tier.
    """
    if name not in TIERS:
        raise ValueError(
            f"unknown gate tier {name!r}; valid: {sorted(TIERS)}"
        )
    return dict(TIERS[name])
