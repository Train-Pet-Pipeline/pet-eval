"""Tests for pet_eval.plugins.gate_tiers and tier integration in apply_gate."""

import pytest

from pet_eval.plugins.gate import apply_gate
from pet_eval.plugins.gate_tiers import TIERS, resolve_tier


def test_tier_known_returns_copy():
    """resolve_tier returns a fresh copy; mutation does not leak to TIERS."""
    strict = resolve_tier("strict")
    assert strict["min_schema_compliance"] == 0.99
    strict["min_schema_compliance"] = 0.0  # mutate copy
    assert TIERS["strict"]["min_schema_compliance"] == 0.99


def test_tier_unknown_raises():
    """Unknown tier name raises ValueError listing valid tiers."""
    with pytest.raises(ValueError, match="unknown gate tier"):
        resolve_tier("super-strict")


def test_apply_gate_with_tier_only():
    """apply_gate(tier=...) loads preset and evaluates against it."""
    metrics = {
        "schema_compliance": 0.99,
        "anomaly_recall": 0.86,
        "mood_spearman": 0.76,
        "narrative_bertscore": 0.81,
        "latency_p95_ms": 3500,
        "kl_divergence": 0.01,
        "overall_accuracy": 0.81,
        "vomit_recall": 0.71,
        "distribution_sum_error": 0.005,
        "anomaly_false_positive": 0.10,
    }
    r = apply_gate(metrics, tier="strict")
    assert r.passed, r.reason


def test_apply_gate_tier_fails_below_strict_bar():
    """Strict tier should fail metrics that pass permissive tier."""
    metrics = {"schema_compliance": 0.90}  # passes permissive (0.85), fails strict (0.99)
    strict_r = apply_gate(metrics, tier="strict")
    permissive_r = apply_gate(metrics, tier="permissive")
    # strict has more keys, will fail on multiple missing ones too; just confirm
    # the schema_compliance check itself differs:
    assert not strict_r.passed
    assert "min_schema_compliance=0.99" in strict_r.reason
    # permissive should at least pass schema_compliance even if other metrics missing
    assert (
        "min_schema_compliance" not in permissive_r.reason
        or "schema_compliance=0.9" not in permissive_r.reason
    )


def test_apply_gate_explicit_overrides_tier():
    """Explicit `thresholds` dict keys override the tier preset values."""
    # strict requires min_schema_compliance=0.99; override down to 0.85
    metrics = {"schema_compliance": 0.86}
    r = apply_gate(
        metrics,
        thresholds={"min_schema_compliance": 0.85},
        tier="strict",
    )
    # schema_compliance check passes via override; other strict keys still fail
    # because metrics dict is missing them — assert override worked by checking
    # the failure reason does NOT include schema_compliance failure.
    assert "schema_compliance=0.86" not in r.reason


def test_apply_gate_no_tier_no_thresholds():
    """No tier and no thresholds → empty gate (vacuous pass)."""
    r = apply_gate({"x": 1.0})
    assert r.passed
    assert r.reason == "all thresholds met"


def test_apply_gate_tier_unknown_raises():
    """apply_gate with unknown tier raises through resolve_tier."""
    with pytest.raises(ValueError, match="unknown gate tier"):
        apply_gate({}, tier="bogus")


def test_all_tier_keys_use_min_or_max_prefix():
    """Every preset key must use min_/max_ prefix or apply_gate silently ignores it."""
    for tier_name, tier_dict in TIERS.items():
        for key in tier_dict:
            assert key.startswith(("min_", "max_")), (
                f"tier {tier_name!r} key {key!r} lacks min_/max_ prefix; "
                "apply_gate would silently ignore it"
            )
