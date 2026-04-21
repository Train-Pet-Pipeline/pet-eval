"""Tests for pet_eval.plugins.gate."""

from pet_eval.plugins.gate import apply_gate


def test_gate_all_pass():
    """Metric above min_ threshold should pass."""
    r = apply_gate({"narrative_quality": 4.0}, {"min_narrative_quality": 3.0})
    assert r.passed
    assert r.reason == "all thresholds met"


def test_gate_min_fails():
    """Metric below min_ threshold should fail with informative reason."""
    r = apply_gate({"narrative_quality": 2.5}, {"min_narrative_quality": 3.0})
    assert not r.passed
    assert "narrative_quality=2.5" in r.reason
    assert "min_narrative_quality=3.0" in r.reason


def test_gate_max_fails():
    """Metric above max_ threshold should fail."""
    r = apply_gate({"calibration_ece": 0.15}, {"max_calibration_ece": 0.10})
    assert not r.passed
    assert "calibration_ece=0.15" in r.reason


def test_gate_missing_min_metric_fails_conservative():
    """Missing metric should default to 0 for min_ check (conservative fail)."""
    r = apply_gate({}, {"min_schema_compliance": 0.9})
    assert not r.passed


def test_gate_missing_max_metric_fails_conservative():
    """Missing metric should default to +inf for max_ check (conservative fail)."""
    r = apply_gate({}, {"max_kl": 0.1})
    assert not r.passed


def test_gate_ignores_unprefixed_keys():
    """Keys without min_/max_ prefix are silently ignored."""
    r = apply_gate({"x": 1.0}, {"description": 42.0})
    assert r.passed  # 'description' is neither min_ nor max_, silently ignored


def test_gate_multiple_failures_concatenated():
    """Multiple failures should all appear in the reason string."""
    r = apply_gate(
        {"a": 1.0, "b": 5.0},
        {"min_a": 2.0, "max_b": 3.0},
    )
    assert not r.passed
    assert "a=1.0" in r.reason
    assert "b=5.0" in r.reason
    assert ";" in r.reason
