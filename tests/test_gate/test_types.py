"""Tests for GateResult data type."""

from __future__ import annotations

from pet_eval.gate.types import GateResult
from pet_eval.plugins.metrics.types import MetricResult


def _make_metric(name: str, value: float, threshold: float | None, passed: bool) -> MetricResult:
    """Helper: build a MetricResult directly for gate tests."""
    operator = "gte"
    if threshold is not None:
        return MetricResult.create(name, value=value, threshold=threshold, operator=operator)
    return MetricResult.create(name, value=value, threshold=None)


def test_all_pass() -> None:
    """Two passing gated metrics → gate passes."""
    results = [
        MetricResult.create("accuracy", value=0.95, threshold=0.90),
        MetricResult.create("recall", value=0.88, threshold=0.85),
    ]
    gate = GateResult.from_results(results, skipped=[])
    assert gate.passed is True
    assert "PASS" in gate.summary


def test_one_fail() -> None:
    """One passing + one failing gated metric → gate fails, summary has fail name."""
    results = [
        MetricResult.create("accuracy", value=0.95, threshold=0.90),
        MetricResult.create("recall", value=0.70, threshold=0.85),
    ]
    gate = GateResult.from_results(results, skipped=[])
    assert gate.passed is False
    assert "FAIL" in gate.summary
    assert "recall" in gate.summary


def test_informational_ignored() -> None:
    """Gated metric passes + informational metric with huge value → gate still passes."""
    results = [
        MetricResult.create("accuracy", value=0.95, threshold=0.90),
        MetricResult.create("info_only", value=9999.0, threshold=None),
    ]
    gate = GateResult.from_results(results, skipped=[])
    assert gate.passed is True


def test_skipped_recorded() -> None:
    """Skipped list is preserved in GateResult."""
    results = [
        MetricResult.create("accuracy", value=0.95, threshold=0.90),
    ]
    skipped = ["latency_p95_ms", "kl_divergence"]
    gate = GateResult.from_results(results, skipped=skipped)
    assert gate.skipped == skipped


def test_empty_results_passes() -> None:
    """Empty results list with skipped items → gate passes (no gated metrics to fail)."""
    gate = GateResult.from_results([], skipped=["latency_p95_ms"])
    assert gate.passed is True
    assert "PASS" in gate.summary
