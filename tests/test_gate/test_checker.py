"""Tests for check_gate() in pet_eval.gate.checker."""
from __future__ import annotations

from pet_eval.gate.checker import check_gate
from pet_eval.metrics.types import MetricResult


def test_vlm_all_pass(sample_params: dict) -> None:
    """Two passing VLM MetricResults → gate.passed is True."""
    results = [
        MetricResult.create("schema_compliance", value=1.0, threshold=0.99, operator="gte"),
        MetricResult.create("anomaly_recall", value=0.90, threshold=0.85, operator="gte"),
    ]
    gate = check_gate(results, skipped=[], gate_type="vlm", params=sample_params)
    assert gate.passed is True
    assert "PASS" in gate.summary


def test_vlm_one_fail(sample_params: dict) -> None:
    """One passing + one failing MetricResult → gate.passed is False, failed name in summary."""
    results = [
        MetricResult.create("anomaly_recall", value=0.90, threshold=0.85, operator="gte"),
        MetricResult.create("schema_compliance", value=0.95, threshold=0.99, operator="gte"),
    ]
    gate = check_gate(results, skipped=[], gate_type="vlm", params=sample_params)
    assert gate.passed is False
    assert "schema_compliance" in gate.summary


def test_skipped_not_fail(sample_params: dict) -> None:
    """One passing metric + two skipped → gate.passed is True, skipped list preserved."""
    results = [
        MetricResult.create("anomaly_recall", value=0.90, threshold=0.85, operator="gte"),
    ]
    skipped = ["latency_p95_ms", "kl_divergence"]
    gate = check_gate(results, skipped=skipped, gate_type="vlm", params=sample_params)
    assert gate.passed is True
    assert len(gate.skipped) == 2
    assert gate.skipped == skipped


def test_audio_gate(sample_params: dict) -> None:
    """Two passing audio MetricResults → gate.passed is True."""
    results = [
        MetricResult.create("overall_accuracy", value=0.85, threshold=0.80, operator="gte"),
        MetricResult.create("vomit_recall", value=0.75, threshold=0.70, operator="gte"),
    ]
    gate = check_gate(results, skipped=[], gate_type="audio", params=sample_params)
    assert gate.passed is True
    assert "PASS" in gate.summary
