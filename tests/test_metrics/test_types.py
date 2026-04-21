"""Tests for MetricResult data type."""

from __future__ import annotations

from pet_eval.metrics.types import MetricResult


def test_gated_metric_pass() -> None:
    """value >= threshold with gte operator → passed=True."""
    result = MetricResult.create("accuracy", value=0.95, threshold=0.90, operator="gte")
    assert result.name == "accuracy"
    assert result.value == 0.95
    assert result.threshold == 0.90
    assert result.passed is True


def test_gated_metric_fail() -> None:
    """value < threshold with gte operator → passed=False."""
    result = MetricResult.create("accuracy", value=0.80, threshold=0.90, operator="gte")
    assert result.passed is False


def test_gated_metric_lte_pass() -> None:
    """value <= threshold with lte operator → passed=True."""
    result = MetricResult.create("kl_div", value=0.01, threshold=0.02, operator="lte")
    assert result.passed is True


def test_gated_metric_lte_fail() -> None:
    """value > threshold with lte operator → passed=False."""
    result = MetricResult.create("kl_div", value=0.03, threshold=0.02, operator="lte")
    assert result.passed is False


def test_informational_metric_always_passes() -> None:
    """threshold=None → informational metric, always passed=True."""
    result = MetricResult.create("info_metric", value=999.0, threshold=None)
    assert result.threshold is None
    assert result.passed is True


def test_details_default_empty() -> None:
    """details field defaults to empty dict when not provided."""
    result = MetricResult.create("accuracy", value=0.95, threshold=0.90)
    assert result.details == {}


def test_details_provided() -> None:
    """details passed through to the result."""
    extra = {"n_samples": 100, "model": "qwen2-vl"}
    result = MetricResult.create("accuracy", value=0.95, threshold=0.90, details=extra)
    assert result.details == extra
