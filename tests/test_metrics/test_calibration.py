"""Tests for calibration ECE metric — TDD, tests written before implementation."""
from __future__ import annotations

import pytest

from pet_eval.metrics.calibration import compute_ece
from pet_eval.metrics.types import MetricResult


# ---------------------------------------------------------------------------
# Test 1: perfectly calibrated model → valid MetricResult, threshold=None, passed=True
# ---------------------------------------------------------------------------


def test_perfectly_calibrated() -> None:
    """A perfectly calibrated model yields a valid MetricResult that always passes."""
    # 10 samples, each with confidence matching their bin's accuracy
    confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Correct exactly as many as the confidence suggests (ideal calibration ≈ 0)
    correct = [False, False, False, False, True, True, True, True, True, True]

    result = compute_ece(confidences, correct)

    assert isinstance(result, MetricResult)
    assert result.name == "calibration_ece"
    assert result.threshold is None
    assert result.passed is True
    assert 0.0 <= result.value <= 1.0


# ---------------------------------------------------------------------------
# Test 2: overconfident model (all max confidence, all wrong) → high ECE (>0.5)
# ---------------------------------------------------------------------------


def test_overconfident() -> None:
    """Model always predicts confidence=0.95 but is never correct → ECE > 0.5."""
    n = 20
    confidences = [0.95] * n
    correct = [False] * n

    result = compute_ece(confidences, correct)

    assert result.value > 0.5, f"Expected ECE > 0.5 for overconfident model, got {result.value}"


# ---------------------------------------------------------------------------
# Test 3: empty inputs → value=0.0
# ---------------------------------------------------------------------------


def test_empty_inputs() -> None:
    """Empty confidence/correct lists produce ECE=0.0."""
    result = compute_ece([], [])

    assert result.name == "calibration_ece"
    assert result.value == pytest.approx(0.0)
    assert result.threshold is None
    assert result.passed is True


# ---------------------------------------------------------------------------
# Test 4: details dict contains "bins" key
# ---------------------------------------------------------------------------


def test_details_has_bins() -> None:
    """MetricResult.details must contain a 'bins' key with per-bin breakdown."""
    confidences = [0.1, 0.5, 0.9]
    correct = [False, True, True]

    result = compute_ece(confidences, correct)

    assert "bins" in result.details
    bins = result.details["bins"]
    assert isinstance(bins, list)
    # Each non-empty bin entry should have the required keys
    for entry in bins:
        assert "range" in entry
        assert "count" in entry
        assert "avg_conf" in entry
        assert "accuracy" in entry
