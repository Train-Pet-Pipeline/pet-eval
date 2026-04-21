"""Tests for anomaly_recall metric — TDD, tests written before implementation."""

from __future__ import annotations

import pytest

from pet_eval.plugins.metrics.anomaly_recall import compute_anomaly_recall
from pet_eval.plugins.metrics.types import MetricResult

# ---------------------------------------------------------------------------
# Test 1: perfect recall — all predictions correct → recall=1.0, fpr=0.0
# ---------------------------------------------------------------------------


def test_perfect_recall() -> None:
    """All predictions match ground truth → recall=1.0, fpr=0.0."""
    predicted = [True, False, False, True]
    actual = [True, False, False, True]

    results = compute_anomaly_recall(predicted, actual)

    assert len(results) == 2
    recall_result, fpr_result = results

    assert isinstance(recall_result, MetricResult)
    assert recall_result.name == "anomaly_recall"
    assert recall_result.value == pytest.approx(1.0)
    assert recall_result.passed is True

    assert isinstance(fpr_result, MetricResult)
    assert fpr_result.name == "anomaly_false_positive"
    assert fpr_result.value == pytest.approx(0.0)
    assert fpr_result.passed is True


# ---------------------------------------------------------------------------
# Test 2: missed anomaly → recall=0.5
# ---------------------------------------------------------------------------


def test_missed_anomaly() -> None:
    """One of two anomalies missed → recall=0.5."""
    predicted = [True, False, False, False]
    actual = [True, True, False, False]

    results = compute_anomaly_recall(predicted, actual)
    recall_result, fpr_result = results

    assert recall_result.name == "anomaly_recall"
    assert recall_result.value == pytest.approx(0.5)
    assert recall_result.details["tp"] == 1
    assert recall_result.details["fn"] == 1
    assert recall_result.details["total_positive"] == 2


# ---------------------------------------------------------------------------
# Test 3: false positive rate = 1.0 when all negatives are misclassified
# ---------------------------------------------------------------------------


def test_false_positive() -> None:
    """All negatives predicted as positive → fpr=1.0."""
    predicted = [True, True, True]
    actual = [True, False, False]

    results = compute_anomaly_recall(predicted, actual)
    recall_result, fpr_result = results

    assert fpr_result.name == "anomaly_false_positive"
    assert fpr_result.value == pytest.approx(1.0)
    assert fpr_result.details["fp"] == 2
    assert fpr_result.details["tn"] == 0
    assert fpr_result.details["total_negative"] == 2


# ---------------------------------------------------------------------------
# Test 4: no anomalies in ground truth → recall=0.0, fpr=0.0
# ---------------------------------------------------------------------------


def test_no_anomalies() -> None:
    """No positive labels in actual → recall=0.0, fpr=0.0 (all-negative case)."""
    predicted = [False, False]
    actual = [False, False]

    results = compute_anomaly_recall(predicted, actual)
    recall_result, fpr_result = results

    assert recall_result.value == pytest.approx(0.0)
    assert fpr_result.value == pytest.approx(0.0)
    assert recall_result.details["total_positive"] == 0
    assert fpr_result.details["total_negative"] == 2


# ---------------------------------------------------------------------------
# Test 5: thresholds forwarded to MetricResult
# ---------------------------------------------------------------------------


def test_thresholds_passed() -> None:
    """recall_threshold and fpr_threshold are forwarded to MetricResult."""
    predicted = [True, False]
    actual = [True, False]

    results = compute_anomaly_recall(
        predicted,
        actual,
        recall_threshold=0.9,
        fpr_threshold=0.1,
    )
    recall_result, fpr_result = results

    assert recall_result.threshold == pytest.approx(0.9)
    assert fpr_result.threshold == pytest.approx(0.1)
    # recall=1.0 >= 0.9 → passed; fpr=0.0 <= 0.1 → passed
    assert recall_result.passed is True
    assert fpr_result.passed is True
