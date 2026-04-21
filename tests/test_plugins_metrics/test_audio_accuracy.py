"""Tests for audio_accuracy metric — TDD, tests written before implementation."""

from __future__ import annotations

import pytest

from pet_eval.plugins.metrics.audio_accuracy import compute_audio_accuracy
from pet_eval.plugins.metrics.types import MetricResult

# ---------------------------------------------------------------------------
# Test 1: perfect accuracy — all predictions correct → accuracy=1.0
# ---------------------------------------------------------------------------


def test_perfect_accuracy() -> None:
    """All predictions match ground truth → accuracy=1.0."""
    classes = ["eat", "vomit", "ambient"]
    predicted = ["eat", "vomit", "ambient", "eat"]
    actual = ["eat", "vomit", "ambient", "eat"]

    results = compute_audio_accuracy(predicted, actual, classes)

    assert len(results) == 2
    acc_result, vomit_result = results

    assert isinstance(acc_result, MetricResult)
    assert acc_result.name == "audio_overall_accuracy"
    assert acc_result.value == pytest.approx(1.0)
    assert acc_result.passed is True

    assert isinstance(vomit_result, MetricResult)
    assert vomit_result.name == "audio_vomit_recall"
    assert vomit_result.value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 2: zero accuracy — all predictions wrong → accuracy=0.0
# ---------------------------------------------------------------------------


def test_zero_accuracy() -> None:
    """All predictions are wrong → accuracy=0.0."""
    classes = ["eat", "vomit", "ambient"]
    predicted = ["vomit", "ambient", "eat"]
    actual = ["eat", "vomit", "ambient"]

    results = compute_audio_accuracy(predicted, actual, classes)
    acc_result, _ = results

    assert acc_result.name == "audio_overall_accuracy"
    assert acc_result.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 3: vomit recall computed separately from overall accuracy
# ---------------------------------------------------------------------------


def test_vomit_recall_separate() -> None:
    """[eat,vomit,ambient,eat] vs [eat,vomit,vomit,ambient] → vomit_recall=0.5."""
    classes = ["eat", "vomit", "ambient"]
    predicted = ["eat", "vomit", "ambient", "eat"]
    actual = ["eat", "vomit", "vomit", "ambient"]

    results = compute_audio_accuracy(predicted, actual, classes)
    acc_result, vomit_result = results

    assert vomit_result.name == "audio_vomit_recall"
    # actual "vomit" at indices 1 and 2: predicted[1]="vomit" (TP), predicted[2]="ambient" (FN)
    # vomit recall = 1 / 2 = 0.5
    assert vomit_result.value == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 4: confusion_matrix and per_class present in accuracy details
# ---------------------------------------------------------------------------


def test_confusion_matrix_in_details() -> None:
    """Details of accuracy result must contain 'confusion_matrix' and 'per_class'."""
    classes = ["eat", "vomit", "ambient"]
    predicted = ["eat", "eat", "vomit"]
    actual = ["eat", "vomit", "vomit"]

    results = compute_audio_accuracy(predicted, actual, classes)
    acc_result, _ = results

    assert "confusion_matrix" in acc_result.details
    assert "per_class" in acc_result.details
    assert "n_samples" in acc_result.details
    assert acc_result.details["n_samples"] == 3


# ---------------------------------------------------------------------------
# Test 5: thresholds forwarded to MetricResult
# ---------------------------------------------------------------------------


def test_thresholds_forwarded() -> None:
    """accuracy_threshold and vomit_recall_threshold are forwarded to MetricResult."""
    classes = ["eat", "vomit", "ambient"]
    predicted = ["eat", "vomit"]
    actual = ["eat", "vomit"]

    results = compute_audio_accuracy(
        predicted,
        actual,
        classes,
        accuracy_threshold=0.85,
        vomit_recall_threshold=0.90,
    )
    acc_result, vomit_result = results

    assert acc_result.threshold == pytest.approx(0.85)
    assert vomit_result.threshold == pytest.approx(0.90)
    # accuracy=1.0 >= 0.85 → passed; vomit_recall=1.0 >= 0.90 → passed
    assert acc_result.passed is True
    assert vomit_result.passed is True
