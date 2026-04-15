"""Tests for mood_correlation metric — TDD, tests written before implementation."""
from __future__ import annotations

import pytest

from pet_eval.metrics.mood_correlation import compute_mood_correlation
from pet_eval.metrics.types import MetricResult

# ---------------------------------------------------------------------------
# Test 1: perfect correlation — identical dicts → value >= 0.0
# ---------------------------------------------------------------------------


def test_perfect_correlation() -> None:
    """Identical model and teacher moods → correlation >= 0.0."""
    moods = [
        {"alertness": 0.2, "anxiety": 0.5, "engagement": 0.8},
        {"alertness": 0.4, "anxiety": 0.3, "engagement": 0.6},
        {"alertness": 0.9, "anxiety": 0.1, "engagement": 0.2},
    ]

    result = compute_mood_correlation(moods, moods)

    assert isinstance(result, MetricResult)
    assert result.name == "mood_spearman"
    assert result.value >= 0.0


# ---------------------------------------------------------------------------
# Test 2: inverse correlation — reversed order → value < 0
# ---------------------------------------------------------------------------


def test_inverse_correlation() -> None:
    """Model moods are reversed relative to teacher → value < 0."""
    teacher_moods = [
        {"alertness": 0.1, "anxiety": 0.2, "engagement": 0.3},
        {"alertness": 0.4, "anxiety": 0.5, "engagement": 0.6},
        {"alertness": 0.7, "anxiety": 0.8, "engagement": 0.9},
    ]
    # Reverse the order so ranks are inverted → negative Spearman
    model_moods = list(reversed(teacher_moods))

    result = compute_mood_correlation(model_moods, teacher_moods)

    assert result.value < 0


# ---------------------------------------------------------------------------
# Test 3: good correlation — positively correlated → value > 0.5
# ---------------------------------------------------------------------------


def test_good_correlation() -> None:
    """Positively correlated (but not identical) moods → value > 0.5."""
    teacher_moods = [
        {"alertness": 0.1, "anxiety": 0.2, "engagement": 0.3},
        {"alertness": 0.5, "anxiety": 0.6, "engagement": 0.7},
        {"alertness": 0.8, "anxiety": 0.9, "engagement": 1.0},
    ]
    # Slightly perturbed but same order
    model_moods = [
        {"alertness": 0.15, "anxiety": 0.25, "engagement": 0.35},
        {"alertness": 0.55, "anxiety": 0.65, "engagement": 0.75},
        {"alertness": 0.85, "anxiety": 0.95, "engagement": 0.95},
    ]

    result = compute_mood_correlation(model_moods, teacher_moods)

    assert result.value > 0.5


# ---------------------------------------------------------------------------
# Test 4: threshold forwarded to MetricResult
# ---------------------------------------------------------------------------


def test_threshold_forwarded() -> None:
    """threshold=0.75 is preserved in the returned MetricResult."""
    moods = [
        {"alertness": 0.1, "anxiety": 0.9, "engagement": 0.5},
        {"alertness": 0.8, "anxiety": 0.2, "engagement": 0.6},
        {"alertness": 0.4, "anxiety": 0.6, "engagement": 0.3},
    ]

    result = compute_mood_correlation(moods, moods, threshold=0.75)

    assert result.threshold == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Test 5: details contains per_dimension breakdown
# ---------------------------------------------------------------------------


def test_details_has_per_dimension() -> None:
    """Result details must contain a 'per_dimension' key with per-dimension scores."""
    moods = [
        {"alertness": 0.1, "anxiety": 0.2, "engagement": 0.3},
        {"alertness": 0.4, "anxiety": 0.5, "engagement": 0.6},
    ]

    result = compute_mood_correlation(moods, moods)

    assert "per_dimension" in result.details
    per_dim = result.details["per_dimension"]
    assert "alertness" in per_dim
    assert "anxiety" in per_dim
    assert "engagement" in per_dim
