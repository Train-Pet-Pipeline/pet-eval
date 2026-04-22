"""Tests for WeightedFusionEvaluator."""

from __future__ import annotations

import pytest

from pet_eval.plugins.fusion.weighted import WeightedFusionEvaluator


def test_weighted_basic() -> None:
    """Weighted sum with 0.7/0.3 split."""
    ev = WeightedFusionEvaluator(weights={"audio": 0.7, "vlm": 0.3})
    assert ev.fuse({"audio": 1.0, "vlm": 0.0}) == pytest.approx(0.7)


def test_weighted_normalizes() -> None:
    """Equal weights normalize to 0.5/0.5 regardless of absolute values."""
    ev = WeightedFusionEvaluator(weights={"audio": 2.0, "vlm": 2.0})
    assert ev.fuse({"audio": 0.6, "vlm": 0.4}) == pytest.approx(0.5)


def test_weighted_missing_modality_treated_as_zero() -> None:
    """Missing modalities are treated as score 0.0."""
    ev = WeightedFusionEvaluator(weights={"audio": 0.5, "vlm": 0.5})
    assert ev.fuse({"audio": 0.8}) == pytest.approx(0.4)
