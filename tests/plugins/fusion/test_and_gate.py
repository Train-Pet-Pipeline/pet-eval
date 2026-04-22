"""Tests for AndGateFusionEvaluator."""

from __future__ import annotations

import pytest

from pet_eval.plugins.fusion.and_gate import AndGateFusionEvaluator


def test_and_gate_all_above_threshold_returns_min() -> None:
    """Returns min score when all modalities clear the threshold."""
    ev = AndGateFusionEvaluator(threshold=0.5)
    assert ev.fuse({"audio": 0.6, "vlm": 0.7}) == 0.6


def test_and_gate_one_below_threshold_returns_zero() -> None:
    """Returns 0.0 when any modality is below the threshold."""
    ev = AndGateFusionEvaluator(threshold=0.5)
    assert ev.fuse({"audio": 0.6, "vlm": 0.3}) == 0.0


def test_and_gate_empty_raises() -> None:
    """Raises ValueError when modality_scores is empty."""
    ev = AndGateFusionEvaluator(threshold=0.5)
    with pytest.raises(ValueError):
        ev.fuse({})
