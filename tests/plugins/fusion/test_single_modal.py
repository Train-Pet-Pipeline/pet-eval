"""Tests for SingleModalFusionEvaluator."""

from __future__ import annotations

import pytest

from pet_eval.plugins.fusion.single_modal import SingleModalFusionEvaluator


def test_single_modal_passes_through_modality() -> None:
    """Returns the score for the named modality."""
    ev = SingleModalFusionEvaluator(modality="audio")
    assert ev.fuse({"audio": 0.8, "vlm": 0.3}) == 0.8


def test_single_modal_missing_raises() -> None:
    """Raises KeyError when the requested modality is absent."""
    ev = SingleModalFusionEvaluator(modality="audio")
    with pytest.raises(KeyError):
        ev.fuse({"vlm": 0.3})
