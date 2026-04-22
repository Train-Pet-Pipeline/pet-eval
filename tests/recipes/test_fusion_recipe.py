"""Validation tests for the cross_modal_fusion_eval recipe fixture (P2-B-3)."""
from __future__ import annotations

from pathlib import Path

import yaml
from pet_schema.recipe import ExperimentRecipe

_RECIPE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "recipes" / "cross_modal_fusion_eval.yaml"
)


def _load() -> ExperimentRecipe:
    """Load and validate the cross_modal_fusion_eval recipe from disk."""
    raw = yaml.safe_load(_RECIPE_PATH.read_text())
    # smoke_foundation.yaml wraps under top-level "recipe:" — match that convention
    payload = raw["recipe"] if "recipe" in raw else raw
    return ExperimentRecipe.model_validate(payload)


def test_fusion_recipe_validates_against_schema() -> None:
    """Recipe YAML parses cleanly against ExperimentRecipe schema."""
    rec = _load()
    assert rec.recipe_id == "cross_modal_fusion_eval"


def test_fusion_recipe_has_eval_stage() -> None:
    """Recipe must declare an 'eval' stage."""
    rec = _load()
    names = {s.name for s in rec.stages}
    assert "eval" in names


def test_fusion_recipe_variation_covers_all_three_strategies() -> None:
    """fusion_strategy ablation axis must cover all 3 fusion plugins."""
    rec = _load()
    axis = next(v for v in rec.variations if v.name == "fusion_strategy")
    assert set(axis.values) == {"single_modal_fusion", "and_gate_fusion", "weighted_fusion"}
    assert axis.stage == "eval"
