"""Validation tests for the cross_modal_fusion_eval recipe fixture (P2-B-3)."""
from __future__ import annotations

from pathlib import Path

import yaml
from pet_schema.recipe import ExperimentRecipe

from pet_eval.plugins.fusion.and_gate import AndGateFusionEvaluator
from pet_eval.plugins.fusion.single_modal import SingleModalFusionEvaluator
from pet_eval.plugins.fusion.weighted import WeightedFusionEvaluator

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RECIPE_PATH = _REPO_ROOT / "recipes" / "cross_modal_fusion_eval.yaml"


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


def test_fusion_recipe_config_path_is_loadable_by_all_three_strategies() -> None:
    """Stage config_path must exist and satisfy every strategy in the sweep.

    Regression for finding ⑧: orchestrator runner.py loads `stage.config_path`
    via yaml.safe_load, then splats the dict as kwargs into the component.
    Because `fusion_strategy` sweeps ``component_type`` while reusing one
    config file, that file must contain keys for every strategy — each fusion
    evaluator's ``**_`` swallows the unused extras.
    """
    rec = _load()
    (stage,) = (s for s in rec.stages if s.name == "eval")
    config_path = _REPO_ROOT / stage.config_path
    assert config_path.exists(), f"config_path missing on disk: {config_path}"

    cfg = yaml.safe_load(config_path.read_text())
    WeightedFusionEvaluator(**cfg)
    AndGateFusionEvaluator(**cfg)
    SingleModalFusionEvaluator(**cfg)
