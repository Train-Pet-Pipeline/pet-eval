"""Tests for pet_eval.plugins.vlm_evaluator."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pet_eval.plugins.vlm_evaluator import VLMEvaluator


@pytest.fixture
def sample_card():
    """Return a minimal valid ModelCard for test use."""
    from pet_schema.model_card import ModelCard

    return ModelCard(
        id="sft-card-1",
        version="1.0.0",
        modality="vision",
        task="sft",
        arch="qwen2vl_lora_r16_a32",
        training_recipe="dummy",
        hydra_config_sha="a" * 64,
        git_shas={},
        dataset_versions={},
        checkpoint_uri="file:///tmp/fake_model",
        metrics={},
        gate_status="pending",
        trained_at=datetime.now(UTC),
        trained_by="ci",
    )


def test_init_builds_metrics_from_config():
    """VLMEvaluator should build metric instances from config list."""
    from pet_eval.plugins._register import register_all

    register_all()
    evaluator = VLMEvaluator(
        metrics=["schema_compliance"],
        thresholds={"min_schema_compliance_rate": 0.8},
        params={},
    )
    assert len(evaluator._metrics) == 1


def test_registers_to_evaluators():
    """vlm_evaluator should be registered in the EVALUATORS registry."""
    from pet_eval.plugins._register import register_all

    register_all()
    from pet_infra.registry import EVALUATORS

    assert "vlm_evaluator" in EVALUATORS.module_dict


def test_registry_build_produces_evaluator():
    """EVALUATORS.build with type=vlm_evaluator should return a VLMEvaluator."""
    from pet_eval.plugins._register import register_all

    register_all()
    from pet_infra.registry import EVALUATORS

    evaluator = EVALUATORS.build(
        {
            "type": "vlm_evaluator",
            "metrics": ["schema_compliance"],
            "thresholds": {},
            "params": {},
        }
    )
    assert isinstance(evaluator, VLMEvaluator)


def test_run_raises_without_input_card():
    """run() should raise ValueError when input_card is None."""
    from pet_eval.plugins._register import register_all

    register_all()
    evaluator = VLMEvaluator(metrics=[], thresholds={}, params={})
    with pytest.raises(ValueError, match="requires a trained model_card"):
        evaluator.run(input_card=None, recipe=SimpleNamespace())


def test_run_sets_gate_status_passed_when_no_thresholds(sample_card):
    """run() with empty thresholds should always pass the gate."""
    from pet_eval.plugins._register import register_all

    register_all()
    evaluator = VLMEvaluator(metrics=[], thresholds={}, params={})
    with patch("pet_eval.plugins.vlm_evaluator.run_inference", return_value=[]):
        card = evaluator.run(input_card=sample_card, recipe=SimpleNamespace())
    assert card.gate_status == "passed"
    assert card.task == "vlm_eval"


def test_run_sets_gate_status_failed_when_thresholds_miss(sample_card):
    """run() should set gate_status='failed' when a min_ threshold is not met."""
    from pet_eval.plugins._register import register_all

    register_all()
    evaluator = VLMEvaluator(
        metrics=[],
        thresholds={"min_missing_metric": 0.9},
        params={},
    )
    with patch("pet_eval.plugins.vlm_evaluator.run_inference", return_value=[]):
        card = evaluator.run(input_card=sample_card, recipe=SimpleNamespace())
    assert card.gate_status == "failed"
    assert "missing_metric" in (card.notes or "")
