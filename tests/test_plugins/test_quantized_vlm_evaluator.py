"""Tests for pet_eval.plugins.quantized_vlm_evaluator."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from pet_schema.model_card import EdgeArtifact, ModelCard


def _make_card(with_rkllm: bool = True) -> ModelCard:
    """Return a minimal valid ModelCard for test use."""
    edges: list[EdgeArtifact] = []
    if with_rkllm:
        edges.append(
            EdgeArtifact(
                format="rkllm",
                target_hardware=["rk3576"],
                artifact_uri="/tmp/m.rkllm",
                sha256="a" * 64,
                size_bytes=100,
                input_shape={"input_ids": [1, 2048]},
            )
        )
    return ModelCard(
        id="q-card-1",
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
        edge_artifacts=edges,
    )


def test_module_load_does_not_import_rkllm_runner():
    """Module-load must NOT transitively import pet_quantize.inference.rkllm_runner."""
    for mod in list(sys.modules):
        if mod.startswith("pet_quantize.inference"):
            del sys.modules[mod]
    # Also drop our plugin module so re-import is fresh
    sys.modules.pop("pet_eval.plugins.quantized_vlm_evaluator", None)
    sys.modules.pop("pet_eval.plugins.quantized_vlm_inference", None)

    import importlib

    importlib.import_module("pet_eval.plugins.quantized_vlm_evaluator")

    assert "pet_quantize.inference.rkllm_runner" not in sys.modules


def test_runs_and_merges_metric_into_card(monkeypatch):
    """Happy path: mock run_inference, assert task and gate_status are set."""
    from pet_eval.plugins._register import register_all

    register_all()
    from pet_eval.plugins import quantized_vlm_evaluator as mod
    from pet_eval.plugins.quantized_vlm_evaluator import QuantizedVlmEvaluator

    monkeypatch.setattr(mod, "run_inference", lambda **kw: ['{"schema_version": "1.0"}'] * 3)
    plugin = QuantizedVlmEvaluator(
        metrics=["schema_compliance"],
        thresholds={},
        eval_set_uri="/tmp/eval",
        params={},
    )
    out = plugin.run(_make_card(), recipe=MagicMock())
    assert out.task == "quantized_vlm_eval"
    assert out.gate_status == "passed"


def test_gate_fail_sets_status_failed(monkeypatch):
    """Gate fails when a required metric is missing (below min threshold)."""
    from pet_eval.plugins._register import register_all

    register_all()
    from pet_eval.plugins import quantized_vlm_evaluator as mod
    from pet_eval.plugins.quantized_vlm_evaluator import QuantizedVlmEvaluator

    monkeypatch.setattr(mod, "run_inference", lambda **kw: [])
    plugin = QuantizedVlmEvaluator(
        metrics=[],
        thresholds={"min_nonexistent_metric": 0.99},
        eval_set_uri="/tmp/eval",
        params={},
    )
    out = plugin.run(_make_card(), recipe=MagicMock())
    assert out.gate_status == "failed"
    assert out.notes is not None and "nonexistent_metric" in out.notes


def test_rejects_card_without_rkllm_edge_artifact():
    """Raises ValueError when card has no rkllm edge artifact."""
    from pet_eval.plugins.quantized_vlm_evaluator import QuantizedVlmEvaluator

    plugin = QuantizedVlmEvaluator(metrics=["schema_compliance"], eval_set_uri="/tmp/x")
    with pytest.raises(ValueError, match="rkllm"):
        plugin.run(_make_card(with_rkllm=False), recipe=MagicMock())


def test_registered_in_evaluators():
    """QuantizedVlmEvaluator is accessible via EVALUATORS registry after register_all."""
    from pet_eval.plugins._register import register_all

    register_all()
    from pet_infra.registry import EVALUATORS

    assert "quantized_vlm_evaluator" in EVALUATORS.module_dict
