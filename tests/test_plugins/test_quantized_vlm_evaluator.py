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

    import importlib

    importlib.import_module("pet_eval.plugins.quantized_vlm_evaluator")

    assert "pet_quantize.inference.rkllm_runner" not in sys.modules


def test_runs_and_merges_vlm_accuracy_into_metrics(monkeypatch):
    """Happy path: mock RKLLMRunner, assert metrics dict contains vlm_accuracy."""
    from pet_eval.plugins.quantized_vlm_evaluator import QuantizedVlmEvaluator

    mock_runner_cls = MagicMock()
    mock_runner_cls.return_value.predict.return_value = [
        {"label": "cat", "score": 0.9},
        {"label": "dog", "score": 0.8},
        {"label": "bird", "score": 0.3},  # below threshold
    ]
    mock_mod = MagicMock(RKLLMRunner=mock_runner_cls)
    monkeypatch.setitem(sys.modules, "pet_quantize.inference.rkllm_runner", mock_mod)

    plugin = QuantizedVlmEvaluator(
        metrics=["vlm_accuracy"],
        thresholds={"min_vlm_accuracy": 0.5},
        eval_set_uri="/tmp/eval",
    )
    out = plugin.run(_make_card(), recipe=MagicMock())

    assert "vlm_accuracy" in out.metrics
    assert out.metrics["vlm_accuracy"] == pytest.approx(2 / 3)
    assert out.gate_status == "passed"
    mock_runner_cls.assert_called_once_with(model_path="/tmp/m.rkllm", target="rk3576")


def test_gate_fail_sets_status_failed(monkeypatch):
    """Gate fails when vlm_accuracy is below min threshold."""
    from pet_eval.plugins.quantized_vlm_evaluator import QuantizedVlmEvaluator

    mock_runner_cls = MagicMock()
    mock_runner_cls.return_value.predict.return_value = [{"score": 0.1}] * 10
    monkeypatch.setitem(
        sys.modules,
        "pet_quantize.inference.rkllm_runner",
        MagicMock(RKLLMRunner=mock_runner_cls),
    )

    plugin = QuantizedVlmEvaluator(
        metrics=["vlm_accuracy"],
        thresholds={"min_vlm_accuracy": 0.9},
        eval_set_uri="/tmp/eval",
    )
    out = plugin.run(_make_card(), recipe=MagicMock())
    assert out.gate_status == "failed"
    assert out.notes is not None and "vlm_accuracy" in out.notes


def test_rejects_card_without_rkllm_edge_artifact():
    """Raises ValueError when card has no rkllm edge artifact."""
    from pet_eval.plugins.quantized_vlm_evaluator import QuantizedVlmEvaluator

    plugin = QuantizedVlmEvaluator(metrics=["vlm_accuracy"], eval_set_uri="/tmp/x")
    with pytest.raises(ValueError, match="rkllm"):
        plugin.run(_make_card(with_rkllm=False), recipe=MagicMock())


def test_registered_in_evaluators():
    """QuantizedVlmEvaluator is accessible via EVALUATORS registry after register_all."""
    from pet_eval.plugins._register import register_all

    register_all()
    from pet_infra.registry import EVALUATORS

    assert "quantized_vlm_evaluator" in EVALUATORS.module_dict
