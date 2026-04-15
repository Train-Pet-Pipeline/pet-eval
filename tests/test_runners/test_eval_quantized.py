"""Tests for pet_eval.runners.eval_quantized."""
from __future__ import annotations

import pathlib
from typing import Any
from unittest.mock import MagicMock, patch

import yaml

from pet_eval.gate.types import GateResult
from pet_eval.runners.eval_quantized import run_eval_quantized


def _write_params(tmp_dir: pathlib.Path, params: dict[str, Any]) -> pathlib.Path:
    """Write params dict to a temporary params.yaml and return the path."""
    params_file = tmp_dir / "params.yaml"
    params_file.write_text(yaml.dump(params))
    return params_file


@patch("pet_eval.report.generate_report.wandb")
@patch("pet_eval.runners.eval_quantized._run_quantized_inference")
def test_no_device_skips_latency(
    mock_inference: MagicMock,
    mock_wandb: MagicMock,
    tmp_dir: pathlib.Path,
    sample_params: dict[str, Any],
) -> None:
    """When device_id is None, latency_p95_ms must appear in skipped."""
    mock_inference.return_value = {
        "outputs": ['{"schema_version": "1.0", "pet_present": false}'],
        "timings": [],
        "fp16_outputs": [],
    }

    params_file = _write_params(tmp_dir, sample_params)

    mock_run = MagicMock()
    mock_run.summary = {}
    mock_wandb.init.return_value = mock_run

    result = run_eval_quantized(
        model_dir="/fake/quantized_model",
        run_name="test-no-device",
        device_id=None,
        params_path=str(params_file),
    )

    assert isinstance(result, GateResult)
    assert "latency_p95_ms" in result.skipped, (
        f"Expected 'latency_p95_ms' in skipped, got {result.skipped}"
    )


@patch("pet_eval.report.generate_report.wandb")
@patch("pet_eval.runners.eval_quantized._run_quantized_inference")
def test_with_device_includes_latency(
    mock_inference: MagicMock,
    mock_wandb: MagicMock,
    tmp_dir: pathlib.Path,
    sample_params: dict[str, Any],
) -> None:
    """When device_id is set and timings are returned, latency_p95_ms must NOT be skipped."""
    mock_inference.return_value = {
        "outputs": ['{"schema_version": "1.0", "pet_present": false}'],
        "timings": [2000.0] * 50,
        "fp16_outputs": [],
    }

    params_file = _write_params(tmp_dir, sample_params)

    mock_run = MagicMock()
    mock_run.summary = {}
    mock_wandb.init.return_value = mock_run

    result = run_eval_quantized(
        model_dir="/fake/quantized_model",
        run_name="test-with-device",
        device_id="12345",
        params_path=str(params_file),
    )

    assert isinstance(result, GateResult)
    assert "latency_p95_ms" not in result.skipped, (
        f"Expected 'latency_p95_ms' NOT in skipped, got {result.skipped}"
    )


@patch("pet_eval.report.generate_report.wandb")
@patch("pet_eval.runners.eval_quantized._run_quantized_inference")
def test_returns_gate_result(
    mock_inference: MagicMock,
    mock_wandb: MagicMock,
    tmp_dir: pathlib.Path,
    sample_params: dict[str, Any],
) -> None:
    """run_eval_quantized must always return a GateResult instance."""
    mock_inference.return_value = {
        "outputs": ['{"schema_version": "1.0", "pet_present": false}'],
        "timings": [],
        "fp16_outputs": [],
    }

    params_file = _write_params(tmp_dir, sample_params)

    mock_run = MagicMock()
    mock_run.summary = {}
    mock_wandb.init.return_value = mock_run

    result = run_eval_quantized(
        model_dir="/fake/quantized_model",
        run_name="test-gate-result",
        device_id=None,
        params_path=str(params_file),
    )

    assert isinstance(result, GateResult)
