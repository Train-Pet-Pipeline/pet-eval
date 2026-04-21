"""Tests for pet_eval.runners.eval_audio."""

from __future__ import annotations

import pathlib
from typing import Any
from unittest.mock import MagicMock, patch

import yaml

from pet_eval.gate.types import GateResult
from pet_eval.runners.eval_audio import run_eval_audio


def _write_params(tmp_dir: pathlib.Path, params: dict[str, Any]) -> pathlib.Path:
    """Write params dict to a temporary params.yaml and return the path."""
    params_file = tmp_dir / "params.yaml"
    params_file.write_text(yaml.dump(params))
    return params_file


@patch("pet_eval.runners.eval_audio.generate_report")
@patch("pet_eval.runners.eval_audio._run_audio_inference")
def test_returns_gate_result(
    mock_inference: MagicMock,
    mock_report: MagicMock,
    tmp_dir: pathlib.Path,
    sample_params: dict[str, Any],
) -> None:
    """run_eval_audio must return a GateResult when inference returns mixed results."""
    mock_inference.return_value = (
        ["eating", "drinking", "eating"],
        ["eating", "drinking", "vomiting"],
    )

    params_file = _write_params(tmp_dir, sample_params)

    result = run_eval_audio(
        model_path="/fake/audio_model",
        run_name="test-audio-run",
        params_path=str(params_file),
    )

    assert isinstance(result, GateResult)


@patch("pet_eval.runners.eval_audio.generate_report")
@patch("pet_eval.runners.eval_audio._run_audio_inference")
def test_uses_audio_gate(
    mock_inference: MagicMock,
    mock_report: MagicMock,
    tmp_dir: pathlib.Path,
    sample_params: dict[str, Any],
) -> None:
    """When all predictions are correct, the audio gate must pass."""
    # All predictions correct — overall_accuracy = 1.0, vomit_recall = 1.0,
    # both above the thresholds of 0.80 and 0.70 respectively.
    mock_inference.return_value = (
        ["eating", "drinking", "vomiting", "ambient", "other"],
        ["eating", "drinking", "vomiting", "ambient", "other"],
    )

    params_file = _write_params(tmp_dir, sample_params)

    result = run_eval_audio(
        model_path="/fake/audio_model",
        run_name="test-audio-pass",
        params_path=str(params_file),
    )

    assert isinstance(result, GateResult)
    assert result.passed is True


@patch("pet_eval.runners.eval_audio.generate_report")
@patch("pet_eval.runners.eval_audio._run_audio_inference")
def test_no_test_data(
    mock_inference: MagicMock,
    mock_report: MagicMock,
    tmp_dir: pathlib.Path,
    sample_params: dict[str, Any],
) -> None:
    """When inference returns empty lists, run_eval_audio must still return GateResult."""
    mock_inference.return_value = ([], [])

    params_file = _write_params(tmp_dir, sample_params)

    result = run_eval_audio(
        model_path="/fake/audio_model",
        run_name="test-audio-empty",
        params_path=str(params_file),
    )

    assert isinstance(result, GateResult)
    # Both audio metrics should be in skipped when there is no test data
    assert "audio_overall_accuracy" in result.skipped
    assert "audio_vomit_recall" in result.skipped
