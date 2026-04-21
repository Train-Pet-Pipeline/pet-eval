"""Tests for pet_eval.runners.eval_trained."""

from __future__ import annotations

import pathlib
from typing import Any
from unittest.mock import MagicMock, patch

import yaml

from pet_eval.gate.types import GateResult
from pet_eval.runners.eval_trained import run_eval_trained


def _write_params(tmp_dir: pathlib.Path, params: dict[str, Any]) -> pathlib.Path:
    """Write params dict to a temporary params.yaml and return the path."""
    params_file = tmp_dir / "params.yaml"
    params_file.write_text(yaml.dump(params))
    return params_file


@patch("pet_eval.runners.eval_trained.generate_report")
@patch("pet_eval.runners.eval_trained._run_inference")
def test_no_gold_set_runs_schema_only(
    mock_inference: MagicMock,
    mock_report: MagicMock,
    tmp_dir: pathlib.Path,
    sample_params: dict[str, Any],
) -> None:
    """When the gold set does not exist, skipped list must contain gold-set
    dependent metrics, and run_eval_trained must still return a GateResult."""
    mock_inference.return_value = ['{"schema_version": "1.0", "pet_present": false}']

    # Point benchmark paths at non-existent files so has_gold_set is False
    sample_params["benchmark"]["gold_set_path"] = str(tmp_dir / "gold_set.jsonl")
    sample_params["benchmark"]["anomaly_set_path"] = str(tmp_dir / "anomaly_set.jsonl")

    params_file = _write_params(tmp_dir, sample_params)

    result = run_eval_trained(
        model_path="/fake/model",
        run_name="test-run",
        params_path=str(params_file),
    )

    assert isinstance(result, GateResult)
    # Gold-set-dependent metrics must be in skipped
    expected_skipped = {
        "anomaly_recall",
        "anomaly_false_positive",
        "calibration_ece",
        "mood_spearman",
        "narrative_bertscore",
    }
    assert expected_skipped.issubset(set(result.skipped)), (
        f"Expected {expected_skipped} in skipped, got {result.skipped}"
    )


@patch("pet_eval.runners.eval_trained.generate_report")
@patch("pet_eval.runners.eval_trained._run_inference")
def test_returns_gate_result(
    mock_inference: MagicMock,
    mock_report: MagicMock,
    tmp_dir: pathlib.Path,
    sample_params: dict[str, Any],
) -> None:
    """run_eval_trained must always return a GateResult instance."""
    mock_inference.return_value = ['{"schema_version": "1.0", "pet_present": false}']

    sample_params["benchmark"]["gold_set_path"] = str(tmp_dir / "gold_set.jsonl")
    sample_params["benchmark"]["anomaly_set_path"] = str(tmp_dir / "anomaly_set.jsonl")

    params_file = _write_params(tmp_dir, sample_params)

    result = run_eval_trained(
        model_path="/fake/model",
        run_name="test-run-2",
        params_path=str(params_file),
    )

    assert isinstance(result, GateResult)


@patch("pet_eval.runners.eval_trained.generate_report")
@patch("pet_eval.runners.eval_trained._run_inference")
def test_exit_code_logic(
    mock_inference: MagicMock,
    mock_report: MagicMock,
    tmp_dir: pathlib.Path,
    sample_params: dict[str, Any],
) -> None:
    """When _run_inference returns [], run_eval_trained still returns a GateResult.

    The gate may pass or fail depending on thresholds; the important invariant
    is that the return type is always GateResult.
    """
    mock_inference.return_value = []

    sample_params["benchmark"]["gold_set_path"] = str(tmp_dir / "gold_set.jsonl")
    sample_params["benchmark"]["anomaly_set_path"] = str(tmp_dir / "anomaly_set.jsonl")

    params_file = _write_params(tmp_dir, sample_params)

    result = run_eval_trained(
        model_path="/fake/model",
        run_name="test-run-empty",
        params_path=str(params_file),
    )

    assert isinstance(result, GateResult)
