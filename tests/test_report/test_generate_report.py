"""Tests for generate_report() in pet_eval.report.generate_report.

All tests mock the wandb module so no real W&B network calls are made.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from pet_eval.gate.types import GateResult
from pet_eval.metrics.types import MetricResult
from pet_eval.report.generate_report import generate_report

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_gate_result() -> GateResult:
    """Build a sample GateResult with 2 MetricResults and 1 skipped metric.

    Returns:
        A :class:`GateResult` containing:
        - ``schema_compliance`` gated metric (passes at 1.0 >= 0.99)
        - ``anomaly_recall`` gated metric (passes at 0.90 >= 0.85)
        - ``"latency_p95_ms"`` skipped
    """
    results = [
        MetricResult.create(
            "schema_compliance",
            value=1.0,
            threshold=0.99,
            operator="gte",
            details={"checked": 100, "passed": 100},
        ),
        MetricResult.create(
            "anomaly_recall",
            value=0.90,
            threshold=0.85,
            operator="gte",
        ),
    ]
    skipped = ["latency_p95_ms"]
    return GateResult.from_results(results, skipped)


_WANDB_CONFIG: dict[str, Any] = {"project": "pet-eval", "entity": ""}
_METADATA: dict[str, Any] = {"model": "test-model-v1", "checkpoint": "step_1000"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("pet_eval.report.generate_report.wandb")
def test_wandb_init_called(mock_wandb: MagicMock) -> None:
    """generate_report() calls wandb.init with correct project and run name.

    Verifies:
    - ``project`` matches wandb_config["project"]
    - ``name`` is formatted as ``"{eval_type}/{run_name}"``
    """
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run

    gate_result = _make_gate_result()
    generate_report(
        gate_result=gate_result,
        run_name="sft-v1",
        eval_type="vlm",
        metadata=_METADATA,
        wandb_config=_WANDB_CONFIG,
    )

    mock_wandb.init.assert_called_once()
    call_kwargs = mock_wandb.init.call_args.kwargs
    assert call_kwargs["project"] == "pet-eval"
    assert call_kwargs["name"] == "vlm/sft-v1"


@patch("pet_eval.report.generate_report.wandb")
def test_metrics_logged(mock_wandb: MagicMock) -> None:
    """generate_report() writes each metric's value, threshold, and passed to run.summary.

    Verifies that both metrics in the sample GateResult appear as
    ``metric/{name}/value``, ``metric/{name}/threshold``, and
    ``metric/{name}/passed`` keys in run.summary.
    """
    mock_run = MagicMock()
    mock_run.summary = {}  # use real dict so __setitem__ is tracked correctly
    mock_wandb.init.return_value = mock_run

    gate_result = _make_gate_result()
    generate_report(
        gate_result=gate_result,
        run_name="sft-v1",
        eval_type="vlm",
        metadata=_METADATA,
        wandb_config=_WANDB_CONFIG,
    )

    summary = mock_run.summary
    # schema_compliance
    assert summary["metric/schema_compliance/value"] == 1.0
    assert summary["metric/schema_compliance/threshold"] == 0.99
    assert summary["metric/schema_compliance/passed"] is True
    # anomaly_recall
    assert summary["metric/anomaly_recall/value"] == 0.90
    assert summary["metric/anomaly_recall/threshold"] == 0.85
    assert summary["metric/anomaly_recall/passed"] is True


@patch("pet_eval.report.generate_report.wandb")
def test_gate_result_logged(mock_wandb: MagicMock) -> None:
    """generate_report() logs gate summary to run.summary and calls run.finish().

    Verifies:
    - ``gate/passed`` is set on run.summary
    - ``gate/skipped`` contains the skipped metric name
    - ``run.finish()`` is called exactly once
    """
    mock_run = MagicMock()
    mock_run.summary = {}  # use real dict so __setitem__ is tracked correctly
    mock_wandb.init.return_value = mock_run

    gate_result = _make_gate_result()
    generate_report(
        gate_result=gate_result,
        run_name="sft-v1",
        eval_type="vlm",
        metadata=_METADATA,
        wandb_config=_WANDB_CONFIG,
    )

    summary = mock_run.summary
    assert summary["gate/passed"] is True
    assert "latency_p95_ms" in summary["gate/skipped"]

    mock_run.finish.assert_called_once()
