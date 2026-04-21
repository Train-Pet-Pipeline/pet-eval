"""Local JSON report generation for pet-eval gate results.

Writes gate evaluation outcomes to a JSON file in the model directory.
Experiment tracking via ClearML is handled by the orchestrator (P0-B/C).

Note: The ``wandb_config`` parameter is accepted for caller compatibility
but ignored — wandb logging was removed in Phase 3A. The parameter will be
dropped when eval_trained/eval_audio runners migrate to plugins (P2-C/D).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pet_eval.gate.types import GateResult

logger = logging.getLogger(__name__)


def generate_report(
    gate_result: GateResult,
    run_name: str,
    eval_type: str,
    metadata: dict[str, Any],
    wandb_config: dict[str, Any] | None = None,
) -> None:
    """Write a gate evaluation result to a local JSON report file.

    Writes a structured JSON report under
    ``{metadata["model_path"]}/eval_reports/{eval_type}_{run_name}.json``
    (falls back to the current directory when ``model_path`` is absent).

    Args:
        gate_result: Aggregated gate evaluation result.
        run_name: Short identifier for the model/checkpoint under test.
        eval_type: Category label (``"vlm_trained"``, ``"audio"``, etc.).
        metadata: Arbitrary key/value pairs stored in the report.
        wandb_config: Ignored — accepted for backward-compat with callers
            that have not yet migrated to the plugin API (P2-C/D).
    """
    detail_payload: dict[str, Any] = {
        "gate_passed": gate_result.passed,
        "gate_summary": gate_result.summary,
        "skipped": gate_result.skipped,
    }
    for metric in gate_result.results:
        detail_payload[f"metric/{metric.name}"] = metric.value

    report_dir = Path(metadata.get("model_path", ".")) / "eval_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{eval_type}_{run_name}.json"

    local_report = {
        "eval_type": eval_type,
        "run_name": run_name,
        "metadata": metadata,
        **detail_payload,
        "metrics_detail": [
            {
                "name": m.name,
                "value": m.value,
                "threshold": m.threshold,
                "passed": m.passed,
            }
            for m in gate_result.results
        ],
    }
    with open(report_path, "w") as fh:
        json.dump(local_report, fh, indent=2, ensure_ascii=False)

    logger.info(
        "generate_report",
        extra={
            "eval_type": eval_type,
            "run_name": run_name,
            "passed": gate_result.passed,
            "path": str(report_path),
        },
    )
