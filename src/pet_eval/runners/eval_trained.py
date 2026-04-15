"""VLM eval_trained runner for pet-eval.

Evaluates a trained VLM checkpoint against the benchmark gold set and anomaly
set, computes schema compliance, gates the result, and publishes a W&B report.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from pet_eval.gate.checker import check_gate
from pet_eval.gate.types import GateResult
from pet_eval.logging_setup import setup_logging
from pet_eval.metrics.schema_compliance import compute_schema_compliance
from pet_eval.report.generate_report import generate_report

logger = logging.getLogger(__name__)


def _run_inference(
    model_path: str,
    gold_set_path: str | None,
    params: dict[str, Any],
) -> list[str]:
    """Run VLM inference over the gold set and return raw output strings.

    Args:
        model_path: Path to the trained model checkpoint directory.
        gold_set_path: Path to the JSONL gold set file, or None if unavailable.
        params: Full params dict (used for inference config).

    Returns:
        List of raw JSON strings produced by the model.  Returns an empty list
        when *gold_set_path* is None or when inference is not yet implemented.
    """
    if gold_set_path is None:
        return []

    records: list[dict[str, Any]] = []
    with open(gold_set_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(
        "_run_inference: loaded gold set",
        extra={"n_records": len(records), "model_path": model_path},
    )

    # TODO: Actual inference depends on HF model loading.
    logger.warning(
        "Model inference not yet implemented",
        extra={"model_path": model_path, "n_records": len(records)},
    )
    return []


def run_eval_trained(
    model_path: str,
    run_name: str,
    params_path: str = "params.yaml",
) -> GateResult:
    """Evaluate a trained VLM checkpoint and gate its quality.

    Steps:
      1. Load params.yaml — extract gates.vlm thresholds, benchmark paths,
         wandb config, and inference config.
      2. Check whether gold_set_path and anomaly_set_path exist and have
         content.
      3. Run ``_run_inference`` against the gold set (or return [] if missing).
      4. Always compute schema_compliance with thresholds from params.
      5. Skip anomaly_recall, anomaly_false_positive, calibration_ece,
         mood_spearman, and narrative_bertscore when the gold set is absent;
         skip anomaly_recall and anomaly_false_positive when only the anomaly
         set is absent.
      6. Call ``check_gate`` to aggregate results.
      7. Call ``generate_report`` to publish to W&B.

    Args:
        model_path: Path to the trained model checkpoint directory.
        run_name: Short human-readable identifier for this evaluation run.
        params_path: Path to params.yaml (defaults to ``"params.yaml"``).

    Returns:
        A frozen :class:`GateResult` instance.
    """
    # 1. Load params
    with open(params_path) as fh:
        params: dict[str, Any] = yaml.safe_load(fh)

    vlm_gates: dict[str, Any] = params["gates"]["vlm"]
    benchmark_cfg: dict[str, Any] = params["benchmark"]
    wandb_cfg: dict[str, Any] = params["wandb"]
    inference_cfg: dict[str, Any] = params.get("inference", {})
    schema_version: str = str(inference_cfg.get("schema_version", "1.0"))

    gold_set_path: str = benchmark_cfg["gold_set_path"]
    anomaly_set_path: str = benchmark_cfg["anomaly_set_path"]

    # 2. Check existence and content
    gold_path_obj = Path(gold_set_path)
    has_gold_set = gold_path_obj.exists() and gold_path_obj.stat().st_size > 0

    anomaly_path_obj = Path(anomaly_set_path)
    has_anomaly_set = anomaly_path_obj.exists() and anomaly_path_obj.stat().st_size > 0

    if not has_gold_set:
        logger.warning(
            "Gold set not found or empty; skipping gold-set-dependent metrics",
            extra={"gold_set_path": gold_set_path},
        )
    if not has_anomaly_set:
        logger.warning(
            "Anomaly set not found or empty; skipping anomaly metrics",
            extra={"anomaly_set_path": anomaly_set_path},
        )

    # 3. Run inference (returns [] when gold set absent)
    outputs = _run_inference(
        model_path,
        gold_set_path if has_gold_set else None,
        params,
    )

    # 4. Schema compliance — always run
    results = compute_schema_compliance(
        outputs,
        compliance_threshold=vlm_gates.get("schema_compliance"),
        sum_error_threshold=vlm_gates.get("distribution_sum_error"),
        schema_version=schema_version,
    )

    # 5. Build skipped list
    #    narrative_bertscore and mood_spearman depend on teacher references (not
    #    gold set).  Per the design spec, they should run when teacher outputs
    #    are available even if the gold set is absent.  For now, teacher
    #    references are not yet available, so they are always skipped.
    skipped: list[str] = []

    # Teacher-reference-dependent metrics — skipped until teacher outputs exist.
    # TODO: Check for teacher reference file and conditionally skip.
    has_teacher_references = False
    if not has_teacher_references:
        skipped.extend(["mood_spearman", "narrative_bertscore"])

    if not has_gold_set:
        skipped.extend(
            [
                "anomaly_recall",
                "anomaly_false_positive",
                "calibration_ece",
            ]
        )
    elif not has_anomaly_set:
        skipped.extend(["anomaly_recall", "anomaly_false_positive"])

    # 6. Gate check
    gate_result = check_gate(results, skipped, "vlm", params)

    # 7. Report
    metadata: dict[str, Any] = {
        "model_path": model_path,
        "params_path": params_path,
        "has_gold_set": has_gold_set,
        "has_anomaly_set": has_anomaly_set,
        "n_outputs": len(outputs),
    }
    generate_report(gate_result, run_name, "vlm_trained", metadata, wandb_cfg)

    return gate_result


def main() -> None:
    """CLI entry point for the eval_trained runner.

    Parses arguments, runs evaluation, and exits with code 0 on pass or 1 on
    fail.
    """
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Evaluate a trained VLM checkpoint against the pet-eval benchmark."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the trained model checkpoint directory.",
    )
    parser.add_argument(
        "--run_name",
        required=True,
        help="Short identifier for this evaluation run (used in W&B).",
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        dest="params_path",
        help="Path to params.yaml (default: params.yaml).",
    )
    args = parser.parse_args()

    result = run_eval_trained(
        model_path=args.model_path,
        run_name=args.run_name,
        params_path=args.params_path,
    )

    logger.info(
        "eval_trained finished",
        extra={"passed": result.passed, "summary": result.summary},
    )
    sys.exit(0 if result.passed else 1)
