"""VLM eval_quantized runner for pet-eval.

Evaluates a quantized VLM model directory against schema compliance and,
when a hardware device is available, on-device latency.  KL divergence is
computed only when fp16 reference outputs are provided.  Gold-set-dependent
metrics (anomaly_recall, anomaly_false_positive, calibration_ece,
mood_spearman, narrative_bertscore) are always skipped because the
quantized runner is designed for fast post-quantization validation, not
full benchmark evaluation.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import yaml
from pet_infra.logging import setup_logging

from pet_eval.gate.checker import check_gate
from pet_eval.gate.types import GateResult
from pet_eval.metrics.latency import compute_latency
from pet_eval.metrics.schema_compliance import compute_schema_compliance
from pet_eval.report.generate_report import generate_report

logger = logging.getLogger(__name__)

# Metrics that depend on a gold/anomaly set and are never run by this runner.
_GOLD_SET_METRICS: list[str] = [
    "anomaly_recall",
    "anomaly_false_positive",
    "calibration_ece",
    "mood_spearman",
    "narrative_bertscore",
]


def _collect_eval_images(params: dict[str, Any]) -> list[str]:
    """Collect image paths for quantized model evaluation.

    Reads from ``eval_images_dir`` in params; falls back to the calibration
    output directory from pet-quantize.

    Args:
        params: Full params dict.

    Returns:
        List of image file paths.

    Raises:
        FileNotFoundError: If no images are found.
    """
    import glob
    from pathlib import Path

    eval_dir = params.get("inference", {}).get("eval_images_dir", "")
    if not eval_dir:
        eval_dir = params.get("calibration", {}).get("output_dir", "artifacts/calibration")

    images = glob.glob(str(Path(eval_dir) / "*.jpg"))
    images += glob.glob(str(Path(eval_dir) / "*.png"))

    if not images:
        msg = f"No evaluation images found in {eval_dir}"
        raise FileNotFoundError(msg)

    return sorted(images)


def _run_on_device(model_dir: str, device_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run quantized inference on a real hardware device via ADB.

    Delegates to ``pet_quantize.inference.run_quantized_pipeline`` with the
    device_id set, enabling RKNN/RKLLM on-device execution and latency
    measurement.

    Args:
        model_dir: Path to the quantized model directory.
        device_id: ADB device serial number.
        params: Full params dict (used for device config).

    Returns:
        Dict with keys ``"outputs"`` (list[str]), ``"timings"`` (list[float]),
        and ``"fp16_outputs"`` (list).
    """
    from pet_quantize.inference import run_quantized_pipeline

    image_paths = _collect_eval_images(params)
    logger.info(
        "_run_on_device: running on %d images",
        len(image_paths),
        extra={"model_dir": model_dir, "device_id": device_id},
    )
    return run_quantized_pipeline(
        model_dir=model_dir,
        image_paths=image_paths,
        device_id=device_id,
        params_path="params.yaml",
    )


def _run_simulated(model_dir: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run quantized inference in simulation mode (no hardware required).

    Delegates to ``pet_quantize.inference.run_quantized_pipeline`` with
    device_id=None, using the RKNN PC simulator.

    Args:
        model_dir: Path to the quantized model directory.
        params: Full params dict (used for inference config).

    Returns:
        Dict with keys ``"outputs"`` (list[str]), ``"timings"`` (list[float]),
        and ``"fp16_outputs"`` (list).
    """
    from pet_quantize.inference import run_quantized_pipeline

    image_paths = _collect_eval_images(params)
    logger.info(
        "_run_simulated: running on %d images",
        len(image_paths),
        extra={"model_dir": model_dir},
    )
    return run_quantized_pipeline(
        model_dir=model_dir,
        image_paths=image_paths,
        device_id=None,
        params_path="params.yaml",
    )


def _run_quantized_inference(
    model_dir: str,
    device_id: str | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch quantized inference to on-device or simulated backend.

    Args:
        model_dir: Path to the quantized model directory.
        device_id: ADB device serial number, or ``None`` for simulated mode.
        params: Full params dict forwarded to the backend.

    Returns:
        Dict with keys:
        - ``"outputs"`` (list[str]): Raw JSON strings from the model.
        - ``"timings"`` (list[float]): Per-sample latencies in milliseconds.
        - ``"fp16_outputs"`` (list): Reference fp16 distributions for KL
          divergence; empty list when not available.
    """
    if device_id is not None:
        return _run_on_device(model_dir, device_id, params)
    return _run_simulated(model_dir, params)


def run_eval_quantized(
    model_dir: str,
    run_name: str,
    device_id: str | None = None,
    params_path: str = "params.yaml",
) -> GateResult:
    """Evaluate a quantized VLM model and gate its quality.

    Steps:
      1. Load params.yaml — extract gates.vlm thresholds and wandb config.
      2. Determine whether a hardware device is available (device_id is not None).
      3. Call ``_run_quantized_inference`` to obtain outputs, timings, and
         fp16 reference distributions.
      4. Always compute schema_compliance on outputs.
      5. If has_device AND timings is non-empty: compute latency and append to
         results.  Otherwise: skip ``latency_p95_ms``.
      6. If fp16_outputs is empty: skip ``kl_divergence``.
      7. Always skip gold-set-dependent metrics: anomaly_recall,
         anomaly_false_positive, calibration_ece, mood_spearman,
         narrative_bertscore.
      8. Call ``check_gate`` to aggregate results.
      9. Call ``generate_report`` with eval_type ``"vlm_quantized"``.

    Args:
        model_dir: Path to the quantized model directory.
        run_name: Short human-readable identifier for this evaluation run.
        device_id: ADB device serial number for on-device evaluation, or
            ``None`` to run in simulation mode without latency measurement.
        params_path: Path to params.yaml (defaults to ``"params.yaml"``).

    Returns:
        A frozen :class:`GateResult` instance.
    """
    # 1. Load params
    with open(params_path) as fh:
        params: dict[str, Any] = yaml.safe_load(fh)

    vlm_gates: dict[str, Any] = params["gates"]["vlm"]
    wandb_cfg: dict[str, Any] = params["wandb"]
    inference_cfg: dict[str, Any] = params.get("inference", {})
    schema_version: str = str(inference_cfg.get("schema_version", "1.0"))

    # 2. Determine device availability
    has_device = device_id is not None

    logger.info(
        "run_eval_quantized: starting",
        extra={"model_dir": model_dir, "has_device": has_device, "device_id": device_id},
    )

    # 3. Run quantized inference
    inference_result = _run_quantized_inference(model_dir, device_id, params)
    outputs: list[str] = inference_result.get("outputs", [])
    timings: list[float] = inference_result.get("timings", [])
    fp16_outputs: list[Any] = inference_result.get("fp16_outputs", [])

    # 4. Schema compliance — always run
    results = compute_schema_compliance(
        outputs,
        compliance_threshold=vlm_gates.get("schema_compliance"),
        sum_error_threshold=vlm_gates.get("distribution_sum_error"),
        schema_version=schema_version,
    )

    # 5. Latency — only when device present and timings available
    skipped: list[str] = []

    if has_device and timings:
        latency_result = compute_latency(
            timings,
            threshold=vlm_gates.get("latency_p95_ms"),
        )
        results.append(latency_result)
    else:
        skipped.append("latency_p95_ms")

    # 6. KL divergence — only when fp16 reference outputs are available
    if not fp16_outputs:
        skipped.append("kl_divergence")

    # 7. Always skip gold-set-dependent metrics
    skipped.extend(_GOLD_SET_METRICS)

    # 8. Gate check
    gate_result = check_gate(results, skipped, "vlm", params)

    # 9. Report
    metadata: dict[str, Any] = {
        "model_dir": model_dir,
        "params_path": params_path,
        "has_device": has_device,
        "device_id": device_id,
        "n_outputs": len(outputs),
        "n_timings": len(timings),
        "has_fp16_outputs": bool(fp16_outputs),
    }
    generate_report(gate_result, run_name, "vlm_quantized", metadata, wandb_cfg)

    return gate_result


def main() -> None:
    """CLI entry point for the eval_quantized runner.

    Parses arguments, runs evaluation, and exits with code 0 on pass or 1 on
    fail.
    """
    setup_logging("pet-eval")

    parser = argparse.ArgumentParser(
        description="Evaluate a quantized VLM model against the pet-eval benchmark."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the quantized model directory.",
    )
    parser.add_argument(
        "--run_name",
        required=True,
        help="Short identifier for this evaluation run (used in W&B).",
    )
    parser.add_argument(
        "--device_id",
        default=None,
        help="ADB device serial number for on-device evaluation (optional).",
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        dest="params_path",
        help="Path to params.yaml (default: params.yaml).",
    )
    args = parser.parse_args()

    result = run_eval_quantized(
        model_dir=args.model_dir,
        run_name=args.run_name,
        device_id=args.device_id,
        params_path=args.params_path,
    )

    logger.info(
        "eval_quantized finished",
        extra={"passed": result.passed, "summary": result.summary},
    )
    sys.exit(0 if result.passed else 1)
