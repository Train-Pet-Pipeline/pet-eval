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
from pet_infra.logging import setup_logging

from pet_eval.gate.checker import check_gate
from pet_eval.gate.types import GateResult
from pet_eval.metrics.schema_compliance import compute_schema_compliance
from pet_eval.report.generate_report import generate_report

logger = logging.getLogger(__name__)


def _load_model(model_path: str, params: dict[str, Any]) -> tuple[Any, Any]:
    """Load base model with LoRA adapter merged for inference.

    Reads adapter_config.json from model_path to determine the base model,
    then loads and merges the LoRA adapter.

    Args:
        model_path: Path to the trained LoRA adapter directory.
        params: Full params dict (used for inference config).

    Returns:
        Tuple of (model, processor) ready for inference.

    Raises:
        FileNotFoundError: If model_path or adapter_config.json does not exist.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    adapter_dir = Path(model_path)
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {model_path}")

    with open(adapter_config_path) as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg.get("base_model_name_or_path", "")

    inference_cfg = params.get("inference", {})
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    logger.info(
        "_load_model",
        extra={"base_model": base_model_name, "adapter": model_path, "device": device},
    )

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)

    # Detect VLM models that need specialized loader instead of AutoModelForCausalLM
    config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    if model_type in ("qwen2_vl", "qwen2-vl"):
        from transformers import Qwen2VLForConditionalGeneration
        model_cls = Qwen2VLForConditionalGeneration
    else:
        model_cls = AutoModelForCausalLM

    base_model = model_cls.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()

    if device in ("mps", "cpu"):
        model = model.to(device)

    model.eval()
    return model, processor


def _run_inference(
    model_path: str,
    gold_set_path: str | None,
    params: dict[str, Any],
) -> list[str]:
    """Run VLM inference over the gold set and return raw output strings.

    Loads the model with LoRA adapter merged, processes each gold set record
    (image + prompt), and returns the model's raw JSON output strings.

    Args:
        model_path: Path to the trained model checkpoint directory.
        gold_set_path: Path to the JSONL gold set file, or None if unavailable.
        params: Full params dict (used for inference config).

    Returns:
        List of raw JSON strings produced by the model.  Returns an empty list
        when *gold_set_path* is None.
    """
    if gold_set_path is None:
        return []

    records: list[dict[str, Any]] = []
    with open(gold_set_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return []

    logger.info(
        "_run_inference: loaded gold set",
        extra={"n_records": len(records), "model_path": model_path},
    )

    import torch
    from PIL import Image

    inference_cfg = params.get("inference", {})
    max_new_tokens = inference_cfg.get("max_new_tokens", 1024)

    model, processor = _load_model(model_path, params)
    outputs: list[str] = []

    for i, record in enumerate(records):
        image_path = record.get("image", record.get("images", [""])[0] if record.get("images") else "")
        prompt_text = record.get("prompt", record.get("instruction", ""))
        system_text = record.get("system", "")

        messages: list[dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})

        user_content: list[dict[str, str]] = []
        if image_path and Path(image_path).exists():
            user_content.append({"type": "image", "image": f"file://{image_path}"})
        user_content.append({"type": "text", "text": prompt_text})
        messages.append({"role": "user", "content": user_content})

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if image_path and Path(image_path).exists():
            from qwen_vl_utils import process_vision_info

            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs or None,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(text=[text], padding=True, return_tensors="pt")

        inputs = inputs.to(model.device)

        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_trimmed = generated[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(generated_trimmed, skip_special_tokens=True)[0]
        outputs.append(output_text.strip())

        if (i + 1) % 10 == 0:
            logger.info(
                "_run_inference progress",
                extra={"completed": i + 1, "total": len(records)},
            )

    logger.info(
        "_run_inference: completed",
        extra={"n_outputs": len(outputs), "model_path": model_path},
    )
    return outputs


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
    params_dir = Path(params_path).resolve().parent
    with open(params_path) as fh:
        params: dict[str, Any] = yaml.safe_load(fh)

    vlm_gates: dict[str, Any] = params["gates"]["vlm"]
    benchmark_cfg: dict[str, Any] = params["benchmark"]
    wandb_cfg: dict[str, Any] = params["wandb"]
    inference_cfg: dict[str, Any] = params.get("inference", {})
    schema_version: str = str(inference_cfg.get("schema_version", "1.0"))

    def _resolve(p: str) -> str:
        """Resolve path relative to params.yaml directory."""
        path = Path(p)
        if not path.is_absolute():
            path = params_dir / path
        return str(path)

    gold_set_path: str = _resolve(benchmark_cfg["gold_set_path"])
    anomaly_set_path: str = _resolve(benchmark_cfg["anomaly_set_path"])
    teacher_ref_path: str = _resolve(benchmark_cfg.get("teacher_reference_path", "")) if benchmark_cfg.get("teacher_reference_path") else ""

    # 2. Check existence and content
    gold_path_obj = Path(gold_set_path)
    has_gold_set = gold_path_obj.exists() and gold_path_obj.stat().st_size > 0

    anomaly_path_obj = Path(anomaly_set_path)
    has_anomaly_set = anomaly_path_obj.exists() and anomaly_path_obj.stat().st_size > 0

    teacher_path_obj = Path(teacher_ref_path) if teacher_ref_path else None
    has_teacher_references = (
        teacher_path_obj is not None
        and teacher_path_obj.exists()
        and teacher_path_obj.stat().st_size > 0
    )

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
    if not has_teacher_references:
        logger.warning(
            "Teacher references not found or empty; skipping mood/narrative metrics",
            extra={"teacher_reference_path": teacher_ref_path},
        )

    # 3. Run inference (returns [] when gold set absent)
    outputs = _run_inference(
        model_path,
        gold_set_path if has_gold_set else None,
        params,
    )

    # 4. Schema compliance — run when outputs exist, skip when no gold set
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

    if not has_teacher_references:
        skipped.extend(["mood_spearman", "narrative_bertscore"])

    if not has_gold_set:
        skipped.extend(
            [
                "compliance_rate",
                "distribution_sum_error",
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
    setup_logging("pet-eval")

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


if __name__ == "__main__":
    main()
