"""Audio eval runner for pet-eval.

Evaluates an audio CNN model checkpoint against a benchmark test directory,
computes audio accuracy and vomit recall, gates the result, and publishes a
W&B report.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torchaudio
import yaml
from pet_infra.device import detect_device
from pet_infra.logging import setup_logging

from pet_eval.gate.checker import check_gate
from pet_eval.gate.types import GateResult
from pet_eval.metrics.audio_accuracy import compute_audio_accuracy
from pet_eval.report.generate_report import generate_report

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def _run_audio_inference(
    model_path: str,
    params: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Run audio CNN inference over the test directory and return (predicted, actual).

    Loads the trained MobileNetV2AudioSet model from *model_path*, iterates
    over ``benchmark.audio_test_dir/{class_name}/*.wav`` and runs inference on
    each file.  The ground-truth label is derived from the subdirectory name.

    Args:
        model_path: Path to the trained audio CNN checkpoint file.
        params: Full params dict, used to look up ``benchmark.audio_test_dir``
            and ``audio.classes``.

    Returns:
        A tuple ``(predicted, actual)`` of parallel string label lists.
        Returns ``([], [])`` when the test directory is absent or unavailable.
    """
    audio_test_dir: str = params.get("benchmark", {}).get("audio_test_dir", "")

    if not audio_test_dir:
        logger.warning(
            "audio_test_dir not configured; skipping audio inference",
            extra={"model_path": model_path},
        )
        return ([], [])

    test_dir_path = Path(audio_test_dir)
    if not test_dir_path.exists():
        logger.warning(
            "audio_test_dir does not exist; skipping audio inference",
            extra={"model_path": model_path, "audio_test_dir": audio_test_dir},
        )
        return ([], [])

    classes: list[str] = params.get("audio", {}).get("classes", [])
    sample_rate: int = params.get("audio", {}).get("sample_rate", 16000)
    max_duration_sec: float = params.get("audio", {}).get("max_duration_sec", 5.0)
    max_samples = int(sample_rate * max_duration_sec)

    if not classes:
        logger.error("audio.classes not configured in params")
        return ([], [])

    # Load model
    from pet_train.audio_model_arch import MobileNetV2AudioSet

    model = MobileNetV2AudioSet(num_classes=len(classes), sample_rate=sample_rate)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Determine device
    device = torch.device(detect_device())
    model = model.to(device)

    predicted: list[str] = []
    actual: list[str] = []

    for cls_name in classes:
        cls_dir = test_dir_path / cls_name
        if not cls_dir.exists():
            continue
        for audio_file in sorted(cls_dir.iterdir()):
            if audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            try:
                waveform, sr = torchaudio.load(str(audio_file))
                if sr != sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if waveform.shape[1] > max_samples:
                    waveform = waveform[:, :max_samples]
                elif waveform.shape[1] < max_samples:
                    pad = max_samples - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad))

                waveform = waveform.squeeze(0).unsqueeze(0).to(device)  # [1, samples]
                with torch.no_grad():
                    logits = model(waveform)
                pred_idx = logits.argmax(dim=1).item()
                predicted.append(classes[pred_idx])
                actual.append(cls_name)
            except Exception:
                logger.warning(
                    "Failed to process audio file",
                    extra={"file": str(audio_file)},
                    exc_info=True,
                )

    logger.info(
        "Audio inference complete",
        extra={"n_samples": len(predicted), "model_path": model_path},
    )
    return (predicted, actual)


def run_eval_audio(
    model_path: str,
    run_name: str,
    params_path: str = "params.yaml",
) -> GateResult:
    """Evaluate an audio CNN checkpoint and gate its quality.

    Steps:
      1. Load params.yaml — extract gates.audio thresholds, wandb config, and
         audio.classes.
      2. Call ``_run_audio_inference`` to obtain ``(predicted, actual)`` label
         lists.
      3. If no results: skip audio metrics, return gate with skipped metrics.
      4. Compute ``compute_audio_accuracy`` with thresholds from params.
      5. Call ``check_gate`` to aggregate results.
      6. Call ``generate_report`` to publish to W&B.

    Args:
        model_path: Path to the trained audio CNN checkpoint.
        run_name: Short human-readable identifier for this evaluation run.
        params_path: Path to params.yaml (defaults to ``"params.yaml"``).

    Returns:
        A frozen :class:`GateResult` instance.
    """
    # 1. Load params
    with open(params_path) as fh:
        params: dict[str, Any] = yaml.safe_load(fh)

    audio_gates: dict[str, Any] = params["gates"]["audio"]
    wandb_cfg: dict[str, Any] = params["wandb"]
    classes: list[str] = params["audio"]["classes"]

    accuracy_threshold: float | None = audio_gates.get("overall_accuracy")
    vomit_recall_threshold: float | None = audio_gates.get("vomit_recall")

    # 2. Run inference
    predicted, actual = _run_audio_inference(model_path, params)

    # 3. If no results: skip audio metrics and return gate with skipped list
    if not predicted:
        skipped = ["audio_overall_accuracy", "audio_vomit_recall"]
        logger.warning(
            "No audio test data available; all audio metrics skipped",
            extra={"model_path": model_path},
        )
        gate_result = check_gate([], skipped, "audio", params)
        metadata: dict[str, Any] = {
            "model_path": model_path,
            "params_path": params_path,
            "n_samples": 0,
        }
        generate_report(gate_result, run_name, "audio", metadata, wandb_cfg)
        return gate_result

    # 4. Compute audio accuracy metrics
    results = compute_audio_accuracy(
        predicted,
        actual,
        classes,
        accuracy_threshold=accuracy_threshold,
        vomit_recall_threshold=vomit_recall_threshold,
    )

    # 5. Gate check
    gate_result = check_gate(results, [], "audio", params)

    # 6. Report
    metadata = {
        "model_path": model_path,
        "params_path": params_path,
        "n_samples": len(predicted),
    }
    generate_report(gate_result, run_name, "audio", metadata, wandb_cfg)

    return gate_result


def main() -> None:
    """CLI entry point for the eval_audio runner.

    Parses arguments, runs evaluation, and exits with code 0 on pass or 1 on
    fail.
    """
    setup_logging("pet-eval")

    parser = argparse.ArgumentParser(
        description="Evaluate an audio CNN checkpoint against the pet-eval benchmark."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the trained audio CNN checkpoint.",
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

    result = run_eval_audio(
        model_path=args.model_path,
        run_name=args.run_name,
        params_path=args.params_path,
    )

    logger.info(
        "eval_audio finished",
        extra={"passed": result.passed, "summary": result.summary},
    )
    sys.exit(0 if result.passed else 1)
