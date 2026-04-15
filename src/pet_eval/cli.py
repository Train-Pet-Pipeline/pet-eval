"""Unified CLI entry point for pet-eval.

Provides sub-commands for evaluating trained VLM checkpoints, audio CNN
checkpoints, and quantized VLM models.
"""
from __future__ import annotations

import argparse
import logging
import sys

from pet_eval.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Unified CLI entry point for pet-eval with sub-commands.

    Sub-commands:
        eval-trained   — Evaluate a trained VLM checkpoint.
        eval-audio     — Evaluate a trained audio CNN checkpoint.
        eval-quantized — Evaluate a quantized VLM model directory.

    Exits with code 0 when the gate passes, 1 when it fails.
    """
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="pet-eval",
        description="pet-eval: evaluation pipeline for Train-Pet-Pipeline models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── eval-trained ────────────────────────────────────────────────────────
    trained_parser = subparsers.add_parser(
        "eval-trained",
        help="Evaluate a trained VLM checkpoint against the pet-eval benchmark.",
    )
    trained_parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the trained model checkpoint directory.",
    )
    trained_parser.add_argument(
        "--run_name",
        required=True,
        help="Short identifier for this evaluation run (used in W&B).",
    )
    trained_parser.add_argument(
        "--params",
        default="params.yaml",
        dest="params_path",
        help="Path to params.yaml (default: params.yaml).",
    )

    # ── eval-audio ──────────────────────────────────────────────────────────
    audio_parser = subparsers.add_parser(
        "eval-audio",
        help="Evaluate a trained audio CNN checkpoint against the pet-eval benchmark.",
    )
    audio_parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the trained audio CNN checkpoint.",
    )
    audio_parser.add_argument(
        "--run_name",
        required=True,
        help="Short identifier for this evaluation run (used in W&B).",
    )
    audio_parser.add_argument(
        "--params",
        default="params.yaml",
        dest="params_path",
        help="Path to params.yaml (default: params.yaml).",
    )

    # ── eval-quantized ──────────────────────────────────────────────────────
    quantized_parser = subparsers.add_parser(
        "eval-quantized",
        help="Evaluate a quantized VLM model directory against the pet-eval benchmark.",
    )
    quantized_parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the quantized model directory.",
    )
    quantized_parser.add_argument(
        "--run_name",
        required=True,
        help="Short identifier for this evaluation run (used in W&B).",
    )
    quantized_parser.add_argument(
        "--device_id",
        default=None,
        help="ADB device serial number for on-device evaluation (optional).",
    )
    quantized_parser.add_argument(
        "--params",
        default="params.yaml",
        dest="params_path",
        help="Path to params.yaml (default: params.yaml).",
    )

    args = parser.parse_args()

    if args.command == "eval-trained":
        from pet_eval.runners.eval_trained import run_eval_trained

        result = run_eval_trained(
            model_path=args.model_path,
            run_name=args.run_name,
            params_path=args.params_path,
        )
        logger.info(
            "eval-trained finished",
            extra={"passed": result.passed, "summary": result.summary},
        )
        sys.exit(0 if result.passed else 1)

    elif args.command == "eval-audio":
        from pet_eval.runners.eval_audio import run_eval_audio

        result = run_eval_audio(
            model_path=args.model_path,
            run_name=args.run_name,
            params_path=args.params_path,
        )
        logger.info(
            "eval-audio finished",
            extra={"passed": result.passed, "summary": result.summary},
        )
        sys.exit(0 if result.passed else 1)

    elif args.command == "eval-quantized":
        from pet_eval.runners.eval_quantized import run_eval_quantized

        result = run_eval_quantized(
            model_dir=args.model_dir,
            run_name=args.run_name,
            device_id=args.device_id,
            params_path=args.params_path,
        )
        logger.info(
            "eval-quantized finished",
            extra={"passed": result.passed, "summary": result.summary},
        )
        sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
