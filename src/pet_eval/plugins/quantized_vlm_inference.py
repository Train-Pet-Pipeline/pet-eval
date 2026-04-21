"""RKLLM inference helpers for pet-eval QuantizedVlmEvaluator plugin (Phase 3B).

Extracted from QuantizedVlmEvaluator so that the evaluator plugin and future
plugins can reuse the RKLLM inference primitives without the evaluator class
coupling to the RKLLMRunner lifecycle.

Public API:
  - run_inference(model_path, eval_set_path, target, params) -> list[str]

Module-load does NOT import pet_quantize.inference.rkllm_runner; the import
is deferred inside run_inference (lazy import pattern).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["run_inference"]


def run_inference(
    model_path: str,
    eval_set_path: str | None,
    target: str,
    params: dict[str, Any],
) -> list[str]:
    """Run RKLLM inference over the eval set and return raw output strings.

    Lifecycle: constructs RKLLMRunner, calls init(), iterates eval records
    calling generate() per record, always calls release() (in finally).
    Returns empty list when eval_set_path is None or records are empty.

    Args:
        model_path: Path to the RKLLM model artifact (.rkllm file).
        eval_set_path: Path to the JSONL eval set file, or None if unavailable.
            Each line must be a JSON object with at least a ``prompt`` key.
        target: RK platform string (e.g. "rk3576") forwarded to RKLLMRunner.
        params: Full params dict; ``params["inference"]["max_new_tokens"]``
            controls generation length (default 2048).

    Returns:
        List of raw output strings produced by the model.  Returns an empty
        list when *eval_set_path* is None or the eval set contains no records.
    """
    if eval_set_path is None:
        return []

    records: list[dict[str, Any]] = []
    with open(eval_set_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return []

    logger.info(
        "run_inference: loaded eval set",
        extra={"n_records": len(records), "model_path": model_path, "target": target},
    )

    max_tokens: int = params.get("inference", {}).get("max_new_tokens", 2048)

    from pet_quantize.inference.rkllm_runner import RKLLMRunner  # lazy import

    runner = RKLLMRunner(model_path=model_path, target=target)
    outputs: list[str] = []
    try:
        runner.init()
        for i, record in enumerate(records):
            prompt: str = record.get("prompt", "")
            text, _latency = runner.generate(
                prompt=prompt,
                visual_features=None,
                max_tokens=max_tokens,
            )
            outputs.append(text)

            if (i + 1) % 10 == 0:
                logger.info(
                    "run_inference progress",
                    extra={"completed": i + 1, "total": len(records)},
                )
    finally:
        runner.release()

    logger.info(
        "run_inference: completed",
        extra={"n_outputs": len(outputs), "model_path": model_path},
    )
    return outputs
