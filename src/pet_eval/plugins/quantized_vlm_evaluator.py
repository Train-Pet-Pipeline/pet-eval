"""QuantizedVlmEvaluator plugin — EVALUATORS-registered adapter for RKLLM eval.

Delegates inference to quantized_vlm_inference.run_inference (which lazy-imports
RKLLMRunner) and delegates metric computation to the METRICS registry, mirroring
the VLMEvaluator → vlm_inference pattern from Phase 3A.

Module-load does NOT require rkllm SDK to be installed.
"""

from __future__ import annotations

import logging
from typing import Any

from pet_infra.registry import EVALUATORS, METRICS
from pet_schema.model_card import ModelCard

from pet_eval.plugins.gate import apply_gate
from pet_eval.plugins.quantized_vlm_inference import run_inference

log = logging.getLogger(__name__)


@EVALUATORS.register_module(name="quantized_vlm_evaluator", force=True)
class QuantizedVlmEvaluator:
    """Evaluate a quantized RKLLM artifact; emit metrics + gate status.

    Expected config keys (via Registry.build kwargs):
      - metrics: list[str] — metric names to build via METRICS.build; supports
        any name registered in METRICS (e.g. "schema_compliance", "latency")
      - thresholds: dict[str, float] — min_*/max_* thresholds for apply_gate
      - gate_tier: optional preset tier name (strict|balanced|permissive)
        — see pet_eval.plugins.gate_tiers
      - target: str — RK platform (default "rk3576")
      - eval_set_uri: str | None — path to JSONL eval set forwarded to run_inference
      - params: dict — inference config passed to run_inference
        (e.g. {"inference": {"max_new_tokens": 2048}})
    """

    def __init__(self, **cfg: Any) -> None:
        """Initialise evaluator from registry build kwargs.

        Args:
            **cfg: Keyword arguments from Registry.build.  See class docstring
                for the expected keys.
        """
        self._cfg: dict[str, Any] = dict(cfg)
        metric_names: list[str] = cfg.get("metrics", [])
        self._metrics: list[Any] = [METRICS.build({"type": name}) for name in metric_names]
        self._metric_names: list[str] = metric_names
        self._thresholds: dict[str, float] = cfg.get("thresholds", {})
        self._gate_tier: str | None = cfg.get("gate_tier")
        self._target: str = cfg.get("target", "rk3576")
        self._eval_set_uri: str | None = cfg.get("eval_set_uri")
        self._params: dict[str, Any] = cfg.get("params", {})

    def run(self, input_card: ModelCard | None, recipe: Any) -> ModelCard:
        """Run inference on the RKLLM artifact and return an updated ModelCard.

        Args:
            input_card: ModelCard that must contain at least one EdgeArtifact
                with format='rkllm'.
            recipe: Recipe object forwarded by the orchestrator (not used
                directly; kept for interface compatibility).

        Returns:
            Updated ModelCard with metrics merged and gate_status set.

        Raises:
            ValueError: If input_card is None or has no rkllm edge artifact.
        """
        if input_card is None:
            raise ValueError("QuantizedVlmEvaluator.run requires a non-None input_card")

        rkllm_artifacts = [a for a in input_card.edge_artifacts if a.format == "rkllm"]
        if not rkllm_artifacts:
            raise ValueError(
                "QuantizedVlmEvaluator requires a card with an edge_artifact "
                "of format='rkllm'; got none."
            )

        outputs = run_inference(
            model_path=rkllm_artifacts[0].artifact_uri,
            eval_set_path=self._eval_set_uri,
            target=self._target,
            params=self._params,
        )

        metrics_out: dict[str, float] = self._compute_metrics(outputs)
        gate = apply_gate(metrics_out, self._thresholds, tier=self._gate_tier)

        merged_metrics = input_card.metrics.copy()
        merged_metrics.update(metrics_out)

        return input_card.model_copy(
            update={
                "metrics": merged_metrics,
                "gate_status": "passed" if gate.passed else "failed",
                "task": "quantized_vlm_eval",
                "notes": gate.reason if not gate.passed else input_card.notes,
            }
        )

    def _compute_metrics(self, outputs: list[str]) -> dict[str, float]:
        """Compute each configured metric over RKLLM outputs.

        The actual metric invocation contract varies per metric (some take
        (predicted, actual), some take (outputs, schema_version)). For Phase 3B
        P3-A we invoke with the outputs as the single positional arg and unpack
        list[MetricResult] into dict[name -> value]. Metrics whose signature
        doesn't match that pattern log a warning and skip.

        Args:
            outputs: List of raw RKLLM output strings from run_inference.

        Returns:
            Dict mapping metric name to float value.
        """
        results: dict[str, float] = {}
        for name, metric in zip(self._metric_names, self._metrics, strict=True):
            try:
                out = metric(outputs)
            except TypeError as e:
                log.warning("metric %s skipped: signature mismatch (%s)", name, e)
                continue
            for mr in out if isinstance(out, list) else [out]:
                if hasattr(mr, "name") and hasattr(mr, "value"):
                    results[mr.name] = float(mr.value)
                else:
                    results[name] = float(mr)
        return results
