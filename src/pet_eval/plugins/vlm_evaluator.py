"""VLMEvaluator plugin — EVALUATORS-registered adapter for VLM gold-set evaluation."""

from __future__ import annotations

import logging
from typing import Any

from pet_infra.registry import EVALUATORS, METRICS
from pet_schema.model_card import ModelCard

from pet_eval.plugins.gate import apply_gate
from pet_eval.plugins.vlm_inference import run_inference

log = logging.getLogger(__name__)


@EVALUATORS.register_module(name="vlm_evaluator")
class VLMEvaluator:
    """Evaluate a VLM checkpoint against a gold set, compute metrics, apply gate.

    Expected config keys (via Registry.build kwargs):
      - metrics: list[str] — metric names to build via METRICS.build
      - thresholds: dict[str, float] — min_<name>/max_<name> thresholds for apply_gate
      - gold_set_path: str — path to gold set JSONL
      - params: dict — inference/benchmark config passed to run_inference
    Optional: schema_version, anomaly_set_path, teacher_reference_path.
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
        self._gold_set_path: str | None = cfg.get("gold_set_path")
        self._params: dict[str, Any] = cfg.get("params", {})

    def run(self, input_card: ModelCard | None, recipe: Any) -> ModelCard:
        """Run evaluation and return an updated ModelCard with metrics + gate_status.

        Args:
            input_card: ModelCard from the preceding training stage.  Must not
                be None; the evaluator needs the checkpoint_uri to locate the
                model weights.
            recipe: Recipe object forwarded by the orchestrator (not used
                directly; kept for interface compatibility).

        Returns:
            Updated ModelCard with metrics merged and gate_status set.

        Raises:
            ValueError: If input_card is None.
        """
        if input_card is None:
            raise ValueError(
                "VLMEvaluator.run requires a trained model_card; got None. "
                "Orchestrator should pass the SFT/DPO stage's output card as prev_card."
            )

        model_path = self._cfg.get("model_path") or input_card.checkpoint_uri.replace("file://", "")
        outputs = run_inference(
            model_path=model_path,
            gold_set_path=self._gold_set_path,
            params=self._params,
        )

        metrics_out: dict[str, float] = self._compute_metrics(outputs)
        gate = apply_gate(metrics_out, self._thresholds)

        updated = input_card.metrics.copy()
        updated.update(metrics_out)

        return input_card.model_copy(
            update={
                "metrics": updated,
                "gate_status": "passed" if gate.passed else "failed",
                "task": "vlm_eval",
                "notes": gate.reason if not gate.passed else input_card.notes,
            }
        )

    def _compute_metrics(self, outputs: list[str]) -> dict[str, float]:
        """Compute each configured metric over VLM outputs.

        The actual metric invocation contract varies per metric (some take
        (predicted, actual), some take (outputs, schema_version)). For Phase 3A
        P2-C we invoke with the kwargs stored in the registry-built adapter —
        a fuller integration (feeding gold_set references, per-metric arg
        marshalling) is explicitly deferred to a follow-up.

        For now: call each metric with outputs as the single positional arg
        and unpack list[MetricResult] into dict[name -> value]. Metrics whose
        signature doesn't match that pattern log a warning and skip.

        Args:
            outputs: List of raw VLM output strings from run_inference.

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
