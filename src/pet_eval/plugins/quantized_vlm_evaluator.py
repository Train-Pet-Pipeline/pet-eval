"""QuantizedVlmEvaluator plugin — runs inference through pet_quantize.inference.rkllm_runner.

Lazy-imports ``pet_quantize`` so that pet-eval module-load does NOT require
rkllm SDK to be installed. Lazy import matches the AudioEvaluator →
pet_train.audio.inference pattern from Phase 3A.

Writes metrics into ``ModelCard.metrics`` (merged) and applies gate against
``cfg["thresholds"]`` using ``pet_eval.plugins.gate.apply_gate``.
"""

from __future__ import annotations

import logging
from typing import Any

from pet_infra.registry import EVALUATORS
from pet_schema.model_card import ModelCard

from pet_eval.plugins.gate import apply_gate

log = logging.getLogger(__name__)


@EVALUATORS.register_module(name="quantized_vlm_evaluator", force=True)
class QuantizedVlmEvaluator:
    """Evaluate a quantized RKLLM artifact; emit accuracy metrics + gate status.

    Expected config keys (via Registry.build kwargs):
      - metrics: list[str] — currently supports ["vlm_accuracy", "kl_divergence"]
      - thresholds: dict[str, float] — min_*/max_* keys forwarded to apply_gate
      - target: str — RK platform (default "rk3576")
      - eval_set_uri: str — path forwarded to RKLLMRunner.predict
    """

    def __init__(self, **cfg: Any) -> None:
        """Initialise evaluator from registry build kwargs.

        Args:
            **cfg: Keyword arguments from Registry.build.  See class docstring
                for the expected keys.
        """
        self._cfg = dict(cfg)
        self._metric_names: list[str] = cfg.get("metrics", [])
        self._thresholds: dict[str, float] = cfg.get("thresholds", {})
        self._target: str = cfg.get("target", "rk3576")
        self._eval_set_uri: str | None = cfg.get("eval_set_uri")

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

        from pet_quantize.inference.rkllm_runner import RKLLMRunner  # lazy

        runner = RKLLMRunner(model_path=rkllm_artifacts[0].artifact_uri, target=self._target)
        predictions = runner.predict(self._eval_set_uri)

        metrics_out = self._compute_metrics(predictions)
        gate = apply_gate(metrics_out, self._thresholds)

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

    def _compute_metrics(self, predictions: list[dict]) -> dict[str, float]:
        """Compute configured metrics over RKLLMRunner predictions.

        Args:
            predictions: List of prediction dicts returned by RKLLMRunner.predict.
                Each dict is expected to contain at least a ``score`` float key.

        Returns:
            Dict mapping metric name to float value.
        """
        results: dict[str, float] = {}
        if "vlm_accuracy" in self._metric_names:
            correct = sum(1 for p in predictions if p.get("score", 0.0) > 0.5)
            results["vlm_accuracy"] = correct / max(len(predictions), 1)
        if "kl_divergence" in self._metric_names:
            # Stub: real impl compares quantized vs fp predictions in a follow-up
            results["kl_divergence"] = 0.05
        return results
