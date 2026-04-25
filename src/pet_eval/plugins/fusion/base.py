"""Abstract base for rule-based cross-modal fusion evaluators (Phase 4 W2, spec §4.5/§7.6)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from pet_schema.model_card import ModelCard


class BaseFusionEvaluator(ABC):
    """Combine per-modality scores into a single fused score."""

    @abstractmethod
    def fuse(self, modality_scores: dict[str, float]) -> float:
        """Return the fused score; raise on invalid input per concrete policy.

        Args:
            modality_scores: Mapping from modality name to its float score.

        Returns:
            A single fused float score.
        """

    def run(self, input_card: ModelCard | None, recipe: Any) -> ModelCard:
        """F014 fix: orchestrator-compatible stage runner method.

        Extracts per-modality scores from input_card.metrics via the convention
        ``modality_score:<name>`` (or empty dict if no input_card or no matching
        keys), calls ``fuse()``, and returns a new ModelCard with the fused
        score in metrics["fused_score"].

        For empty input (no upstream eval card), returns a synthetic ModelCard
        with the fused score = 0.0 — useful as smoke-test that the registry
        wiring works end-to-end.
        """
        modality_scores: dict[str, float] = {}
        if input_card is not None:
            for k, v in (input_card.metrics or {}).items():
                if k.startswith("modality_score:"):
                    modality_scores[k.split(":", 1)[1]] = float(v)

        fused = self.fuse(modality_scores) if modality_scores else 0.0

        recipe_id = getattr(recipe, "recipe_id", "unknown")
        version = getattr(recipe, "schema_version", "0.0.0")
        return ModelCard(
            id="",
            version=str(version),
            modality="multimodal",
            task="fusion_eval",
            arch=type(self).__name__,
            training_recipe=recipe_id,
            recipe_id=recipe_id,
            hydra_config_sha="",
            git_shas={},
            dataset_versions={},
            checkpoint_uri=input_card.checkpoint_uri if input_card else "",
            metrics={"fused_score": fused, **(input_card.metrics if input_card else {})},
            trained_at=datetime.now(UTC),
            trained_by=type(self).__name__,
            gate_status="passed",
        )
