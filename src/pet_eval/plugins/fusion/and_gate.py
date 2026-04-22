"""AndGateFusionEvaluator — all modalities must clear threshold."""

from __future__ import annotations

from pet_infra.registry import EVALUATORS

from .base import BaseFusionEvaluator


@EVALUATORS.register_module(name="and_gate_fusion", force=True)
class AndGateFusionEvaluator(BaseFusionEvaluator):
    """All modalities must clear threshold; returns min when they do, else 0."""

    def __init__(self, threshold: float, **_: object) -> None:
        """Initialise with the minimum passing threshold.

        Args:
            threshold: Each modality score must be >= this value to pass.
            **_: Extra kwargs ignored for forward-compatibility.
        """
        self._threshold = threshold

    def fuse(self, modality_scores: dict[str, float]) -> float:
        """Return min score if all modalities pass, else 0.0.

        Args:
            modality_scores: Mapping from modality name to its float score.

        Returns:
            min(scores) if all scores >= threshold, else 0.0.

        Raises:
            ValueError: If modality_scores is empty.
        """
        if not modality_scores:
            raise ValueError("AndGateFusionEvaluator requires at least one modality score")
        if all(s >= self._threshold for s in modality_scores.values()):
            return min(modality_scores.values())
        return 0.0
