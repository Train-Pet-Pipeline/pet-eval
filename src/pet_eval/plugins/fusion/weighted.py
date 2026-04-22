"""WeightedFusionEvaluator — weighted sum with normalized weights."""

from __future__ import annotations

from pet_infra.registry import EVALUATORS

from .base import BaseFusionEvaluator


@EVALUATORS.register_module(name="weighted_fusion", force=True)
class WeightedFusionEvaluator(BaseFusionEvaluator):
    """Weighted sum (weights normalized to sum=1); missing modalities treated as 0."""

    def __init__(self, weights: dict[str, float], **_: object) -> None:
        """Initialise with per-modality weights; normalizes to sum=1.

        Args:
            weights: Mapping from modality name to its relative weight.
                     Must have positive total.
            **_: Extra kwargs ignored for forward-compatibility.

        Raises:
            ValueError: If the sum of weights is <= 0.
        """
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("WeightedFusionEvaluator requires positive weight sum")
        self._weights = {k: v / total for k, v in weights.items()}

    def fuse(self, modality_scores: dict[str, float]) -> float:
        """Return the normalized weighted sum of modality scores.

        Args:
            modality_scores: Mapping from modality name to its float score.
                             Missing modalities are treated as 0.0.

        Returns:
            Weighted sum of scores (weights already normalized to sum=1).
        """
        return sum(
            self._weights.get(m, 0.0) * modality_scores.get(m, 0.0) for m in self._weights
        )
