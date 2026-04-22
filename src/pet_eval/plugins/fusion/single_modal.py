"""SingleModalFusionEvaluator — pass-through fusion for one named modality."""

from __future__ import annotations

from pet_infra.registry import EVALUATORS

from .base import BaseFusionEvaluator


@EVALUATORS.register_module(name="single_modal_fusion", force=True)
class SingleModalFusionEvaluator(BaseFusionEvaluator):
    """Pass-through fusion: returns the score for one named modality."""

    def __init__(self, modality: str, **_: object) -> None:
        """Initialise with the modality to pass through.

        Args:
            modality: The key in modality_scores to return.
            **_: Extra kwargs ignored for forward-compatibility.
        """
        self._modality = modality

    def fuse(self, modality_scores: dict[str, float]) -> float:
        """Return the score for the configured modality.

        Args:
            modality_scores: Mapping from modality name to its float score.

        Returns:
            The score for the configured modality.

        Raises:
            KeyError: If the configured modality is absent from modality_scores.
        """
        return modality_scores[self._modality]  # KeyError if absent (intentional fail-fast)
