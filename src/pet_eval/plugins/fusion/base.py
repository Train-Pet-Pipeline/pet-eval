"""Abstract base for rule-based cross-modal fusion evaluators (Phase 4 W2, spec §4.5/§7.6)."""

from __future__ import annotations

from abc import ABC, abstractmethod


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
