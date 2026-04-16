"""Core GateResult data type for pet-eval gate checking."""
from __future__ import annotations

from dataclasses import dataclass

from pet_eval.metrics.types import MetricResult


@dataclass(frozen=True)
class GateResult:
    """Immutable aggregate result of a gate check over multiple metrics.

    Attributes:
        passed: True when all gated metrics (threshold is not None) passed.
        results: All MetricResult instances (gated and informational).
        skipped: Names of metrics that were skipped during evaluation.
        summary: Human-readable one-line verdict string.
    """

    passed: bool
    results: list[MetricResult]
    skipped: list[str]
    summary: str

    @classmethod
    def from_results(
        cls,
        results: list[MetricResult],
        skipped: list[str],
    ) -> GateResult:
        """Build a GateResult from a list of MetricResult objects.

        Only gated metrics (those with ``threshold is not None``) contribute to
        the pass/fail decision.  Informational metrics are included in
        ``results`` but do not affect ``passed``.

        Args:
            results: All computed MetricResult objects.
            skipped: Names of metrics that were skipped.

        Returns:
            A frozen :class:`GateResult` instance.
        """
        skipped_set = set(skipped)
        gated = [r for r in results if r.threshold is not None and r.name not in skipped_set]
        failed = [r for r in gated if not r.passed]
        all_pass = len(failed) == 0

        n_skipped = len(skipped)
        if all_pass:
            summary = (
                f"PASS — {len(gated)} gated metrics passed, {n_skipped} skipped"
            )
        else:
            failed_names = ", ".join(r.name for r in failed)
            summary = f"FAIL — failed: {failed_names}"

        return cls(
            passed=all_pass,
            results=results,
            skipped=skipped,
            summary=summary,
        )
