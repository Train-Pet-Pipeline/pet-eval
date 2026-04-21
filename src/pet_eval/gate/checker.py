"""Gate checker for pet-eval.

Aggregates a list of MetricResult objects into a GateResult and logs the outcome.
Thresholds are embedded in each MetricResult at creation time by the metric runners;
this module purely aggregates and reports.
"""

from __future__ import annotations

import logging
from typing import Any

from pet_eval.gate.types import GateResult
from pet_eval.metrics.types import MetricResult

logger = logging.getLogger(__name__)


def check_gate(
    results: list[MetricResult],
    skipped: list[str],
    gate_type: str,
    params: dict[str, Any],
) -> GateResult:
    """Aggregate MetricResult objects into a GateResult and log the outcome.

    Thresholds are already embedded in each MetricResult (applied at metric
    creation time in runners).  This function simply builds the aggregate
    GateResult via :meth:`GateResult.from_results` and emits a structured log
    entry.

    Args:
        results: All computed MetricResult objects for this gate check.
        skipped: Names of metrics that were skipped during evaluation.
        gate_type: Label identifying the gate type (e.g. ``"vlm"`` or ``"audio"``).
        params: Full params dict (passed through for context; not used internally).

    Returns:
        A frozen :class:`GateResult` instance.
    """
    gate = GateResult.from_results(results, skipped)

    logger.info(
        "check_gate",
        extra={
            "gate_type": gate_type,
            "passed": gate.passed,
            "n_results": len(results),
            "n_skipped": len(skipped),
            "summary": gate.summary,
        },
    )

    return gate
