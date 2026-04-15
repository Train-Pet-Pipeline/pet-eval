"""Latency metric (P50 / P95 / P99) for pet-eval.

Computes P95 as the primary gate value.  P50, P99, min, max, mean, and
n_samples are surfaced in ``details``.  The metric uses linear interpolation
(nearest-rank with interpolation) and is gated with operator ``lte``
(lower latency is better).
"""
from __future__ import annotations

import logging
import statistics

from pet_eval.metrics.types import MetricResult

logger = logging.getLogger(__name__)


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute a percentile with linear interpolation on sorted data.

    Uses the same formula as numpy's ``percentile`` with
    ``interpolation='linear'``.

    Args:
        sorted_data: Pre-sorted list of floats (ascending).
        pct: Percentile in [0, 100].

    Returns:
        Interpolated percentile value.
    """
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]

    # Map percentile to a 0-based index
    virtual_index = (pct / 100.0) * (n - 1)
    lo = int(virtual_index)
    hi = lo + 1
    frac = virtual_index - lo

    if hi >= n:
        return sorted_data[-1]

    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


def compute_latency(
    timings_ms: list[float],
    *,
    threshold: float | None = None,
) -> MetricResult:
    """Compute latency percentile statistics and gate on P95.

    Sorts ``timings_ms``, computes P50 / P95 / P99 via linear interpolation,
    and returns a :class:`MetricResult` with ``name="latency_p95_ms"`` and
    ``operator="lte"``.  Empty inputs return ``value=0.0``.

    Args:
        timings_ms: List of end-to-end inference latencies in milliseconds.
        threshold: Optional gate threshold.  When provided, ``passed`` is
            ``True`` iff ``P95 <= threshold``.

    Returns:
        A single :class:`MetricResult` with P95 as the primary value and
        ``details`` containing p50, p99, min, max, mean, and n_samples.
    """
    if not timings_ms:
        logger.info("compute_latency: empty timings, returning value=0.0")
        return MetricResult.create(
            "latency_p95_ms",
            value=0.0,
            threshold=threshold,
            operator="lte",
            details={"p50": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "n_samples": 0},
        )

    sorted_data = sorted(timings_ms)
    p50 = _percentile(sorted_data, 50)
    p95 = _percentile(sorted_data, 95)
    p99 = _percentile(sorted_data, 99)
    mean = statistics.mean(timings_ms)

    logger.info(
        "compute_latency",
        extra={
            "n_samples": len(sorted_data),
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "min": sorted_data[0],
            "max": sorted_data[-1],
            "mean": mean,
        },
    )

    return MetricResult.create(
        "latency_p95_ms",
        value=p95,
        threshold=threshold,
        operator="lte",
        details={
            "p50": p50,
            "p99": p99,
            "min": sorted_data[0],
            "max": sorted_data[-1],
            "mean": mean,
            "n_samples": len(sorted_data),
        },
    )
