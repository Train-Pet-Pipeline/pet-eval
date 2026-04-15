"""Tests for latency P50/P95/P99 metric — TDD, tests written before implementation."""
from __future__ import annotations

import pytest

from pet_eval.metrics.latency import compute_latency
from pet_eval.metrics.types import MetricResult

# ---------------------------------------------------------------------------
# Test 1: normal distribution of 100 timings → P95 between 2900 and 3000 ms
# ---------------------------------------------------------------------------


def test_normal_timings() -> None:
    """100 evenly-spaced timings from 2000 to 2990 ms → P95 falls in [2900, 3000)."""
    timings = [2000.0 + i * 10.0 for i in range(100)]  # 2000, 2010, ..., 2990

    result = compute_latency(timings)

    assert isinstance(result, MetricResult)
    assert result.name == "latency_p95_ms"
    assert 2900.0 <= result.value < 3000.0, (
        f"Expected P95 in [2900, 3000), got {result.value}"
    )


# ---------------------------------------------------------------------------
# Test 2: single timing → value equals that single timing
# ---------------------------------------------------------------------------


def test_single_timing() -> None:
    """A single timing [3500.0] → P95 == 3500.0."""
    result = compute_latency([3500.0])

    assert result.name == "latency_p95_ms"
    assert result.value == pytest.approx(3500.0)


# ---------------------------------------------------------------------------
# Test 3: empty list → value == 0.0
# ---------------------------------------------------------------------------


def test_empty_timings() -> None:
    """Empty timings list → value=0.0, passed=True."""
    result = compute_latency([])

    assert result.name == "latency_p95_ms"
    assert result.value == pytest.approx(0.0)
    assert result.threshold is None
    assert result.passed is True


# ---------------------------------------------------------------------------
# Test 4: threshold pass — 50 identical timings below threshold
# ---------------------------------------------------------------------------


def test_threshold_pass() -> None:
    """50 timings at 1000 ms with threshold=4000 → P95=1000, passed=True."""
    result = compute_latency([1000.0] * 50, threshold=4000.0)

    assert result.threshold == pytest.approx(4000.0)
    assert result.passed is True


# ---------------------------------------------------------------------------
# Test 5: threshold fail — 50 identical timings above threshold
# ---------------------------------------------------------------------------


def test_threshold_fail() -> None:
    """50 timings at 5000 ms with threshold=4000 → P95=5000, passed=False."""
    result = compute_latency([5000.0] * 50, threshold=4000.0)

    assert result.threshold == pytest.approx(4000.0)
    assert result.passed is False
