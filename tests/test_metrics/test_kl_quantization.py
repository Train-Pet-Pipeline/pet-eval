"""Tests for kl_quantization metric — TDD, tests written before implementation."""

from __future__ import annotations

import pytest
import torch

from pet_eval.metrics.kl_quantization import compute_kl_divergence
from pet_eval.metrics.types import MetricResult

# ---------------------------------------------------------------------------
# Test 1: identical distributions → KL < 0.001
# ---------------------------------------------------------------------------


def test_identical_distributions() -> None:
    """Identical fp16 and quantized distributions should yield near-zero KL divergence."""
    dist = torch.softmax(torch.randn(100), dim=0)
    fp16 = [dist]
    quantized = [dist.clone()]

    result = compute_kl_divergence(fp16, quantized)

    assert isinstance(result, MetricResult)
    assert result.name == "kl_divergence"
    assert result.value < 0.001, f"Expected KL < 0.001 for identical dists, got {result.value}"


# ---------------------------------------------------------------------------
# Test 2: different distributions → KL > 0.0
# ---------------------------------------------------------------------------


def test_different_distributions() -> None:
    """Random fp16 and quantized distributions should yield KL > 0.0."""
    torch.manual_seed(42)
    fp16 = [torch.softmax(torch.randn(100), dim=0) for _ in range(5)]
    quantized = [torch.softmax(torch.randn(100), dim=0) for _ in range(5)]

    result = compute_kl_divergence(fp16, quantized)

    assert result.name == "kl_divergence"
    assert result.value > 0.0, f"Expected KL > 0.0 for different dists, got {result.value}"


# ---------------------------------------------------------------------------
# Test 3: threshold pass — identical distributions + threshold=0.02 → passed=True
# ---------------------------------------------------------------------------


def test_threshold_pass() -> None:
    """Identical distributions with threshold=0.02 should have passed=True."""
    dist = torch.softmax(torch.randn(100), dim=0)
    fp16 = [dist]
    quantized = [dist.clone()]

    result = compute_kl_divergence(fp16, quantized, threshold=0.02)

    assert result.threshold == pytest.approx(0.02)
    assert result.passed is True, "Expected passed=True when KL < threshold"


# ---------------------------------------------------------------------------
# Test 4: empty inputs → value=0.0
# ---------------------------------------------------------------------------


def test_empty_inputs() -> None:
    """Empty input lists should return KL divergence of 0.0."""
    result = compute_kl_divergence([], [])

    assert result.name == "kl_divergence"
    assert result.value == pytest.approx(0.0)
    assert result.passed is True
