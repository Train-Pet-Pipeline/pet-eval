"""Tests for narrative_quality BERTScore metric — TDD, tests written before implementation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from pet_eval.metrics.narrative_quality import compute_narrative_quality
from pet_eval.metrics.types import MetricResult

# ---------------------------------------------------------------------------
# Test 1: high similarity — mock returns 0.95 → value > 0.9
# ---------------------------------------------------------------------------


@patch("pet_eval.metrics.narrative_quality.bert_score.score")
def test_high_similarity(mock_score) -> None:
    """Mock bert_score returning 0.95 → mean F1 > 0.9."""
    mock_score.return_value = (
        torch.tensor([0.95]),
        torch.tensor([0.95]),
        torch.tensor([0.95]),
    )

    result = compute_narrative_quality(
        ["猫咪今天很开心"],
        ["猫咪今天非常开心"],
    )

    assert isinstance(result, MetricResult)
    assert result.name == "narrative_bertscore"
    assert result.value > 0.9


# ---------------------------------------------------------------------------
# Test 2: low similarity — mock returns 0.30 → value < 0.5
# ---------------------------------------------------------------------------


@patch("pet_eval.metrics.narrative_quality.bert_score.score")
def test_low_similarity(mock_score) -> None:
    """Mock bert_score returning 0.30 → mean F1 < 0.5."""
    mock_score.return_value = (
        torch.tensor([0.30]),
        torch.tensor([0.30]),
        torch.tensor([0.30]),
    )

    result = compute_narrative_quality(
        ["狗狗很悲伤"],
        ["猫咪很快乐"],
    )

    assert result.value < 0.5


# ---------------------------------------------------------------------------
# Test 3: multiple samples — mock returns [0.90, 0.80] → value ≈ 0.85
# ---------------------------------------------------------------------------


@patch("pet_eval.metrics.narrative_quality.bert_score.score")
def test_multiple_samples(mock_score) -> None:
    """Mock bert_score returning [0.90, 0.80] → mean F1 ≈ 0.85."""
    mock_score.return_value = (
        torch.tensor([0.90, 0.80]),
        torch.tensor([0.90, 0.80]),
        torch.tensor([0.90, 0.80]),
    )

    result = compute_narrative_quality(
        ["猫咪今天很开心", "狗狗在玩耍"],
        ["猫咪今天非常快乐", "狗狗正在嬉戏"],
    )

    assert result.value == pytest.approx(0.85, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 4: empty inputs — no mock needed → value = 0.0
# ---------------------------------------------------------------------------


def test_empty_inputs() -> None:
    """Empty input lists → value = 0.0 without calling bert_score."""
    result = compute_narrative_quality([], [])

    assert isinstance(result, MetricResult)
    assert result.name == "narrative_bertscore"
    assert result.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 5: threshold forwarded — mock + threshold=0.80 → threshold preserved
# ---------------------------------------------------------------------------


@patch("pet_eval.metrics.narrative_quality.bert_score.score")
def test_threshold_forwarded(mock_score) -> None:
    """threshold=0.80 is preserved in the returned MetricResult."""
    mock_score.return_value = (
        torch.tensor([0.88]),
        torch.tensor([0.88]),
        torch.tensor([0.88]),
    )

    result = compute_narrative_quality(
        ["猫咪在睡觉"],
        ["猫咪正在休息"],
        threshold=0.80,
    )

    assert result.threshold == pytest.approx(0.80)
