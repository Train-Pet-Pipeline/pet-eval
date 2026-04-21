"""Tests for schema_compliance metric — TDD, tests written before implementation."""

from __future__ import annotations

import json

import pytest

from pet_eval.plugins.metrics.schema_compliance import compute_schema_compliance
from pet_eval.plugins.metrics.types import MetricResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json(d: dict) -> str:
    """Serialise a dict to a JSON string."""
    return json.dumps(d)


# ---------------------------------------------------------------------------
# Test 1: all valid outputs → compliance=1.0, sum_error near 0
# ---------------------------------------------------------------------------


def test_all_valid(sample_vlm_output_valid: dict) -> None:
    """Five valid outputs should yield compliance_rate=1.0 and sum_error<0.01."""
    outputs = [_json(sample_vlm_output_valid)] * 5
    results = compute_schema_compliance(outputs, schema_version="1.0")

    assert len(results) == 2
    compliance, sum_error = results

    assert compliance.name == "compliance_rate"
    assert compliance.value == pytest.approx(1.0)

    assert sum_error.name == "distribution_sum_error"
    assert sum_error.value < 0.01


# ---------------------------------------------------------------------------
# Test 2: all invalid JSON → compliance=0.0
# ---------------------------------------------------------------------------


def test_all_invalid_json() -> None:
    """All non-parseable strings should yield compliance_rate=0.0."""
    outputs = ["not json", "{bad"]
    results = compute_schema_compliance(outputs, schema_version="1.0")

    assert len(results) == 2
    compliance, _sum_error = results

    assert compliance.name == "compliance_rate"
    assert compliance.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 3: mixed valid and invalid → compliance≈0.667
# ---------------------------------------------------------------------------


def test_mixed_valid_invalid(sample_vlm_output_valid: dict) -> None:
    """Two valid + one invalid string → compliance_rate ≈ 0.667."""
    outputs = [_json(sample_vlm_output_valid), "not json", _json(sample_vlm_output_valid)]
    results = compute_schema_compliance(outputs, schema_version="1.0")

    assert len(results) == 2
    compliance, _sum_error = results

    assert compliance.name == "compliance_rate"
    assert compliance.value == pytest.approx(2 / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 4: output with bad distribution sum → sum_error > 0.01
# ---------------------------------------------------------------------------


def test_bad_distribution_sum(sample_vlm_output_invalid: dict) -> None:
    """A schema-invalid output (bad distribution sum) should report sum_error > 0.01.

    Note: the invalid fixture has a distribution that sums to 1.75; the function
    computes the error on the raw JSON regardless of schema validity, so we pass
    the JSON directly and expect a large error value.
    """
    # Build an output that is syntactically valid JSON but has a distribution
    # that clearly sums to != 1.0 so we can test _distribution_sum_error logic.
    # We build a minimal valid-structure JSON with a bad sum manually so that
    # validate_output may or may not pass — we just care sum_error is large.
    bad_output = {
        "schema_version": "1.0",
        "pet_present": True,
        "pet_count": 1,
        "pet": {
            "species": "cat",
            "breed_estimate": "domestic shorthair",
            "id_tag": "cat_001",
            "id_confidence": 0.90,
            "action": {
                "primary": "eating",
                "distribution": {
                    "eating": 0.75,
                    "drinking": 0.05,
                    "sniffing_only": 0.10,
                    "leaving_bowl": 0.05,
                    "sitting_idle": 0.30,
                    "other": 0.50,  # sum = 1.75
                },
            },
            "eating_metrics": {
                "speed": {"fast": 0.20, "normal": 0.60, "slow": 0.20},
                "engagement": 0.85,
                "abandoned_midway": 0.05,
            },
            "mood": {"alertness": 0.70, "anxiety": 0.10, "engagement": 0.85},
            "body_signals": {"posture": "relaxed", "ear_position": "forward"},
            "anomaly_signals": {
                "vomit_gesture": 0.02,
                "food_rejection": 0.05,
                "excessive_sniffing": 0.10,
                "lethargy": 0.03,
                "aggression": 0.01,
            },
        },
        "bowl": {"food_fill_ratio": 0.65, "water_fill_ratio": None, "food_type_visible": "dry"},
        "scene": {"lighting": "bright", "image_quality": "clear", "confidence_overall": 0.94},
        "narrative": "Cat eating dry food at normal pace.",
    }
    # Compute distribution sum error directly via the helper logic.
    # We feed a list of one such output; if validate_output flags it invalid
    # the sum_error will still be computed from the raw JSON.
    results = compute_schema_compliance([_json(bad_output)], schema_version="1.0")

    assert len(results) == 2
    _compliance, sum_error = results

    assert sum_error.name == "distribution_sum_error"
    # action distribution sums to 1.75 → error ≈ 0.75; speed sums to 1.0 → error 0.0
    # mean ≈ 0.375 >> 0.01
    assert sum_error.value > 0.01


# ---------------------------------------------------------------------------
# Test 5: empty outputs list → compliance=0.0, sum_error=1.0
# ---------------------------------------------------------------------------


def test_empty_outputs() -> None:
    """Empty input list should return compliance=0.0 and sum_error=1.0."""
    results = compute_schema_compliance([], schema_version="1.0")

    assert len(results) == 2
    compliance, sum_error = results

    assert compliance.name == "compliance_rate"
    assert compliance.value == pytest.approx(0.0)

    assert sum_error.name == "distribution_sum_error"
    assert sum_error.value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 6: thresholds forwarded correctly to MetricResult
# ---------------------------------------------------------------------------


def test_thresholds_from_params(sample_vlm_output_valid: dict) -> None:
    """Explicit threshold kwargs must be forwarded to the MetricResult objects."""
    outputs = [_json(sample_vlm_output_valid)]
    results = compute_schema_compliance(
        outputs,
        compliance_threshold=0.99,
        sum_error_threshold=0.01,
        schema_version="1.0",
    )

    assert len(results) == 2
    compliance, sum_error = results

    assert compliance.threshold == pytest.approx(0.99)
    assert sum_error.threshold == pytest.approx(0.01)
    # Returned objects must be MetricResult instances
    assert isinstance(compliance, MetricResult)
    assert isinstance(sum_error, MetricResult)
