"""Schema compliance metric for pet-eval.

Computes two MetricResult objects from a list of raw VLM output strings:
  - compliance_rate (gte): fraction of outputs that pass pet_schema validation
  - distribution_sum_error (lte): mean absolute deviation of probability
    distributions from 1.0 across all valid outputs
"""

from __future__ import annotations

import json
import logging

import pet_schema

from pet_eval.metrics.types import MetricResult

logger = logging.getLogger(__name__)


def _distribution_sum_error(raw: str) -> float:
    """Compute mean absolute error of distribution sums from 1.0.

    Parses *raw* as JSON and inspects:
      - pet.action.distribution  → |sum(values) - 1.0|
      - pet.eating_metrics.speed → |sum(values) - 1.0|

    Returns the mean of all errors found, or 0.0 if no distributions exist.

    Args:
        raw: Raw JSON string from VLM output.

    Returns:
        Mean absolute deviation of all distribution sums from 1.0.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.debug("_distribution_sum_error: could not parse JSON")
        return 0.0

    errors: list[float] = []

    pet = data.get("pet") or {}

    # pet.action.distribution
    action_dist = pet.get("action", {}).get("distribution", None)
    if isinstance(action_dist, dict) and action_dist:
        total = sum(action_dist.values())
        errors.append(abs(total - 1.0))

    # pet.eating_metrics.speed
    speed_dist = pet.get("eating_metrics", {}).get("speed", None)
    if isinstance(speed_dist, dict) and speed_dist:
        total = sum(speed_dist.values())
        errors.append(abs(total - 1.0))

    if not errors:
        return 0.0
    return sum(errors) / len(errors)


def compute_schema_compliance(
    outputs: list[str],
    *,
    compliance_threshold: float | None = None,
    sum_error_threshold: float | None = None,
    schema_version: str = "1.0",
) -> list[MetricResult]:
    """Compute schema compliance and distribution sum error for VLM outputs.

    For each output string, calls ``pet_schema.validate_output`` and determines
    whether the output is valid.  Two MetricResult objects are returned:

    1. **compliance_rate** (operator ``"gte"``): fraction of outputs that pass
       schema validation.
    2. **distribution_sum_error** (operator ``"lte"``): mean absolute deviation
       of probability-distribution sums from 1.0, computed over *all* outputs
       (valid or invalid) that can be parsed as JSON.

    Empty *outputs* list → compliance_rate=0.0, distribution_sum_error=1.0.

    Args:
        outputs: List of raw JSON strings produced by the VLM.
        compliance_threshold: Gate threshold for compliance_rate (``"gte"``).
            ``None`` → informational only.
        sum_error_threshold: Gate threshold for distribution_sum_error (``"lte"``).
            ``None`` → informational only.
        schema_version: Schema version string forwarded to ``validate_output``.

    Returns:
        A list of exactly two :class:`MetricResult` instances:
        ``[compliance_rate, distribution_sum_error]``.
    """
    if not outputs:
        logger.info("compute_schema_compliance: empty outputs list")
        return [
            MetricResult.create(
                "compliance_rate",
                value=0.0,
                threshold=compliance_threshold,
                operator="gte",
                details={"n_outputs": 0, "n_valid": 0},
            ),
            MetricResult.create(
                "distribution_sum_error",
                value=1.0,
                threshold=sum_error_threshold,
                operator="lte",
                details={"n_outputs": 0},
            ),
        ]

    n_valid = 0
    sum_errors: list[float] = []

    for raw in outputs:
        try:
            result = pet_schema.validate_output(raw, version=schema_version)
        except Exception:
            logger.debug("validate_output raised an exception; treating as invalid")
            result_valid = False
        else:
            result_valid = result.valid

        if result_valid:
            n_valid += 1

        # Always compute the raw distribution sum error so we can report it
        # even when the output fails schema validation.
        sum_errors.append(_distribution_sum_error(raw))

    compliance_rate = n_valid / len(outputs)
    mean_sum_error = sum(sum_errors) / len(sum_errors) if sum_errors else 1.0

    logger.info(
        "compute_schema_compliance",
        extra={
            "n_outputs": len(outputs),
            "n_valid": n_valid,
            "compliance_rate": compliance_rate,
            "distribution_sum_error": mean_sum_error,
        },
    )

    return [
        MetricResult.create(
            "compliance_rate",
            value=compliance_rate,
            threshold=compliance_threshold,
            operator="gte",
            details={"n_outputs": len(outputs), "n_valid": n_valid},
        ),
        MetricResult.create(
            "distribution_sum_error",
            value=mean_sum_error,
            threshold=sum_error_threshold,
            operator="lte",
            details={"n_outputs": len(outputs)},
        ),
    ]
