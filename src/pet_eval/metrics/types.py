"""Core MetricResult data type for pet-eval metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MetricResult:
    """Immutable result produced by a single metric computation.

    Attributes:
        name: Metric identifier (e.g. "schema_compliance").
        value: Computed scalar value.
        threshold: Gate threshold; None means the metric is informational only.
        passed: Whether the metric meets its gate requirement.
        details: Optional free-form metadata dict (e.g. per-class breakdown).
    """

    name: str
    value: float
    threshold: float | None
    passed: bool
    details: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        *,
        value: float,
        threshold: float | None,
        operator: Literal["gte", "lte"] = "gte",
        details: dict | None = None,
    ) -> MetricResult:
        """Create a MetricResult with auto-computed ``passed`` field.

        Args:
            name: Metric identifier.
            value: Computed scalar value.
            threshold: Gate threshold. Pass ``None`` for informational metrics
                that always pass.
            operator: Comparison direction:
                - ``"gte"`` → ``passed = value >= threshold``
                - ``"lte"`` → ``passed = value <= threshold``
            details: Optional metadata dict; defaults to empty dict.

        Returns:
            A frozen :class:`MetricResult` instance.
        """
        if threshold is None:
            passed = True
        elif operator == "gte":
            passed = value >= threshold
        else:  # "lte"
            passed = value <= threshold

        return cls(
            name=name,
            value=value,
            threshold=threshold,
            passed=passed,
            details=details if details is not None else {},
        )
