"""Gate evaluation for evaluator plugins (Phase 3A).

Checks metric values against threshold config with `min_*` / `max_*` conventions.
Returns a GateResult indicating pass/fail and the offending thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    """Result of a gate check against min_*/max_* thresholds.

    Attributes:
        passed: True when all applicable thresholds are met.
        reason: Human-readable summary; "all thresholds met" on pass.
        thresholds: The threshold dict that was evaluated.
    """

    passed: bool
    reason: str
    thresholds: dict[str, float]


def apply_gate(metrics: dict[str, float], thresholds: dict[str, float]) -> GateResult:
    """Check metrics against min_*/max_* thresholds.

    For each threshold key:
      - `min_<metric>`: fail if metrics[<metric>] < threshold
      - `max_<metric>`: fail if metrics[<metric>] > threshold
      - other prefix: ignored (treated as informational metadata)

    Missing metrics default to 0 for min_ checks and +inf for max_ checks
    (conservative: missing = likely fail).

    Args:
        metrics: Computed metric values keyed by metric name.
        thresholds: Threshold dict; only keys starting with min_/max_ are evaluated.

    Returns:
        A frozen :class:`GateResult` instance.
    """
    failures: list[str] = []
    for key, threshold in thresholds.items():
        if key.startswith("min_"):
            metric_name = key[len("min_") :]
            value = metrics.get(metric_name, 0)
            if value < threshold:
                failures.append(f"{metric_name}={value} < {key}={threshold}")
        elif key.startswith("max_"):
            metric_name = key[len("max_") :]
            value = metrics.get(metric_name, float("inf"))
            if value > threshold:
                failures.append(f"{metric_name}={value} > {key}={threshold}")
    return GateResult(
        passed=not failures,
        reason="; ".join(failures) if failures else "all thresholds met",
        thresholds=thresholds,
    )
