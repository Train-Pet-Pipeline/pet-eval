"""W&B report generation for pet-eval gate results.

Publishes gate evaluation outcomes to Weights & Biases for experiment tracking.
All wandb config is sourced from params.yaml via wandb_config dict.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import wandb
from tenacity import retry, stop_after_attempt, wait_exponential

from pet_eval.gate.types import GateResult

logger = logging.getLogger(__name__)


@dataclass
class _WandbConfig:
    """Typed view of the wandb section from params.yaml."""

    project: str
    entity: str | None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> _WandbConfig:
        """Build a _WandbConfig from the wandb sub-dict of params.yaml.

        Args:
            cfg: The ``wandb`` sub-dict from params.yaml, e.g.
                 ``{"project": "pet-eval", "entity": ""}``.

        Returns:
            A populated :class:`_WandbConfig` instance.
        """
        entity_raw = cfg.get("entity", "") or ""
        return cls(
            project=cfg["project"],
            entity=entity_raw if entity_raw else None,
        )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def _init_wandb(
    config: _WandbConfig,
    eval_type: str,
    run_name: str,
    tags: list[str],
    metadata: dict[str, Any],
) -> Any:
    """Initialise a W&B run with retry for transient network failures.

    Args:
        config: Parsed wandb configuration.
        eval_type: Category label for the evaluation.
        run_name: Short identifier for the run.
        tags: Tags to apply to the run.
        metadata: Key/value pairs stored as the run config.

    Returns:
        The ``wandb.Run`` instance.
    """
    return wandb.init(
        project=config.project,
        entity=config.entity,
        name=f"{eval_type}/{run_name}",
        tags=tags,
        config=metadata,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def _log_and_finish(run: Any, payload: dict[str, Any]) -> None:
    """Log detail payload and finish the W&B run with retry.

    Args:
        run: An active ``wandb.Run`` instance.
        payload: Structured detail payload to log.
    """
    run.log(payload)
    run.finish()


def generate_report(
    gate_result: GateResult,
    run_name: str,
    eval_type: str,
    metadata: dict[str, Any],
    wandb_config: dict[str, Any],
) -> None:
    """Publish a gate evaluation result to Weights & Biases.

    Initialises a W&B run, logs every metric's value/threshold/passed state to
    ``run.summary``, logs gate-level summary fields, calls ``run.log()`` for
    structured detail, then finishes the run.

    Tags applied to the run:

    - ``eval_type`` (e.g. ``"vlm"`` or ``"audio"``)
    - ``"pass"`` or ``"fail"`` depending on ``gate_result.passed``
    - ``"has_skipped"`` if ``gate_result.skipped`` is non-empty

    Args:
        gate_result: Aggregated gate evaluation result.
        run_name: Short identifier for the model/checkpoint under test.
        eval_type: Category label (``"vlm"``, ``"audio"``, etc.).
        metadata: Arbitrary key/value pairs stored as the W&B run config.
        wandb_config: The ``wandb`` sub-dict from params.yaml with ``project``
            and optional ``entity`` keys.
    """
    config = _WandbConfig.from_dict(wandb_config)

    tags: list[str] = [eval_type, "pass" if gate_result.passed else "fail"]
    if gate_result.skipped:
        tags.append("has_skipped")

    run = _init_wandb(config, eval_type, run_name, tags, metadata)

    # Log individual metric results to summary.
    for metric in gate_result.results:
        run.summary[f"metric/{metric.name}/value"] = metric.value
        run.summary[f"metric/{metric.name}/threshold"] = metric.threshold
        run.summary[f"metric/{metric.name}/passed"] = metric.passed

    # Log gate-level summary fields.
    run.summary["gate/passed"] = gate_result.passed
    run.summary["gate/summary"] = gate_result.summary
    run.summary["gate/skipped"] = gate_result.skipped

    # Structured detail log for timeline view.
    detail_payload: dict[str, Any] = {
        "gate_passed": gate_result.passed,
        "gate_summary": gate_result.summary,
        "skipped": gate_result.skipped,
    }
    for metric in gate_result.results:
        detail_payload[f"metric/{metric.name}"] = metric.value

    _log_and_finish(run, detail_payload)

    logger.info(
        "generate_report",
        extra={
            "eval_type": eval_type,
            "run_name": run_name,
            "passed": gate_result.passed,
            "wandb_project": config.project,
        },
    )
