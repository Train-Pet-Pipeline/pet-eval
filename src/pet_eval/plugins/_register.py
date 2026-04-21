"""Plugin registration entry-point for pet-eval (Phase 3A).

Invoked via the ``pet_infra.plugins`` entry-point. Registers all metric
adapters and (in P2-C/D) the VLM/Audio evaluators.
"""

from __future__ import annotations


def register_all() -> None:
    """Import plugin modules to trigger registration side-effects.

    Guards against missing peer-deps (pet-infra, pet-train) before importing
    any plugin modules. Metric modules are imported here to trigger
    ``@METRICS.register_module`` decorators.
    """
    try:
        import pet_infra  # noqa: F401  # peer-dep guard
    except ImportError as e:
        raise RuntimeError(
            "pet-eval v2 requires pet-infra. Install via matrix row 2026.07-rc."
        ) from e
    try:
        import pet_train  # noqa: F401  # cross-repo peer-dep for AudioEvaluator (P2-D)
    except ImportError as e:
        raise RuntimeError(
            "pet-eval v2 requires pet-train runtime (audio inference). "
            "Install via matrix row 2026.07-rc."
        ) from e

    # Metric plugins — import to trigger @METRICS.register_module side-effects
    from pet_eval.plugins.metrics import (  # noqa: F401
        anomaly_recall,
        audio_accuracy,
        calibration,
        kl_quantization,
        latency,
        mood_correlation,
        narrative_quality,
        schema_compliance,
    )
    # Evaluator plugins (filled in P2-C/D):
    # from pet_eval.plugins import vlm_evaluator, audio_evaluator  # noqa: F401
