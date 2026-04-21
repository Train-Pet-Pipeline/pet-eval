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
            "pet-eval v2 requires pet-infra. Install via matrix row 2026.08-rc."
        ) from e
    try:
        import pet_train  # noqa: F401  # cross-repo peer-dep for AudioEvaluator (P2-D)
    except ImportError as e:
        raise RuntimeError(
            "pet-eval v2 requires pet-train runtime (audio inference). "
            "Install via matrix row 2026.08-rc."
        ) from e

    # Cross-repo peer-dep (Phase 3B): pet-quantize for QuantizedVlmEvaluator.
    # Soft check — peer-dep-smoke CI is the real gate; warn here rather than
    # hard-fail because partial installs can delete pet_quantize transiently.
    try:
        import pet_quantize  # noqa: F401
        _pq_ver = getattr(pet_quantize, "__version__", "0.0.0")
        if not _pq_ver.startswith("2."):
            import logging
            logging.getLogger(__name__).warning(
                "pet-eval expects pet-quantize 2.x per matrix 2026.08-rc; got %s", _pq_ver
            )
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "pet-quantize not importable; QuantizedVlmEvaluator will fail at run-time"
        )

    # Metric plugins — import to trigger @METRICS.register_module side-effects
    # Evaluator plugins
    from pet_eval.plugins import (
        audio_evaluator,  # noqa: F401
        quantized_vlm_evaluator,  # noqa: F401
        vlm_evaluator,  # noqa: F401
    )
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
