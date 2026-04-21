"""Verify all 8 metrics are discoverable via METRICS registry after register_all."""

from __future__ import annotations

import pytest

EXPECTED_METRICS = {
    "anomaly_recall",
    "audio_accuracy",
    "calibration",
    "kl_quantization",
    "latency",
    "mood_correlation",
    "narrative_quality",
    "schema_compliance",
}


def test_register_all_populates_metrics_registry() -> None:
    """All 8 metrics appear in METRICS.module_dict after register_all."""
    from pet_eval.plugins._register import register_all

    register_all()
    from pet_infra.registry import METRICS

    got = set(METRICS.module_dict.keys())
    missing = EXPECTED_METRICS - got
    assert not missing, f"missing metrics: {missing}; got: {got}"


@pytest.mark.parametrize("name", sorted(EXPECTED_METRICS))
def test_metric_registry_build_produces_callable(name: str) -> None:
    """METRICS.build({'type': name}) returns a callable instance."""
    from pet_eval.plugins._register import register_all

    register_all()
    from pet_infra.registry import METRICS

    metric = METRICS.build({"type": name})
    assert callable(metric)


def test_register_all_fails_without_pet_infra(monkeypatch) -> None:
    """register_all raises RuntimeError when pet_infra is not importable."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pet_infra":
            raise ImportError("simulated missing pet-infra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    import importlib

    import pet_eval.plugins._register as register_mod

    importlib.reload(register_mod)

    with pytest.raises(RuntimeError, match="pet-infra"):
        register_mod.register_all()
