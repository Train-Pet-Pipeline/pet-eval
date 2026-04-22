"""Smoke tests verifying installed peer-dep versions match Phase 4 W2 pins."""
from __future__ import annotations

import importlib.metadata


def test_pet_schema_version() -> None:
    """pet-schema must resolve to 2.4.x (Phase 4 RC peer-dep pin)."""
    assert importlib.metadata.version("pet-schema").startswith("2.4")


def test_pet_infra_version() -> None:
    """pet-infra must resolve to 2.5.x (Phase 4 RC peer-dep pin)."""
    assert importlib.metadata.version("pet-infra").startswith("2.5")


def test_evaluators_registry_reachable() -> None:
    """pet-eval evaluator plugins must be importable against the pinned peer-deps."""
    from pet_eval.plugins import _register

    _register.register_all()
    from pet_infra.registry import EVALUATORS

    names = set(EVALUATORS.module_dict.keys())
    assert names, "EVALUATORS registry empty after register_all()"
