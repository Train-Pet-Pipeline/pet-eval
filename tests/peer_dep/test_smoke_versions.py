"""Smoke tests verifying installed peer-dep versions match Phase 5/6 pins."""
from __future__ import annotations

import importlib.metadata


def test_pet_schema_version() -> None:
    """pet-schema must resolve to 3.x (Phase 5/6 current peer-dep pin)."""
    assert importlib.metadata.version("pet-schema").startswith("3.")


def test_pet_infra_version() -> None:
    """pet-infra must resolve to 2.6.x (Phase 6 current peer-dep pin)."""
    assert importlib.metadata.version("pet-infra").startswith("2.6")


def test_evaluators_registry_reachable() -> None:
    """pet-eval evaluator plugins must be importable against the pinned peer-deps."""
    from pet_eval.plugins import _register

    _register.register_all()
    from pet_infra.registry import EVALUATORS

    names = set(EVALUATORS.module_dict.keys())
    assert names, "EVALUATORS registry empty after register_all()"
