"""Registry contract test — pins that all 3 fusion evaluators register correctly."""

from __future__ import annotations


def test_fusion_evaluators_registered() -> None:
    """All 3 fusion evaluator names must appear in EVALUATORS after register_all()."""
    from pet_eval.plugins._register import register_all

    register_all()

    from pet_infra.registry import EVALUATORS

    expected = {"single_modal_fusion", "and_gate_fusion", "weighted_fusion"}
    registered = set(EVALUATORS.module_dict.keys())
    missing = expected - registered
    assert not missing, f"Missing fusion evaluators in EVALUATORS registry: {missing}"
