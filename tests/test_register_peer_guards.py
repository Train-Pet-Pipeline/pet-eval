"""Tests for peer-dep guards in _register.register_all() (DEV_GUIDE §11.3 Mode B)."""
from __future__ import annotations

import sys
from unittest.mock import patch


def test_missing_pet_schema_raises_runtime_error() -> None:
    """register_all() must raise RuntimeError when pet-schema is not importable."""
    import importlib

    # Ensure _register is re-imported fresh so the guard runs again
    modules_to_remove = [
        "pet_eval.plugins._register",
        "pet_schema",
    ]
    saved = {k: sys.modules.pop(k) for k in modules_to_remove if k in sys.modules}
    try:
        with patch.dict(sys.modules, {"pet_schema": None}):  # type: ignore[dict-item]
            import importlib.util

            spec = importlib.util.find_spec("pet_eval.plugins._register")
            assert spec is not None

            # Re-import the module to trigger fresh guards
            import pet_eval.plugins._register as reg  # noqa: F401

            importlib.reload(reg)
            import pytest

            with pytest.raises(RuntimeError, match="pet-schema"):
                reg.register_all()
    finally:
        sys.modules.update(saved)


def test_missing_pet_infra_raises_runtime_error() -> None:
    """register_all() must raise RuntimeError when pet-infra is not importable."""
    import importlib

    modules_to_remove = [
        "pet_eval.plugins._register",
        "pet_infra",
    ]
    saved = {k: sys.modules.pop(k) for k in modules_to_remove if k in sys.modules}
    try:
        with patch.dict(sys.modules, {"pet_infra": None}):  # type: ignore[dict-item]
            import pet_eval.plugins._register as reg

            importlib.reload(reg)
            import pytest

            with pytest.raises(RuntimeError, match="pet-infra"):
                reg.register_all()
    finally:
        sys.modules.update(saved)
