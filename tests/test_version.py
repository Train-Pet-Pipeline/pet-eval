"""Parity test: __version__ in __init__.py must match importlib.metadata version."""
from __future__ import annotations

import importlib.metadata

import pet_eval


def test_version_attribute_matches_metadata() -> None:
    """pet_eval.__version__ must equal the installed package version from pip metadata."""
    installed = importlib.metadata.version("pet-eval")
    assert pet_eval.__version__ == installed, (
        f"pet_eval.__version__ ({pet_eval.__version__!r}) does not match "
        f"installed package metadata ({installed!r}). "
        "Update src/pet_eval/__init__.py to match pyproject.toml version."
    )
