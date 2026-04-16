"""Constrained decoding for structured JSON output via outlines.

Wraps the ``outlines`` library to enforce that VLM generation produces
valid JSON conforming to the PetFeederEvent JSON Schema.  This guarantees
100% schema compliance at the token-sampling level, rather than relying
on post-hoc validation.

Usage::

    from pet_eval.inference.constrained import build_constrained_generator

    generator = build_constrained_generator(model, processor, schema_version="1.0")
    output = generator(prompt_text, max_tokens=1024)

When ``outlines`` is not installed, :func:`build_constrained_generator` raises
``ImportError`` with a clear message.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_DIR = None

try:
    from pet_schema.validator import VERSIONS_DIR

    _SCHEMA_DIR = VERSIONS_DIR
except ImportError:
    pass


def _load_json_schema(schema_version: str = "1.0") -> dict[str, Any]:
    """Load the PetFeederEvent JSON Schema for the given version.

    Args:
        schema_version: Schema version string (e.g. ``"1.0"``).

    Returns:
        Parsed JSON Schema dict.

    Raises:
        FileNotFoundError: If the schema file does not exist.
    """
    if _SCHEMA_DIR is None:
        raise ImportError(
            "pet_schema package is required for constrained decoding"
        )
    schema_path = _SCHEMA_DIR / f"v{schema_version}" / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"JSON Schema not found: {schema_path}"
        )
    return json.loads(schema_path.read_text())


def is_available() -> bool:
    """Check if the outlines library is installed and usable."""
    try:
        import outlines  # noqa: F401

        return True
    except ImportError:
        return False


def build_constrained_generator(
    model: Any,
    tokenizer: Any,
    schema_version: str = "1.0",
) -> Any:
    """Build an outlines JSON-schema-constrained generator.

    The returned callable accepts ``(prompt: str, max_tokens: int)`` and
    returns a JSON string guaranteed to conform to the PetFeederEvent schema.

    Args:
        model: A HuggingFace model (merged, on device).
        tokenizer: The corresponding tokenizer/processor.
        schema_version: PetFeederEvent schema version.

    Returns:
        A callable ``generator(prompt, max_tokens) -> str``.

    Raises:
        ImportError: If ``outlines`` is not installed.
    """
    try:
        from outlines import generate, models
    except ImportError as e:
        raise ImportError(
            "outlines library is required for constrained decoding. "
            "Install with: pip install outlines"
        ) from e

    schema = _load_json_schema(schema_version)
    schema_str = json.dumps(schema)

    logger.info(
        "Building constrained generator",
        extra={"schema_version": schema_version},
    )

    # Wrap the already-loaded HuggingFace model for outlines
    outlines_model = models.Transformers(model, tokenizer)
    generator = generate.json(outlines_model, schema_str)

    return generator
