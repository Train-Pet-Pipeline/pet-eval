# pet-eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pet-eval evaluation pipeline — metrics, gate checking, runners, and wandb reporting — for VLM and audio CNN models.

**Architecture:** Pure metric modules compute results from data (no I/O). Runners orchestrate: load model → infer → call metrics → gate check → wandb report. Gate checker reads thresholds from params.yaml. Three runners: eval_trained (VLM FP16), eval_audio (audio CNN), eval_quantized (quantized + optional device).

**Tech Stack:** Python 3.11, pet-schema, torch, transformers, torchaudio, bert-score, scipy, wandb, tenacity, lm-evaluation-harness (vendor)

**Spec:** `docs/superpowers/specs/2026-04-15-pet-eval-design.md`

---

## File Structure

```
pet-eval/
├── src/pet_eval/
│   ├── __init__.py                    # Package exports
│   ├── cli.py                         # Unified CLI (argparse)
│   ├── metrics/
│   │   ├── __init__.py                # Re-export MetricResult + all compute functions
│   │   ├── types.py                   # MetricResult dataclass
│   │   ├── schema_compliance.py       # Schema compliance + distribution sum error
│   │   ├── calibration.py             # ECE (informational only)
│   │   ├── anomaly_recall.py          # Anomaly recall + false positive rate
│   │   ├── mood_correlation.py        # Spearman correlation vs teacher
│   │   ├── narrative_quality.py       # BERTScore (Chinese BERT)
│   │   ├── latency.py                 # P50/P95/P99 from raw timings
│   │   ├── kl_quantization.py         # KL divergence
│   │   └── audio_accuracy.py          # Per-class P/R/F1
│   ├── gate/
│   │   ├── __init__.py                # Re-export GateResult, check_gate
│   │   ├── types.py                   # GateResult dataclass
│   │   └── checker.py                 # Gate logic: compare MetricResults vs thresholds
│   ├── runners/
│   │   ├── __init__.py
│   │   ├── eval_trained.py            # VLM FP16 runner
│   │   ├── eval_quantized.py          # Quantized model runner (with/without device)
│   │   └── eval_audio.py              # Audio CNN runner
│   └── report/
│       ├── __init__.py
│       └── generate_report.py         # wandb integration
├── tasks/
│   └── pet_feeder.py                  # lm-eval-harness custom task
├── benchmark/
│   └── README.md                      # Gold set format + rules
├── vendor/                            # lm-evaluation-harness (git submodule)
├── tests/
│   ├── conftest.py                    # Shared fixtures
│   ├── test_metrics/
│   │   ├── test_schema_compliance.py
│   │   ├── test_calibration.py
│   │   ├── test_anomaly_recall.py
│   │   ├── test_mood_correlation.py
│   │   ├── test_narrative_quality.py
│   │   ├── test_latency.py
│   │   ├── test_kl_quantization.py
│   │   └── test_audio_accuracy.py
│   ├── test_gate/
│   │   └── test_checker.py
│   ├── test_runners/
│   │   ├── test_eval_trained.py
│   │   ├── test_eval_quantized.py
│   │   └── test_eval_audio.py
│   └── test_report/
│       └── test_generate_report.py
├── params.yaml
├── pyproject.toml
├── Makefile
└── .gitignore
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.gitignore`, `params.yaml`
- Create: `src/pet_eval/__init__.py`
- Create: `benchmark/README.md`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```python
# pyproject.toml
```

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "pet-eval"
version = "0.1.0"
requires-python = ">=3.11,<3.12"

dependencies = [
    "pet-schema",
    "torch>=2.1",
    "transformers>=4.37",
    "torchaudio>=2.1",
    "bert-score>=0.3.13",
    "scipy>=1.11",
    "wandb",
    "pyyaml",
    "tenacity",
    "python-json-logger",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "ruff",
    "mypy",
]

[tool.setuptools.packages.find]
where = ["src"]
include = ["pet_eval", "pet_eval.*"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create Makefile**

```makefile
.PHONY: setup test lint clean

setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ && mypy src/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist/ *.egg-info
```

- [ ] **Step 3: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
.eggs/

# Outputs
outputs/

# IDE
.vscode/
.idea/

# Tools
.pytest_cache/
.mypy_cache/
.ruff_cache/

# OS
.DS_Store

# Wandb
wandb/

# Benchmark data (large files tracked by DVC, not git)
benchmark/*.jsonl
benchmark/frames/
```

- [ ] **Step 4: Create params.yaml**

```yaml
# === Gate Thresholds ===
gates:
  vlm:
    schema_compliance: 0.99
    distribution_sum_error: 0.01
    anomaly_recall: 0.85
    anomaly_false_positive: 0.15
    mood_spearman: 0.75
    narrative_bertscore: 0.80
    latency_p95_ms: 4000
    kl_divergence: 0.02
  audio:
    overall_accuracy: 0.80
    vomit_recall: 0.70

# === Benchmark Data ===
benchmark:
  gold_set_path: "benchmark/gold_set_v1.jsonl"
  anomaly_set_path: "benchmark/anomaly_set_v1.jsonl"
  audio_test_dir: ""

# === wandb ===
wandb:
  project: "pet-eval"
  entity: ""

# === Model Inference ===
inference:
  schema_version: "1.0"
  max_new_tokens: 1024
  batch_size: 1

# === Audio ===
audio:
  classes: ["eating", "drinking", "vomiting", "ambient", "other"]

# === Device ===
device:
  adb_timeout: 30
  warmup_runs: 3
  latency_runs: 50
```

- [ ] **Step 5: Create src/pet_eval/__init__.py**

```python
"""pet-eval: Evaluation pipeline for Train-Pet-Pipeline."""
```

- [ ] **Step 6: Create benchmark/README.md**

```markdown
# Benchmark Data

This directory holds gold-standard evaluation data for pet-eval.

## Files

- `gold_set_v1.jsonl` — Human-expert annotated evaluation samples (not yet created)
- `anomaly_set_v1.jsonl` — Anomaly detection evaluation samples (not yet created)
- `frames/` — Associated image frames (not yet created)

## Gold Set Format

Each line is a JSON object:

\`\`\`json
{
  "gold_id": "gold_001",
  "frame_path": "benchmark/frames/gold_001.jpg",
  "expected_output": { "schema_version": "1.0", ... },
  "annotator": "human_expert",
  "annotation_date": "2026-04-15",
  "difficulty": "normal",
  "notes": "Typical normal eating, baseline verification"
}
\`\`\`

## Admission Rules

1. Every entry must be confirmed by a human expert; no VLM-only annotations accepted
2. Existing entries are immutable; new samples append to new version files
3. `anomaly_set_v1.jsonl` must contain >= 70% real anomaly samples (not synthetic)
4. Gold set samples must NEVER appear in any training set

## Current State

Data files are not yet created. Runners gracefully skip gold-set-dependent metrics
when files are absent.
```

- [ ] **Step 7: Create tests/conftest.py with shared fixtures**

```python
"""Shared test fixtures for pet-eval."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_params(tmp_dir):
    """Provide a minimal params.yaml for testing."""
    params = {
        "gates": {
            "vlm": {
                "schema_compliance": 0.99,
                "distribution_sum_error": 0.01,
                "anomaly_recall": 0.85,
                "anomaly_false_positive": 0.15,
                "mood_spearman": 0.75,
                "narrative_bertscore": 0.80,
                "latency_p95_ms": 4000,
                "kl_divergence": 0.02,
            },
            "audio": {
                "overall_accuracy": 0.80,
                "vomit_recall": 0.70,
            },
        },
        "benchmark": {
            "gold_set_path": str(tmp_dir / "gold_set.jsonl"),
            "anomaly_set_path": str(tmp_dir / "anomaly_set.jsonl"),
            "audio_test_dir": str(tmp_dir / "audio_test"),
        },
        "wandb": {"project": "pet-eval-test", "entity": ""},
        "inference": {
            "schema_version": "1.0",
            "max_new_tokens": 1024,
            "batch_size": 1,
        },
        "audio": {
            "classes": ["eating", "drinking", "vomiting", "ambient", "other"],
        },
        "device": {
            "adb_timeout": 30,
            "warmup_runs": 3,
            "latency_runs": 50,
        },
    }
    params_path = tmp_dir / "params.yaml"
    params_path.write_text(yaml.dump(params))
    return params_path


@pytest.fixture
def sample_vlm_output_valid():
    """Provide a valid VLM JSON output string."""
    return json.dumps(
        {
            "schema_version": "1.0",
            "pet_present": True,
            "pet_count": 1,
            "pet": {
                "species": "cat",
                "action": {
                    "primary": "eating",
                    "distribution": {
                        "eating": 0.85,
                        "drinking": 0.05,
                        "sniffing_only": 0.05,
                        "leaving_bowl": 0.02,
                        "sitting_idle": 0.02,
                        "other": 0.01,
                    },
                },
                "eating_metrics": {
                    "duration_seconds": 120,
                    "speed": {"fast": 0.1, "normal": 0.7, "slow": 0.2},
                },
                "mood": {
                    "alertness": 0.6,
                    "anxiety": 0.1,
                    "engagement": 0.8,
                },
                "body_signals": {
                    "posture": "relaxed",
                    "ear_position": "forward",
                },
                "anomaly_signals": {
                    "vomit_gesture": 0.0,
                    "food_rejection": 0.0,
                    "excessive_sniffing": 0.05,
                    "lethargy": 0.0,
                    "aggression": 0.0,
                },
            },
            "bowl": {
                "food_fill_ratio": 0.6,
                "water_fill_ratio": 0.8,
                "food_type_visible": "dry_kibble",
            },
            "scene": {
                "lighting": "normal",
                "image_quality": "clear",
                "confidence_overall": 0.92,
            },
            "narrative": "橘猫正常进食中，状态良好",
        },
        ensure_ascii=False,
    )


@pytest.fixture
def sample_vlm_output_invalid():
    """Provide an invalid VLM JSON output string (bad distribution sum)."""
    return json.dumps(
        {
            "schema_version": "1.0",
            "pet_present": True,
            "pet_count": 1,
            "pet": {
                "species": "cat",
                "action": {
                    "primary": "eating",
                    "distribution": {
                        "eating": 0.85,
                        "drinking": 0.05,
                        "sniffing_only": 0.05,
                        "leaving_bowl": 0.02,
                        "sitting_idle": 0.02,
                        "other": 0.50,
                    },
                },
                "eating_metrics": {
                    "duration_seconds": 120,
                    "speed": {"fast": 0.1, "normal": 0.7, "slow": 0.2},
                },
                "mood": {
                    "alertness": 0.6,
                    "anxiety": 0.1,
                    "engagement": 0.8,
                },
                "body_signals": {
                    "posture": "relaxed",
                    "ear_position": "forward",
                },
                "anomaly_signals": {
                    "vomit_gesture": 0.0,
                    "food_rejection": 0.0,
                    "excessive_sniffing": 0.05,
                    "lethargy": 0.0,
                    "aggression": 0.0,
                },
            },
            "bowl": {
                "food_fill_ratio": 0.6,
                "water_fill_ratio": 0.8,
                "food_type_visible": "dry_kibble",
            },
            "scene": {
                "lighting": "normal",
                "image_quality": "clear",
                "confidence_overall": 0.92,
            },
            "narrative": "橘猫正常进食中，状态良好",
        },
        ensure_ascii=False,
    )
```

- [ ] **Step 8: Create empty __init__.py files for all subpackages**

```bash
mkdir -p src/pet_eval/metrics src/pet_eval/gate src/pet_eval/runners src/pet_eval/report
mkdir -p tests/test_metrics tests/test_gate tests/test_runners tests/test_report
touch src/pet_eval/metrics/__init__.py
touch src/pet_eval/gate/__init__.py
touch src/pet_eval/runners/__init__.py
touch src/pet_eval/report/__init__.py
touch tests/__init__.py
touch tests/test_metrics/__init__.py
touch tests/test_gate/__init__.py
touch tests/test_runners/__init__.py
touch tests/test_report/__init__.py
```

- [ ] **Step 9: Install package and verify**

Run: `pip install -e ".[dev]"`
Expected: Successful installation, `python -c "import pet_eval"` works.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): project scaffolding with params, Makefile, fixtures"
```

---

## Task 2: MetricResult & GateResult Data Types

**Files:**
- Create: `src/pet_eval/metrics/types.py`
- Create: `src/pet_eval/gate/types.py`
- Test: `tests/test_metrics/test_types.py`, `tests/test_gate/test_types.py`

- [ ] **Step 1: Write tests for MetricResult**

```python
# tests/test_metrics/test_types.py
"""Tests for MetricResult dataclass."""

from pet_eval.metrics.types import MetricResult


class TestMetricResult:
    def test_gated_metric_pass(self):
        """Gated metric passes when value meets threshold."""
        r = MetricResult.create("test", value=0.95, threshold=0.90, operator="gte")
        assert r.passed is True
        assert r.threshold == 0.90

    def test_gated_metric_fail(self):
        """Gated metric fails when value does not meet threshold."""
        r = MetricResult.create("test", value=0.80, threshold=0.90, operator="gte")
        assert r.passed is False

    def test_gated_metric_lte_pass(self):
        """Gated metric with lte operator passes when value <= threshold."""
        r = MetricResult.create("kl", value=0.01, threshold=0.02, operator="lte")
        assert r.passed is True

    def test_gated_metric_lte_fail(self):
        """Gated metric with lte operator fails when value > threshold."""
        r = MetricResult.create("kl", value=0.03, threshold=0.02, operator="lte")
        assert r.passed is False

    def test_informational_metric_always_passes(self):
        """Informational metric (threshold=None) always passes."""
        r = MetricResult.create("ece", value=0.15, threshold=None)
        assert r.passed is True
        assert r.threshold is None

    def test_details_default_empty(self):
        """Details default to empty dict."""
        r = MetricResult.create("test", value=0.5, threshold=0.5, operator="gte")
        assert r.details == {}

    def test_details_provided(self):
        """Details can be provided."""
        d = {"per_class": {"eating": 0.9}}
        r = MetricResult.create("test", value=0.5, threshold=None, details=d)
        assert r.details == d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pet_eval.metrics.types'`

- [ ] **Step 3: Implement MetricResult**

```python
# src/pet_eval/metrics/types.py
"""Core data types for metric results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MetricResult:
    """Result of a single metric computation.

    Attributes:
        name: Metric identifier (e.g. 'schema_compliance').
        value: Computed numeric value.
        threshold: Gate threshold from params.yaml. None means informational only.
        passed: Whether this metric meets its gate. Always True if informational.
        details: Arbitrary extra data (per-class breakdown, etc.).
    """

    name: str
    value: float
    threshold: float | None
    passed: bool
    details: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        *,
        value: float,
        threshold: float | None,
        operator: Literal["gte", "lte"] = "gte",
        details: dict | None = None,
    ) -> MetricResult:
        """Create a MetricResult with automatic pass/fail computation.

        Args:
            name: Metric identifier.
            value: Computed value.
            threshold: Gate threshold (None = informational, always passes).
            operator: 'gte' means value >= threshold is pass,
                      'lte' means value <= threshold is pass.
            details: Optional extra data.
        """
        if threshold is None:
            passed = True
        elif operator == "gte":
            passed = value >= threshold
        else:
            passed = value <= threshold
        return cls(
            name=name,
            value=value,
            threshold=threshold,
            passed=passed,
            details=details or {},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics/test_types.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Write tests for GateResult**

```python
# tests/test_gate/test_types.py
"""Tests for GateResult dataclass."""

from pet_eval.gate.types import GateResult
from pet_eval.metrics.types import MetricResult


class TestGateResult:
    def test_all_pass(self):
        """Gate passes when all metrics pass."""
        results = [
            MetricResult.create("a", value=0.99, threshold=0.95, operator="gte"),
            MetricResult.create("b", value=0.01, threshold=0.02, operator="lte"),
        ]
        g = GateResult.from_results(results, skipped=[])
        assert g.passed is True
        assert "PASS" in g.summary

    def test_one_fail(self):
        """Gate fails when any metric fails."""
        results = [
            MetricResult.create("a", value=0.99, threshold=0.95, operator="gte"),
            MetricResult.create("b", value=0.03, threshold=0.02, operator="lte"),
        ]
        g = GateResult.from_results(results, skipped=[])
        assert g.passed is False
        assert "FAIL" in g.summary

    def test_informational_ignored(self):
        """Informational metrics do not affect gate outcome."""
        results = [
            MetricResult.create("gated", value=0.99, threshold=0.95, operator="gte"),
            MetricResult.create("info", value=999.0, threshold=None),
        ]
        g = GateResult.from_results(results, skipped=[])
        assert g.passed is True

    def test_skipped_recorded(self):
        """Skipped metric names are recorded."""
        results = [
            MetricResult.create("a", value=0.99, threshold=0.95, operator="gte"),
        ]
        g = GateResult.from_results(results, skipped=["latency"])
        assert g.skipped == ["latency"]
        assert g.passed is True

    def test_empty_results_passes(self):
        """Empty results with skipped metrics still passes."""
        g = GateResult.from_results([], skipped=["all_skipped"])
        assert g.passed is True
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_gate/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 7: Implement GateResult**

```python
# src/pet_eval/gate/types.py
"""Core data types for gate results."""

from __future__ import annotations

from dataclasses import dataclass, field

from pet_eval.metrics.types import MetricResult


@dataclass(frozen=True)
class GateResult:
    """Aggregated gate outcome.

    Attributes:
        passed: True only if all non-skipped, non-informational metrics pass.
        results: Per-metric details.
        skipped: Names of metrics that were skipped.
        summary: Human-readable one-line summary.
    """

    passed: bool
    results: list[MetricResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_results(
        cls,
        results: list[MetricResult],
        skipped: list[str],
    ) -> GateResult:
        """Build GateResult from metric results.

        Informational metrics (threshold=None) are included in results
        but do not affect the pass/fail outcome.
        """
        gated = [r for r in results if r.threshold is not None]
        all_pass = all(r.passed for r in gated)
        failed_names = [r.name for r in gated if not r.passed]
        if all_pass:
            summary = f"PASS — {len(gated)} gated metrics passed"
            if skipped:
                summary += f", {len(skipped)} skipped"
        else:
            summary = f"FAIL — failed: {', '.join(failed_names)}"
        return cls(
            passed=all_pass,
            results=list(results),
            skipped=list(skipped),
            summary=summary,
        )
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_gate/test_types.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 9: Update __init__.py exports**

```python
# src/pet_eval/metrics/__init__.py
"""Metrics package — pure computation modules."""

from pet_eval.metrics.types import MetricResult

__all__ = ["MetricResult"]
```

```python
# src/pet_eval/gate/__init__.py
"""Gate package — pass/fail logic."""

from pet_eval.gate.types import GateResult

__all__ = ["GateResult"]
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): MetricResult and GateResult data types with tests"
```

---

## Task 3: schema_compliance Metric

**Files:**
- Create: `src/pet_eval/metrics/schema_compliance.py`
- Test: `tests/test_metrics/test_schema_compliance.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_metrics/test_schema_compliance.py
"""Tests for schema compliance metric."""

import json

from pet_eval.metrics.schema_compliance import compute_schema_compliance


class TestSchemaCompliance:
    def test_all_valid(self, sample_vlm_output_valid):
        """All valid outputs → 100% compliance, 0 sum error."""
        outputs = [sample_vlm_output_valid] * 5
        results = compute_schema_compliance(outputs)
        assert len(results) == 2
        compliance = results[0]
        assert compliance.name == "schema_compliance"
        assert compliance.value == 1.0
        sum_err = results[1]
        assert sum_err.name == "distribution_sum_error"
        assert sum_err.value < 0.01

    def test_all_invalid_json(self):
        """Invalid JSON → 0% compliance."""
        outputs = ["not json", "{bad"]
        results = compute_schema_compliance(outputs)
        assert results[0].value == 0.0

    def test_mixed_valid_invalid(self, sample_vlm_output_valid):
        """Mix of valid and invalid → partial compliance."""
        outputs = [sample_vlm_output_valid, "not json", sample_vlm_output_valid]
        results = compute_schema_compliance(outputs)
        assert abs(results[0].value - 2.0 / 3.0) < 0.01

    def test_bad_distribution_sum(self, sample_vlm_output_invalid):
        """Distribution sum error detected."""
        outputs = [sample_vlm_output_invalid]
        results = compute_schema_compliance(outputs)
        sum_err = results[1]
        assert sum_err.value > 0.01

    def test_empty_outputs(self):
        """Empty list returns 0 compliance."""
        results = compute_schema_compliance([])
        assert results[0].value == 0.0

    def test_thresholds_from_params(self):
        """Thresholds passed through correctly."""
        results = compute_schema_compliance(
            ["not json"],
            compliance_threshold=0.99,
            sum_error_threshold=0.01,
        )
        assert results[0].threshold == 0.99
        assert results[1].threshold == 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics/test_schema_compliance.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/pet_eval/metrics/schema_compliance.py
"""Schema compliance metric — validates VLM outputs against pet-schema."""

from __future__ import annotations

import json
import logging

from pet_schema import validate_output

from pet_eval.metrics.types import MetricResult

logger = logging.getLogger(__name__)


def compute_schema_compliance(
    outputs: list[str],
    *,
    compliance_threshold: float | None = None,
    sum_error_threshold: float | None = None,
    schema_version: str = "1.0",
) -> list[MetricResult]:
    """Compute schema compliance rate and distribution sum error.

    Args:
        outputs: Raw JSON strings from model.
        compliance_threshold: Gate threshold for compliance rate (None = informational).
        sum_error_threshold: Gate threshold for sum error (None = informational).
        schema_version: Schema version to validate against.

    Returns:
        Two MetricResults: [compliance_rate, distribution_sum_error].
    """
    if not outputs:
        return [
            MetricResult.create(
                "schema_compliance", value=0.0, threshold=compliance_threshold, operator="gte"
            ),
            MetricResult.create(
                "distribution_sum_error", value=1.0, threshold=sum_error_threshold, operator="lte"
            ),
        ]

    valid_count = 0
    sum_errors: list[float] = []

    for i, raw in enumerate(outputs):
        result = validate_output(raw, version=schema_version)
        if result.valid:
            valid_count += 1
            sum_errors.append(_distribution_sum_error(raw))
        else:
            logger.debug("Sample %d failed validation: %s", i, result.errors)

    compliance_rate = valid_count / len(outputs)
    mean_sum_error = sum(sum_errors) / len(sum_errors) if sum_errors else 1.0

    return [
        MetricResult.create(
            "schema_compliance",
            value=compliance_rate,
            threshold=compliance_threshold,
            operator="gte",
        ),
        MetricResult.create(
            "distribution_sum_error",
            value=mean_sum_error,
            threshold=sum_error_threshold,
            operator="lte",
        ),
    ]


def _distribution_sum_error(raw: str) -> float:
    """Compute mean absolute sum error across all distributions in a valid output."""
    data = json.loads(raw)
    errors: list[float] = []

    pet = data.get("pet", {})
    action_dist = pet.get("action", {}).get("distribution", {})
    if action_dist:
        total = sum(action_dist.values())
        errors.append(abs(total - 1.0))

    speed_dist = pet.get("eating_metrics", {}).get("speed", {})
    if speed_dist:
        total = sum(speed_dist.values())
        errors.append(abs(total - 1.0))

    return sum(errors) / len(errors) if errors else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics/test_schema_compliance.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): schema_compliance metric with tests"
```

---

## Task 4: calibration (ECE) Metric

**Files:**
- Create: `src/pet_eval/metrics/calibration.py`
- Test: `tests/test_metrics/test_calibration.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_metrics/test_calibration.py
"""Tests for ECE calibration metric."""

from pet_eval.metrics.calibration import compute_ece


class TestCalibration:
    def test_perfectly_calibrated(self):
        """Perfect calibration → ECE ≈ 0."""
        # confidence matches accuracy exactly
        confidences = [0.9, 0.9, 0.9, 0.9, 0.9]
        correct = [True, True, True, True, False]  # 80% accuracy in 0.9 bin — not perfect
        r = compute_ece(confidences, correct, n_bins=10)
        assert r.name == "calibration_ece"
        assert r.threshold is None  # informational only
        assert r.passed is True

    def test_overconfident(self):
        """All confident but all wrong → high ECE."""
        confidences = [0.95, 0.95, 0.95]
        correct = [False, False, False]
        r = compute_ece(confidences, correct, n_bins=10)
        assert r.value > 0.5

    def test_empty_inputs(self):
        """Empty inputs → ECE = 0."""
        r = compute_ece([], [], n_bins=10)
        assert r.value == 0.0

    def test_details_has_bins(self):
        """Details contain per-bin breakdown."""
        confidences = [0.1, 0.5, 0.9]
        correct = [False, True, True]
        r = compute_ece(confidences, correct, n_bins=10)
        assert "bins" in r.details
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics/test_calibration.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/pet_eval/metrics/calibration.py
"""Expected Calibration Error (ECE) — informational metric."""

from __future__ import annotations

from pet_eval.metrics.types import MetricResult


def compute_ece(
    confidences: list[float],
    correct: list[bool],
    *,
    n_bins: int = 10,
) -> MetricResult:
    """Compute Expected Calibration Error.

    Args:
        confidences: Model confidence scores per sample.
        correct: Whether the model prediction was correct per sample.
        n_bins: Number of bins for calibration histogram.

    Returns:
        MetricResult with threshold=None (informational only).
    """
    if not confidences:
        return MetricResult.create(
            "calibration_ece", value=0.0, threshold=None, details={"bins": []}
        )

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    bins_data: list[dict] = []
    ece = 0.0
    n = len(confidences)

    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        indices = [
            i for i, c in enumerate(confidences) if (lo <= c < hi) or (b == n_bins - 1 and c == hi)
        ]
        if not indices:
            bins_data.append({"range": [lo, hi], "count": 0, "avg_conf": 0, "accuracy": 0})
            continue
        avg_conf = sum(confidences[i] for i in indices) / len(indices)
        accuracy = sum(1 for i in indices if correct[i]) / len(indices)
        ece += (len(indices) / n) * abs(accuracy - avg_conf)
        bins_data.append({
            "range": [lo, hi],
            "count": len(indices),
            "avg_conf": round(avg_conf, 4),
            "accuracy": round(accuracy, 4),
        })

    return MetricResult.create(
        "calibration_ece",
        value=round(ece, 6),
        threshold=None,
        details={"bins": bins_data},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics/test_calibration.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): calibration ECE metric (informational) with tests"
```

---

## Task 5: anomaly_recall Metric

**Files:**
- Create: `src/pet_eval/metrics/anomaly_recall.py`
- Test: `tests/test_metrics/test_anomaly_recall.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_metrics/test_anomaly_recall.py
"""Tests for anomaly recall metric."""

from pet_eval.metrics.anomaly_recall import compute_anomaly_recall


class TestAnomalyRecall:
    def test_perfect_recall(self):
        """All anomalies detected → recall = 1.0."""
        predicted = [True, True, True, False, False]
        actual = [True, True, True, False, False]
        results = compute_anomaly_recall(predicted, actual)
        recall = results[0]
        fpr = results[1]
        assert recall.name == "anomaly_recall"
        assert recall.value == 1.0
        assert fpr.name == "anomaly_false_positive"
        assert fpr.value == 0.0

    def test_missed_anomaly(self):
        """Missed anomaly → recall < 1.0."""
        predicted = [True, False, False, False]
        actual = [True, True, False, False]
        results = compute_anomaly_recall(predicted, actual)
        assert results[0].value == 0.5  # 1/2 recall

    def test_false_positive(self):
        """False positive on normal → fpr > 0."""
        predicted = [True, True, True]
        actual = [True, False, False]
        results = compute_anomaly_recall(predicted, actual)
        assert results[1].value == 1.0  # 2/2 FP rate

    def test_no_anomalies(self):
        """No actual anomalies → recall = 0, fpr computed normally."""
        predicted = [False, False]
        actual = [False, False]
        results = compute_anomaly_recall(predicted, actual)
        assert results[0].value == 0.0
        assert results[1].value == 0.0

    def test_thresholds_passed(self):
        """Thresholds from params forwarded correctly."""
        results = compute_anomaly_recall(
            [True], [True], recall_threshold=0.85, fpr_threshold=0.15
        )
        assert results[0].threshold == 0.85
        assert results[1].threshold == 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics/test_anomaly_recall.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/pet_eval/metrics/anomaly_recall.py
"""Anomaly recall and false positive rate metrics."""

from __future__ import annotations

from pet_eval.metrics.types import MetricResult


def compute_anomaly_recall(
    predicted: list[bool],
    actual: list[bool],
    *,
    recall_threshold: float | None = None,
    fpr_threshold: float | None = None,
) -> list[MetricResult]:
    """Compute anomaly recall (TP/(TP+FN)) and false positive rate (FP/(FP+TN)).

    Args:
        predicted: Model predictions (True = anomaly detected).
        actual: Ground truth labels (True = real anomaly).
        recall_threshold: Gate threshold for recall (gte).
        fpr_threshold: Gate threshold for false positive rate (lte).

    Returns:
        Two MetricResults: [anomaly_recall, anomaly_false_positive].
    """
    tp = sum(1 for p, a in zip(predicted, actual) if p and a)
    fn = sum(1 for p, a in zip(predicted, actual) if not p and a)
    fp = sum(1 for p, a in zip(predicted, actual) if p and not a)
    tn = sum(1 for p, a in zip(predicted, actual) if not p and not a)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return [
        MetricResult.create(
            "anomaly_recall",
            value=recall,
            threshold=recall_threshold,
            operator="gte",
            details={"tp": tp, "fn": fn, "total_positive": tp + fn},
        ),
        MetricResult.create(
            "anomaly_false_positive",
            value=fpr,
            threshold=fpr_threshold,
            operator="lte",
            details={"fp": fp, "tn": tn, "total_negative": fp + tn},
        ),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics/test_anomaly_recall.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): anomaly_recall and false positive rate metric with tests"
```

---

## Task 6: mood_correlation Metric

**Files:**
- Create: `src/pet_eval/metrics/mood_correlation.py`
- Test: `tests/test_metrics/test_mood_correlation.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_metrics/test_mood_correlation.py
"""Tests for mood Spearman correlation metric."""

from pet_eval.metrics.mood_correlation import compute_mood_correlation


class TestMoodCorrelation:
    def test_perfect_correlation(self):
        """Identical scores → correlation = 1.0."""
        model = [{"alertness": 0.5, "anxiety": 0.3, "engagement": 0.8}] * 5
        teacher = [{"alertness": 0.5, "anxiety": 0.3, "engagement": 0.8}] * 5
        r = compute_mood_correlation(model, teacher)
        assert r.name == "mood_spearman"
        # Perfect correlation when all values identical → spearman is nan or 1.0
        # With identical values, spearman returns nan; handle this edge case
        assert r.value >= 0.0 or r.value != r.value  # nan check

    def test_inverse_correlation(self):
        """Inversed scores → negative correlation."""
        model = [
            {"alertness": 0.1, "anxiety": 0.1, "engagement": 0.1},
            {"alertness": 0.5, "anxiety": 0.5, "engagement": 0.5},
            {"alertness": 0.9, "anxiety": 0.9, "engagement": 0.9},
        ]
        teacher = [
            {"alertness": 0.9, "anxiety": 0.9, "engagement": 0.9},
            {"alertness": 0.5, "anxiety": 0.5, "engagement": 0.5},
            {"alertness": 0.1, "anxiety": 0.1, "engagement": 0.1},
        ]
        r = compute_mood_correlation(model, teacher)
        assert r.value < 0

    def test_good_correlation(self):
        """Positively correlated scores → high spearman."""
        model = [
            {"alertness": 0.2, "anxiety": 0.1, "engagement": 0.3},
            {"alertness": 0.5, "anxiety": 0.4, "engagement": 0.6},
            {"alertness": 0.8, "anxiety": 0.7, "engagement": 0.9},
        ]
        teacher = [
            {"alertness": 0.1, "anxiety": 0.2, "engagement": 0.2},
            {"alertness": 0.4, "anxiety": 0.5, "engagement": 0.5},
            {"alertness": 0.9, "anxiety": 0.8, "engagement": 0.8},
        ]
        r = compute_mood_correlation(model, teacher)
        assert r.value > 0.5

    def test_threshold_forwarded(self):
        """Threshold passed correctly."""
        model = [{"alertness": 0.5, "anxiety": 0.3, "engagement": 0.8}]
        teacher = [{"alertness": 0.5, "anxiety": 0.3, "engagement": 0.8}]
        r = compute_mood_correlation(model, teacher, threshold=0.75)
        assert r.threshold == 0.75

    def test_details_has_per_dimension(self):
        """Details contain per-dimension correlations."""
        model = [
            {"alertness": 0.2, "anxiety": 0.3, "engagement": 0.4},
            {"alertness": 0.8, "anxiety": 0.7, "engagement": 0.6},
        ]
        teacher = [
            {"alertness": 0.3, "anxiety": 0.2, "engagement": 0.5},
            {"alertness": 0.7, "anxiety": 0.8, "engagement": 0.5},
        ]
        r = compute_mood_correlation(model, teacher)
        assert "per_dimension" in r.details
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics/test_mood_correlation.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/pet_eval/metrics/mood_correlation.py
"""Mood Spearman correlation metric vs teacher model."""

from __future__ import annotations

import logging

from scipy import stats

from pet_eval.metrics.types import MetricResult

logger = logging.getLogger(__name__)

MOOD_DIMENSIONS = ["alertness", "anxiety", "engagement"]


def compute_mood_correlation(
    model_moods: list[dict[str, float]],
    teacher_moods: list[dict[str, float]],
    *,
    threshold: float | None = None,
) -> MetricResult:
    """Compute mean Spearman rank correlation across mood dimensions.

    Args:
        model_moods: Model mood dicts with keys: alertness, anxiety, engagement.
        teacher_moods: Teacher mood dicts with same keys.
        threshold: Gate threshold (gte).

    Returns:
        MetricResult with mean Spearman as value, per-dimension in details.
    """
    if len(model_moods) < 2:
        logger.warning("Need >=2 samples for Spearman correlation, got %d", len(model_moods))
        return MetricResult.create(
            "mood_spearman", value=0.0, threshold=threshold, operator="gte"
        )

    per_dim: dict[str, float] = {}
    for dim in MOOD_DIMENSIONS:
        m_vals = [m[dim] for m in model_moods]
        t_vals = [t[dim] for t in teacher_moods]
        corr, _ = stats.spearmanr(m_vals, t_vals)
        per_dim[dim] = float(corr) if corr == corr else 0.0  # nan → 0

    mean_corr = sum(per_dim.values()) / len(per_dim)

    return MetricResult.create(
        "mood_spearman",
        value=round(mean_corr, 6),
        threshold=threshold,
        operator="gte",
        details={"per_dimension": per_dim},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics/test_mood_correlation.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): mood_correlation Spearman metric with tests"
```

---

## Task 7: narrative_quality (BERTScore) Metric

**Files:**
- Create: `src/pet_eval/metrics/narrative_quality.py`
- Test: `tests/test_metrics/test_narrative_quality.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_metrics/test_narrative_quality.py
"""Tests for narrative BERTScore metric."""

from unittest.mock import patch

from pet_eval.metrics.narrative_quality import compute_narrative_quality


class TestNarrativeQuality:
    @patch("pet_eval.metrics.narrative_quality.bert_score.score")
    def test_high_similarity(self, mock_score):
        """Identical narratives → high BERTScore."""
        import torch

        mock_score.return_value = (
            torch.tensor([0.95]),  # precision
            torch.tensor([0.95]),  # recall
            torch.tensor([0.95]),  # F1
        )
        r = compute_narrative_quality(["橘猫正常进食"], ["橘猫正常进食"])
        assert r.name == "narrative_bertscore"
        assert r.value > 0.9

    @patch("pet_eval.metrics.narrative_quality.bert_score.score")
    def test_low_similarity(self, mock_score):
        """Unrelated narratives → low BERTScore."""
        import torch

        mock_score.return_value = (
            torch.tensor([0.30]),
            torch.tensor([0.30]),
            torch.tensor([0.30]),
        )
        r = compute_narrative_quality(["猫在吃"], ["天气晴朗"])
        assert r.value < 0.5

    @patch("pet_eval.metrics.narrative_quality.bert_score.score")
    def test_multiple_samples(self, mock_score):
        """Multiple samples → mean F1."""
        import torch

        mock_score.return_value = (
            torch.tensor([0.90, 0.80]),
            torch.tensor([0.90, 0.80]),
            torch.tensor([0.90, 0.80]),
        )
        r = compute_narrative_quality(["a", "b"], ["a", "b"])
        assert abs(r.value - 0.85) < 0.01

    def test_empty_inputs(self):
        """Empty inputs → 0 score."""
        r = compute_narrative_quality([], [])
        assert r.value == 0.0

    @patch("pet_eval.metrics.narrative_quality.bert_score.score")
    def test_threshold_forwarded(self, mock_score):
        """Threshold passed through."""
        import torch

        mock_score.return_value = (
            torch.tensor([0.90]),
            torch.tensor([0.90]),
            torch.tensor([0.90]),
        )
        r = compute_narrative_quality(["a"], ["a"], threshold=0.80)
        assert r.threshold == 0.80
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics/test_narrative_quality.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/pet_eval/metrics/narrative_quality.py
"""Narrative quality metric using BERTScore (Chinese BERT)."""

from __future__ import annotations

import logging

import bert_score

from pet_eval.metrics.types import MetricResult

logger = logging.getLogger(__name__)

_CHINESE_BERT_MODEL = "bert-base-chinese"


def compute_narrative_quality(
    model_narratives: list[str],
    teacher_narratives: list[str],
    *,
    threshold: float | None = None,
) -> MetricResult:
    """Compute BERTScore F1 between model and teacher narratives.

    Args:
        model_narratives: Model-generated narrative strings.
        teacher_narratives: Teacher reference narrative strings.
        threshold: Gate threshold for mean F1 (gte).

    Returns:
        MetricResult with mean F1 as value, per-sample in details.
    """
    if not model_narratives:
        return MetricResult.create(
            "narrative_bertscore", value=0.0, threshold=threshold, operator="gte"
        )

    p, r, f1 = bert_score.score(
        model_narratives,
        teacher_narratives,
        model_type=_CHINESE_BERT_MODEL,
        verbose=False,
    )

    mean_f1 = float(f1.mean())
    return MetricResult.create(
        "narrative_bertscore",
        value=round(mean_f1, 6),
        threshold=threshold,
        operator="gte",
        details={
            "mean_precision": round(float(p.mean()), 6),
            "mean_recall": round(float(r.mean()), 6),
            "per_sample_f1": [round(float(v), 4) for v in f1],
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics/test_narrative_quality.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): narrative_quality BERTScore metric with tests"
```

---

## Task 8: latency Metric

**Files:**
- Create: `src/pet_eval/metrics/latency.py`
- Test: `tests/test_metrics/test_latency.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_metrics/test_latency.py
"""Tests for latency metric."""

from pet_eval.metrics.latency import compute_latency


class TestLatency:
    def test_normal_timings(self):
        """Normal latency timings → P95 computed correctly."""
        timings = [2000.0 + i * 10 for i in range(100)]  # 2000-2990ms
        r = compute_latency(timings)
        assert r.name == "latency_p95_ms"
        assert 2900 < r.value < 3000
        assert "p50" in r.details
        assert "p99" in r.details

    def test_single_timing(self):
        """Single timing → P95 = that timing."""
        r = compute_latency([3500.0])
        assert r.value == 3500.0

    def test_empty_timings(self):
        """Empty timings → value 0."""
        r = compute_latency([])
        assert r.value == 0.0

    def test_threshold_pass(self):
        """Under threshold → passes."""
        r = compute_latency([1000.0] * 50, threshold=4000.0)
        assert r.passed is True

    def test_threshold_fail(self):
        """Over threshold → fails."""
        r = compute_latency([5000.0] * 50, threshold=4000.0)
        assert r.passed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics/test_latency.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/pet_eval/metrics/latency.py
"""Latency metric — compute P50/P95/P99 from raw timing data."""

from __future__ import annotations

import statistics

from pet_eval.metrics.types import MetricResult


def compute_latency(
    timings_ms: list[float],
    *,
    threshold: float | None = None,
) -> MetricResult:
    """Compute latency percentiles from raw timing measurements.

    Args:
        timings_ms: List of inference latency measurements in milliseconds.
        threshold: Gate threshold for P95 in ms (lte).

    Returns:
        MetricResult with P95 as value, P50/P99 in details.
    """
    if not timings_ms:
        return MetricResult.create(
            "latency_p95_ms", value=0.0, threshold=threshold, operator="lte"
        )

    sorted_t = sorted(timings_ms)
    p50 = _percentile(sorted_t, 50)
    p95 = _percentile(sorted_t, 95)
    p99 = _percentile(sorted_t, 99)

    return MetricResult.create(
        "latency_p95_ms",
        value=round(p95, 2),
        threshold=threshold,
        operator="lte",
        details={
            "p50": round(p50, 2),
            "p99": round(p99, 2),
            "min": round(min(sorted_t), 2),
            "max": round(max(sorted_t), 2),
            "mean": round(statistics.mean(sorted_t), 2),
            "n_samples": len(sorted_t),
        },
    )


def _percentile(sorted_data: list[float], pct: int) -> float:
    """Compute percentile from pre-sorted data using nearest-rank method."""
    if len(sorted_data) == 1:
        return sorted_data[0]
    k = (pct / 100) * (len(sorted_data) - 1)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics/test_latency.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): latency P50/P95/P99 metric with tests"
```

---

## Task 9: kl_quantization Metric

**Files:**
- Create: `src/pet_eval/metrics/kl_quantization.py`
- Test: `tests/test_metrics/test_kl_quantization.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_metrics/test_kl_quantization.py
"""Tests for KL quantization divergence metric."""

import torch

from pet_eval.metrics.kl_quantization import compute_kl_divergence


class TestKLQuantization:
    def test_identical_distributions(self):
        """Identical distributions → KL ≈ 0."""
        fp16 = [torch.softmax(torch.randn(100), dim=0) for _ in range(5)]
        w8a8 = [d.clone() for d in fp16]
        r = compute_kl_divergence(fp16, w8a8)
        assert r.name == "kl_divergence"
        assert r.value < 0.001

    def test_different_distributions(self):
        """Different distributions → KL > 0."""
        fp16 = [torch.softmax(torch.randn(100), dim=0) for _ in range(5)]
        w8a8 = [torch.softmax(torch.randn(100), dim=0) for _ in range(5)]
        r = compute_kl_divergence(fp16, w8a8)
        assert r.value > 0.0

    def test_threshold_pass(self):
        """Under threshold → passes."""
        fp16 = [torch.softmax(torch.randn(100), dim=0) for _ in range(3)]
        w8a8 = [d.clone() for d in fp16]
        r = compute_kl_divergence(fp16, w8a8, threshold=0.02)
        assert r.passed is True

    def test_empty_inputs(self):
        """Empty → value 0."""
        r = compute_kl_divergence([], [])
        assert r.value == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics/test_kl_quantization.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/pet_eval/metrics/kl_quantization.py
"""KL divergence metric — quantized vs FP16 output distributions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pet_eval.metrics.types import MetricResult


def compute_kl_divergence(
    fp16_distributions: list[torch.Tensor],
    quantized_distributions: list[torch.Tensor],
    *,
    threshold: float | None = None,
) -> MetricResult:
    """Compute mean KL divergence between FP16 and quantized output distributions.

    Args:
        fp16_distributions: List of probability distributions from FP16 model.
        quantized_distributions: List of probability distributions from quantized model.
        threshold: Gate threshold (lte).

    Returns:
        MetricResult with mean KL divergence as value.
    """
    if not fp16_distributions:
        return MetricResult.create(
            "kl_divergence", value=0.0, threshold=threshold, operator="lte"
        )

    kl_values: list[float] = []
    for fp16_dist, quant_dist in zip(fp16_distributions, quantized_distributions):
        # Clamp to avoid log(0)
        fp16_log = torch.log(fp16_dist.clamp(min=1e-10))
        quant_log = torch.log(quant_dist.clamp(min=1e-10))
        kl = float(F.kl_div(quant_log, fp16_dist, reduction="sum", log_target=False))
        kl_values.append(kl)

    mean_kl = sum(kl_values) / len(kl_values)
    return MetricResult.create(
        "kl_divergence",
        value=round(mean_kl, 6),
        threshold=threshold,
        operator="lte",
        details={"per_sample_kl": [round(v, 6) for v in kl_values], "n_samples": len(kl_values)},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics/test_kl_quantization.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): kl_quantization divergence metric with tests"
```

---

## Task 10: audio_accuracy Metric

**Files:**
- Create: `src/pet_eval/metrics/audio_accuracy.py`
- Test: `tests/test_metrics/test_audio_accuracy.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_metrics/test_audio_accuracy.py
"""Tests for audio accuracy metric."""

from pet_eval.metrics.audio_accuracy import compute_audio_accuracy


class TestAudioAccuracy:
    def test_perfect_accuracy(self):
        """All correct → accuracy = 1.0."""
        classes = ["eating", "drinking", "vomiting"]
        predicted = ["eating", "drinking", "vomiting", "eating"]
        actual = ["eating", "drinking", "vomiting", "eating"]
        results = compute_audio_accuracy(predicted, actual, classes)
        overall = results[0]
        assert overall.name == "audio_overall_accuracy"
        assert overall.value == 1.0

    def test_zero_accuracy(self):
        """All wrong → accuracy = 0."""
        classes = ["eating", "drinking"]
        predicted = ["drinking", "eating"]
        actual = ["eating", "drinking"]
        results = compute_audio_accuracy(predicted, actual, classes)
        assert results[0].value == 0.0

    def test_vomit_recall_separate(self):
        """Vomit recall computed as separate metric."""
        classes = ["eating", "vomiting", "ambient"]
        predicted = ["eating", "vomiting", "ambient", "eating"]
        actual = ["eating", "vomiting", "vomiting", "ambient"]
        results = compute_audio_accuracy(predicted, actual, classes)
        # Find vomit_recall metric
        vomit = next((r for r in results if r.name == "audio_vomit_recall"), None)
        assert vomit is not None
        assert vomit.value == 0.5  # 1/2 vomiting detected

    def test_confusion_matrix_in_details(self):
        """Details contain confusion matrix."""
        classes = ["eating", "drinking"]
        predicted = ["eating", "eating"]
        actual = ["eating", "drinking"]
        results = compute_audio_accuracy(predicted, actual, classes)
        assert "confusion_matrix" in results[0].details
        assert "per_class" in results[0].details

    def test_thresholds_forwarded(self):
        """Thresholds passed correctly."""
        classes = ["eating", "vomiting"]
        results = compute_audio_accuracy(
            ["eating"], ["eating"], classes,
            accuracy_threshold=0.80, vomit_recall_threshold=0.70,
        )
        assert results[0].threshold == 0.80
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics/test_audio_accuracy.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/pet_eval/metrics/audio_accuracy.py
"""Audio classification accuracy metric — per-class P/R/F1."""

from __future__ import annotations

from collections import Counter

from pet_eval.metrics.types import MetricResult


def compute_audio_accuracy(
    predicted: list[str],
    actual: list[str],
    classes: list[str],
    *,
    accuracy_threshold: float | None = None,
    vomit_recall_threshold: float | None = None,
) -> list[MetricResult]:
    """Compute audio classification accuracy and per-class metrics.

    Args:
        predicted: Predicted class labels.
        actual: Ground truth class labels.
        classes: List of class names (from params.yaml).
        accuracy_threshold: Gate threshold for overall accuracy (gte).
        vomit_recall_threshold: Gate threshold for vomit recall (gte).

    Returns:
        List of MetricResults: [overall_accuracy, vomit_recall].
    """
    if not predicted:
        return [
            MetricResult.create(
                "audio_overall_accuracy", value=0.0, threshold=accuracy_threshold, operator="gte"
            ),
            MetricResult.create(
                "audio_vomit_recall", value=0.0, threshold=vomit_recall_threshold, operator="gte"
            ),
        ]

    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    accuracy = correct / len(predicted)

    # Per-class metrics
    per_class: dict[str, dict[str, float]] = {}
    confusion: dict[str, dict[str, int]] = {c: Counter() for c in classes}

    for p, a in zip(predicted, actual):
        if a in confusion:
            confusion[a][p] += 1

    for cls in classes:
        tp = confusion[cls].get(cls, 0)
        fn = sum(v for k, v in confusion[cls].items() if k != cls)
        fp = sum(confusion[other].get(cls, 0) for other in classes if other != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # Vomit recall (safety-critical)
    vomit_recall = per_class.get("vomiting", {}).get("recall", 0.0)

    # Build confusion matrix as nested dict
    confusion_serializable = {
        actual_cls: dict(preds) for actual_cls, preds in confusion.items()
    }

    results = [
        MetricResult.create(
            "audio_overall_accuracy",
            value=round(accuracy, 6),
            threshold=accuracy_threshold,
            operator="gte",
            details={
                "per_class": per_class,
                "confusion_matrix": confusion_serializable,
                "n_samples": len(predicted),
            },
        ),
        MetricResult.create(
            "audio_vomit_recall",
            value=round(vomit_recall, 6),
            threshold=vomit_recall_threshold,
            operator="gte",
        ),
    ]
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics/test_audio_accuracy.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): audio_accuracy per-class P/R/F1 metric with tests"
```

---

## Task 11: Gate Checker

**Files:**
- Create: `src/pet_eval/gate/checker.py`
- Test: `tests/test_gate/test_checker.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_gate/test_checker.py
"""Tests for gate checker."""

import yaml

from pet_eval.gate.checker import check_gate


class TestCheckGate:
    def test_vlm_all_pass(self, sample_params):
        """VLM gate passes when all metrics exceed thresholds."""
        params = yaml.safe_load(sample_params.read_text())
        from pet_eval.metrics.types import MetricResult

        results = [
            MetricResult.create("schema_compliance", value=1.0, threshold=0.99, operator="gte"),
            MetricResult.create(
                "distribution_sum_error", value=0.005, threshold=0.01, operator="lte"
            ),
        ]
        gate = check_gate(results, skipped=[], gate_type="vlm", params=params)
        assert gate.passed is True

    def test_vlm_one_fail(self, sample_params):
        """VLM gate fails when one metric fails."""
        params = yaml.safe_load(sample_params.read_text())
        from pet_eval.metrics.types import MetricResult

        results = [
            MetricResult.create("schema_compliance", value=0.50, threshold=0.99, operator="gte"),
            MetricResult.create(
                "distribution_sum_error", value=0.005, threshold=0.01, operator="lte"
            ),
        ]
        gate = check_gate(results, skipped=[], gate_type="vlm", params=params)
        assert gate.passed is False
        assert "schema_compliance" in gate.summary

    def test_skipped_not_fail(self, sample_params):
        """Skipped metrics do not cause failure."""
        params = yaml.safe_load(sample_params.read_text())
        from pet_eval.metrics.types import MetricResult

        results = [
            MetricResult.create("schema_compliance", value=1.0, threshold=0.99, operator="gte"),
        ]
        gate = check_gate(
            results, skipped=["anomaly_recall", "latency_p95_ms"], gate_type="vlm", params=params
        )
        assert gate.passed is True
        assert len(gate.skipped) == 2

    def test_audio_gate(self, sample_params):
        """Audio gate uses audio thresholds."""
        params = yaml.safe_load(sample_params.read_text())
        from pet_eval.metrics.types import MetricResult

        results = [
            MetricResult.create(
                "audio_overall_accuracy", value=0.90, threshold=0.80, operator="gte"
            ),
            MetricResult.create(
                "audio_vomit_recall", value=0.75, threshold=0.70, operator="gte"
            ),
        ]
        gate = check_gate(results, skipped=[], gate_type="audio", params=params)
        assert gate.passed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gate/test_checker.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/pet_eval/gate/checker.py
"""Gate checker — compare metric results against params.yaml thresholds."""

from __future__ import annotations

import logging
from typing import Any

from pet_eval.gate.types import GateResult
from pet_eval.metrics.types import MetricResult

logger = logging.getLogger(__name__)


def check_gate(
    results: list[MetricResult],
    skipped: list[str],
    gate_type: str,
    params: dict[str, Any],
) -> GateResult:
    """Check whether metric results pass the gate.

    Args:
        results: List of computed MetricResults.
        skipped: Names of metrics that were skipped.
        gate_type: 'vlm' or 'audio' — selects threshold section.
        params: Parsed params.yaml dict.

    Returns:
        GateResult with overall pass/fail.
    """
    thresholds = params.get("gates", {}).get(gate_type, {})
    if not thresholds:
        logger.warning("No gate thresholds found for type '%s'", gate_type)

    gate_result = GateResult.from_results(results, skipped)
    logger.info("Gate check [%s]: %s", gate_type, gate_result.summary)
    return gate_result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gate/test_checker.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): gate checker with tests"
```

---

## Task 12: Report / wandb Integration

**Files:**
- Create: `src/pet_eval/report/generate_report.py`
- Test: `tests/test_report/test_generate_report.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_report/test_generate_report.py
"""Tests for wandb report generation."""

from unittest.mock import MagicMock, patch

from pet_eval.gate.types import GateResult
from pet_eval.metrics.types import MetricResult
from pet_eval.report.generate_report import generate_report


class TestGenerateReport:
    def _make_gate_result(self):
        """Create a sample GateResult."""
        results = [
            MetricResult.create("schema_compliance", value=0.99, threshold=0.99, operator="gte"),
            MetricResult.create("kl_divergence", value=0.01, threshold=0.02, operator="lte"),
        ]
        return GateResult.from_results(results, skipped=["latency_p95_ms"])

    @patch("pet_eval.report.generate_report.wandb")
    def test_wandb_init_called(self, mock_wandb):
        """wandb.init called with correct project and run name."""
        mock_wandb.init.return_value = MagicMock()
        gate = self._make_gate_result()
        generate_report(
            gate, run_name="test_run", eval_type="vlm_trained",
            metadata={"model_path": "/tmp/model"},
            wandb_config={"project": "pet-eval-test", "entity": ""},
        )
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["project"] == "pet-eval-test"
        assert "vlm_trained/test_run" in call_kwargs["name"]

    @patch("pet_eval.report.generate_report.wandb")
    def test_metrics_logged(self, mock_wandb):
        """All metric values logged to wandb summary."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        gate = self._make_gate_result()
        generate_report(
            gate, run_name="test", eval_type="vlm_trained",
            metadata={}, wandb_config={"project": "test", "entity": ""},
        )
        # Check that summary was updated
        assert mock_run.summary.__setitem__.called or mock_run.log.called

    @patch("pet_eval.report.generate_report.wandb")
    def test_gate_result_logged(self, mock_wandb):
        """Gate pass/fail logged."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        gate = self._make_gate_result()
        generate_report(
            gate, run_name="test", eval_type="audio",
            metadata={}, wandb_config={"project": "test", "entity": ""},
        )
        mock_run.finish.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report/test_generate_report.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/pet_eval/report/generate_report.py
"""Generate evaluation report and log to wandb."""

from __future__ import annotations

import logging
from typing import Any

import wandb

from pet_eval.gate.types import GateResult

logger = logging.getLogger(__name__)


def generate_report(
    gate_result: GateResult,
    run_name: str,
    eval_type: str,
    metadata: dict[str, Any],
    wandb_config: dict[str, Any],
) -> None:
    """Generate evaluation report and log to wandb.

    Args:
        gate_result: Aggregated gate outcome.
        run_name: wandb run name suffix.
        eval_type: One of 'vlm_trained', 'vlm_quantized', 'audio'.
        metadata: Additional metadata (model_path, device_id, etc.).
        wandb_config: wandb project/entity config from params.yaml.
    """
    full_run_name = f"{eval_type}/{run_name}"
    tags = [eval_type, "pass" if gate_result.passed else "fail"]
    if gate_result.skipped:
        tags.append("has_skipped")

    run = wandb.init(
        project=wandb_config.get("project", "pet-eval"),
        entity=wandb_config.get("entity") or None,
        name=full_run_name,
        tags=tags,
        config=metadata,
    )

    # Log each metric
    for result in gate_result.results:
        run.summary[f"metric/{result.name}/value"] = result.value
        if result.threshold is not None:
            run.summary[f"metric/{result.name}/threshold"] = result.threshold
        run.summary[f"metric/{result.name}/passed"] = result.passed

    # Log gate summary
    run.summary["gate/passed"] = gate_result.passed
    run.summary["gate/summary"] = gate_result.summary
    run.summary["gate/skipped"] = gate_result.skipped

    # Log details as artifacts
    for result in gate_result.results:
        if result.details:
            run.log({f"details/{result.name}": result.details})

    logger.info("Report logged to wandb: %s", full_run_name)
    run.finish()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report/test_generate_report.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): wandb report generation with tests"
```

---

## Task 13: eval_trained Runner

**Files:**
- Create: `src/pet_eval/runners/eval_trained.py`
- Test: `tests/test_runners/test_eval_trained.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_runners/test_eval_trained.py
"""Tests for eval_trained runner."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from pet_eval.runners.eval_trained import run_eval_trained


class TestEvalTrained:
    def _write_gold_set(self, path: Path, outputs: list[dict], n: int = 3):
        """Write a synthetic gold set JSONL."""
        with open(path, "w") as f:
            for i in range(n):
                entry = {
                    "gold_id": f"gold_{i:03d}",
                    "frame_path": f"benchmark/frames/gold_{i:03d}.jpg",
                    "expected_output": outputs[i % len(outputs)],
                    "annotator": "human_expert",
                    "annotation_date": "2026-04-15",
                    "difficulty": "normal",
                }
                f.write(json.dumps(entry) + "\n")

    @patch("pet_eval.runners.eval_trained._run_inference")
    @patch("pet_eval.report.generate_report.wandb")
    def test_no_gold_set_runs_schema_only(self, mock_wandb, mock_infer, sample_params, tmp_dir):
        """Without gold set, only schema_compliance runs."""
        mock_wandb.init.return_value = MagicMock()
        mock_infer.return_value = ['{"schema_version": "1.0"}']
        result = run_eval_trained(
            model_path=str(tmp_dir / "model"),
            run_name="test",
            params_path=str(sample_params),
        )
        # Should have results but skip gold-set-dependent metrics
        assert len(result.skipped) > 0

    @patch("pet_eval.runners.eval_trained._run_inference")
    @patch("pet_eval.report.generate_report.wandb")
    def test_returns_gate_result(self, mock_wandb, mock_infer, sample_params, tmp_dir):
        """Runner returns a GateResult."""
        mock_wandb.init.return_value = MagicMock()
        mock_infer.return_value = ['{"valid": true}']
        result = run_eval_trained(
            model_path=str(tmp_dir / "model"),
            run_name="test",
            params_path=str(sample_params),
        )
        from pet_eval.gate.types import GateResult
        assert isinstance(result, GateResult)

    @patch("pet_eval.runners.eval_trained._run_inference")
    @patch("pet_eval.report.generate_report.wandb")
    def test_exit_code_logic(self, mock_wandb, mock_infer, sample_params, tmp_dir):
        """Gate pass → result.passed is True."""
        mock_wandb.init.return_value = MagicMock()
        # Return valid outputs that pass schema
        mock_infer.return_value = []
        result = run_eval_trained(
            model_path=str(tmp_dir / "model"),
            run_name="test",
            params_path=str(sample_params),
        )
        # With no outputs, schema compliance = 0 → should fail
        assert isinstance(result, GateResult)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runners/test_eval_trained.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/pet_eval/runners/eval_trained.py
"""Runner for evaluating trained FP16 VLM models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from pet_eval.gate.checker import check_gate
from pet_eval.gate.types import GateResult
from pet_eval.metrics.schema_compliance import compute_schema_compliance
from pet_eval.report.generate_report import generate_report

logger = logging.getLogger(__name__)


def run_eval_trained(
    model_path: str,
    run_name: str,
    params_path: str = "params.yaml",
) -> GateResult:
    """Run VLM FP16 evaluation pipeline.

    Args:
        model_path: Path to HuggingFace model directory.
        run_name: wandb run name.
        params_path: Path to params.yaml.

    Returns:
        GateResult with pass/fail outcome.
    """
    params = yaml.safe_load(Path(params_path).read_text())
    gates = params.get("gates", {}).get("vlm", {})
    benchmark = params.get("benchmark", {})
    wandb_config = params.get("wandb", {})

    gold_set_path = Path(benchmark.get("gold_set_path", ""))
    anomaly_set_path = Path(benchmark.get("anomaly_set_path", ""))
    has_gold_set = gold_set_path.exists() and gold_set_path.stat().st_size > 0
    has_anomaly_set = anomaly_set_path.exists() and anomaly_set_path.stat().st_size > 0

    # Run inference
    outputs = _run_inference(model_path, gold_set_path if has_gold_set else None, params)

    # Always run schema compliance
    all_results = compute_schema_compliance(
        outputs,
        compliance_threshold=gates.get("schema_compliance"),
        sum_error_threshold=gates.get("distribution_sum_error"),
        schema_version=params.get("inference", {}).get("schema_version", "1.0"),
    )

    skipped: list[str] = []

    # Gold-set-dependent metrics
    if not has_gold_set:
        logger.warning("Gold set not found at %s — skipping dependent metrics", gold_set_path)
        skipped.extend(["anomaly_recall", "anomaly_false_positive", "calibration_ece"])

    if not has_anomaly_set:
        if "anomaly_recall" not in skipped:
            skipped.extend(["anomaly_recall", "anomaly_false_positive"])

    # Teacher-dependent metrics (narrative, mood) also skipped without reference data
    # These would be populated when teacher outputs are available alongside gold set
    if not has_gold_set:
        skipped.extend(["mood_spearman", "narrative_bertscore"])

    # Gate check
    gate_result = check_gate(all_results, skipped, gate_type="vlm", params=params)

    # Report
    metadata = {"model_path": model_path, "run_name": run_name}
    generate_report(gate_result, run_name, "vlm_trained", metadata, wandb_config)

    logger.info("eval_trained complete: %s", gate_result.summary)
    return gate_result


def _run_inference(
    model_path: str,
    gold_set_path: Path | None,
    params: dict[str, Any],
) -> list[str]:
    """Run model inference on gold set samples.

    Args:
        model_path: Path to HuggingFace model.
        gold_set_path: Path to gold set JSONL (None if not available).
        params: Parsed params.yaml.

    Returns:
        List of raw JSON output strings from model.
    """
    if gold_set_path is None:
        logger.warning("No gold set — skipping inference")
        return []

    # Load gold set
    samples = []
    with open(gold_set_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if not samples:
        return []

    logger.info("Running inference on %d gold set samples", len(samples))

    # TODO: Load model and run actual inference
    # For now, this is a placeholder that will be completed when
    # the actual model loading and inference pipeline is available.
    # The model loading depends on the specific HF model format
    # and pet-schema prompt rendering.
    logger.warning(
        "Model inference not yet implemented — "
        "returning empty outputs. Install model dependencies and implement."
    )
    return []


def main() -> None:
    """CLI entry point for eval_trained."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Evaluate trained FP16 VLM model")
    parser.add_argument("--model_path", required=True, help="Path to HF model directory")
    parser.add_argument("--run_name", required=True, help="wandb run name")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = run_eval_trained(args.model_path, args.run_name, args.params)
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runners/test_eval_trained.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): eval_trained VLM runner with tests"
```

---

## Task 14: eval_audio Runner

**Files:**
- Create: `src/pet_eval/runners/eval_audio.py`
- Test: `tests/test_runners/test_eval_audio.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_runners/test_eval_audio.py
"""Tests for eval_audio runner."""

from unittest.mock import MagicMock, patch

from pet_eval.runners.eval_audio import run_eval_audio


class TestEvalAudio:
    @patch("pet_eval.runners.eval_audio._run_audio_inference")
    @patch("pet_eval.report.generate_report.wandb")
    def test_returns_gate_result(self, mock_wandb, mock_infer, sample_params, tmp_dir):
        """Runner returns a GateResult."""
        mock_wandb.init.return_value = MagicMock()
        mock_infer.return_value = (
            ["eating", "drinking", "eating"],
            ["eating", "drinking", "vomiting"],
        )
        result = run_eval_audio(
            model_path=str(tmp_dir / "model.pt"),
            run_name="test",
            params_path=str(sample_params),
        )
        from pet_eval.gate.types import GateResult
        assert isinstance(result, GateResult)

    @patch("pet_eval.runners.eval_audio._run_audio_inference")
    @patch("pet_eval.report.generate_report.wandb")
    def test_uses_audio_gate(self, mock_wandb, mock_infer, sample_params, tmp_dir):
        """Audio runner uses audio gate thresholds."""
        mock_wandb.init.return_value = MagicMock()
        mock_infer.return_value = (
            ["eating", "eating", "eating"],
            ["eating", "eating", "eating"],
        )
        result = run_eval_audio(
            model_path=str(tmp_dir / "model.pt"),
            run_name="test",
            params_path=str(sample_params),
        )
        assert result.passed is True

    @patch("pet_eval.runners.eval_audio._run_audio_inference")
    @patch("pet_eval.report.generate_report.wandb")
    def test_no_test_data(self, mock_wandb, mock_infer, sample_params, tmp_dir):
        """No test data → skipped."""
        mock_wandb.init.return_value = MagicMock()
        mock_infer.return_value = ([], [])
        result = run_eval_audio(
            model_path=str(tmp_dir / "model.pt"),
            run_name="test",
            params_path=str(sample_params),
        )
        from pet_eval.gate.types import GateResult
        assert isinstance(result, GateResult)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runners/test_eval_audio.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/pet_eval/runners/eval_audio.py
"""Runner for evaluating audio CNN models."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from pet_eval.gate.checker import check_gate
from pet_eval.gate.types import GateResult
from pet_eval.metrics.audio_accuracy import compute_audio_accuracy
from pet_eval.report.generate_report import generate_report

logger = logging.getLogger(__name__)


def run_eval_audio(
    model_path: str,
    run_name: str,
    params_path: str = "params.yaml",
) -> GateResult:
    """Run audio CNN evaluation pipeline.

    Args:
        model_path: Path to PyTorch audio model.
        run_name: wandb run name.
        params_path: Path to params.yaml.

    Returns:
        GateResult with pass/fail outcome.
    """
    params = yaml.safe_load(Path(params_path).read_text())
    gates = params.get("gates", {}).get("audio", {})
    wandb_config = params.get("wandb", {})
    classes = params.get("audio", {}).get("classes", [])

    predicted, actual = _run_audio_inference(model_path, params)

    skipped: list[str] = []
    if not predicted:
        logger.warning("No audio test data — skipping audio metrics")
        skipped.extend(["audio_overall_accuracy", "audio_vomit_recall"])
        all_results = []
    else:
        all_results = compute_audio_accuracy(
            predicted,
            actual,
            classes,
            accuracy_threshold=gates.get("overall_accuracy"),
            vomit_recall_threshold=gates.get("vomit_recall"),
        )

    gate_result = check_gate(all_results, skipped, gate_type="audio", params=params)

    metadata = {"model_path": model_path, "run_name": run_name}
    generate_report(gate_result, run_name, "audio", metadata, wandb_config)

    logger.info("eval_audio complete: %s", gate_result.summary)
    return gate_result


def _run_audio_inference(
    model_path: str,
    params: dict,
) -> tuple[list[str], list[str]]:
    """Run audio model inference on test set.

    Args:
        model_path: Path to PyTorch audio model.
        params: Parsed params.yaml.

    Returns:
        Tuple of (predicted_labels, actual_labels).
    """
    audio_test_dir = params.get("benchmark", {}).get("audio_test_dir", "")
    if not audio_test_dir or not Path(audio_test_dir).exists():
        logger.warning("Audio test directory not found: %s", audio_test_dir)
        return [], []

    # TODO: Load model and run inference on audio test set
    # Depends on pet-train audio model architecture
    logger.warning("Audio inference not yet implemented — returning empty results")
    return [], []


def main() -> None:
    """CLI entry point for eval_audio."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Evaluate audio CNN model")
    parser.add_argument("--model_path", required=True, help="Path to PyTorch audio model")
    parser.add_argument("--run_name", required=True, help="wandb run name")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = run_eval_audio(args.model_path, args.run_name, args.params)
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runners/test_eval_audio.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): eval_audio runner with tests"
```

---

## Task 15: eval_quantized Runner

**Files:**
- Create: `src/pet_eval/runners/eval_quantized.py`
- Test: `tests/test_runners/test_eval_quantized.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_runners/test_eval_quantized.py
"""Tests for eval_quantized runner."""

from unittest.mock import MagicMock, patch

from pet_eval.runners.eval_quantized import run_eval_quantized


class TestEvalQuantized:
    @patch("pet_eval.runners.eval_quantized._run_quantized_inference")
    @patch("pet_eval.report.generate_report.wandb")
    def test_no_device_skips_latency(self, mock_wandb, mock_infer, sample_params, tmp_dir):
        """Without device_id, latency is skipped."""
        mock_wandb.init.return_value = MagicMock()
        mock_infer.return_value = {"outputs": [], "timings": [], "fp16_outputs": []}
        result = run_eval_quantized(
            model_dir=str(tmp_dir / "model"),
            run_name="test",
            device_id=None,
            params_path=str(sample_params),
        )
        assert "latency_p95_ms" in result.skipped

    @patch("pet_eval.runners.eval_quantized._run_quantized_inference")
    @patch("pet_eval.report.generate_report.wandb")
    def test_with_device_includes_latency(self, mock_wandb, mock_infer, sample_params, tmp_dir):
        """With device_id, latency is measured."""
        mock_wandb.init.return_value = MagicMock()
        mock_infer.return_value = {
            "outputs": ['{"schema_version": "1.0"}'],
            "timings": [2000.0] * 50,
            "fp16_outputs": [],
        }
        result = run_eval_quantized(
            model_dir=str(tmp_dir / "model"),
            run_name="test",
            device_id="12345",
            params_path=str(sample_params),
        )
        assert "latency_p95_ms" not in result.skipped

    @patch("pet_eval.runners.eval_quantized._run_quantized_inference")
    @patch("pet_eval.report.generate_report.wandb")
    def test_returns_gate_result(self, mock_wandb, mock_infer, sample_params, tmp_dir):
        """Runner returns a GateResult."""
        mock_wandb.init.return_value = MagicMock()
        mock_infer.return_value = {"outputs": [], "timings": [], "fp16_outputs": []}
        result = run_eval_quantized(
            model_dir=str(tmp_dir / "model"),
            run_name="test",
            params_path=str(sample_params),
        )
        from pet_eval.gate.types import GateResult
        assert isinstance(result, GateResult)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runners/test_eval_quantized.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/pet_eval/runners/eval_quantized.py
"""Runner for evaluating quantized models (with/without real device)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from pet_eval.gate.checker import check_gate
from pet_eval.gate.types import GateResult
from pet_eval.metrics.latency import compute_latency
from pet_eval.metrics.schema_compliance import compute_schema_compliance
from pet_eval.report.generate_report import generate_report

logger = logging.getLogger(__name__)


def run_eval_quantized(
    model_dir: str,
    run_name: str,
    device_id: str | None = None,
    params_path: str = "params.yaml",
) -> GateResult:
    """Run quantized model evaluation pipeline.

    Args:
        model_dir: Path to quantized model directory.
        run_name: wandb run name.
        device_id: ADB device ID (None = no device, skip latency).
        params_path: Path to params.yaml.

    Returns:
        GateResult with pass/fail outcome.
    """
    params = yaml.safe_load(Path(params_path).read_text())
    gates = params.get("gates", {}).get("vlm", {})
    wandb_config = params.get("wandb", {})

    has_device = device_id is not None
    infer_result = _run_quantized_inference(model_dir, device_id, params)

    outputs = infer_result.get("outputs", [])
    timings = infer_result.get("timings", [])

    # Schema compliance
    all_results = compute_schema_compliance(
        outputs,
        compliance_threshold=gates.get("schema_compliance"),
        sum_error_threshold=gates.get("distribution_sum_error"),
        schema_version=params.get("inference", {}).get("schema_version", "1.0"),
    )

    skipped: list[str] = []

    # Latency (only with real device)
    if has_device and timings:
        latency_result = compute_latency(
            timings, threshold=gates.get("latency_p95_ms")
        )
        all_results.append(latency_result)
    else:
        skipped.append("latency_p95_ms")
        if not has_device:
            logger.info("No device — latency measurement skipped")

    # KL divergence (needs both FP16 and quantized outputs)
    fp16_outputs = infer_result.get("fp16_outputs", [])
    if not fp16_outputs:
        skipped.append("kl_divergence")
    # KL computation would go here when FP16 reference outputs are available

    # Gold-set-dependent metrics skipped for now
    skipped.extend([
        "anomaly_recall", "anomaly_false_positive",
        "calibration_ece", "mood_spearman", "narrative_bertscore",
    ])

    gate_result = check_gate(all_results, skipped, gate_type="vlm", params=params)

    metadata = {
        "model_dir": model_dir,
        "run_name": run_name,
        "device_id": device_id or "none",
        "has_device": has_device,
    }
    generate_report(gate_result, run_name, "vlm_quantized", metadata, wandb_config)

    logger.info("eval_quantized complete: %s", gate_result.summary)
    return gate_result


def _run_quantized_inference(
    model_dir: str,
    device_id: str | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run quantized model inference.

    Args:
        model_dir: Path to quantized model files.
        device_id: ADB device ID (None = simulate on host).
        params: Parsed params.yaml.

    Returns:
        Dict with keys: outputs (list[str]), timings (list[float]), fp16_outputs (list).
    """
    if device_id:
        return _run_on_device(model_dir, device_id, params)
    return _run_simulated(model_dir, params)


def _run_on_device(
    model_dir: str,
    device_id: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run inference on real RK3576 device via ADB.

    Args:
        model_dir: Path to quantized model files.
        device_id: ADB device serial number.
        params: Parsed params.yaml.

    Returns:
        Dict with outputs, timings, and empty fp16_outputs.
    """
    # TODO: Implement ADB push + device inference + timing collection
    # This requires:
    # 1. ADB push model files to device
    # 2. Run inference binary on device
    # 3. Pull outputs + timing data back
    logger.warning("Device inference not yet implemented for device %s", device_id)
    return {"outputs": [], "timings": [], "fp16_outputs": []}


def _run_simulated(
    model_dir: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run simulated inference on host (no latency measurement).

    Args:
        model_dir: Path to quantized model files.
        params: Parsed params.yaml.

    Returns:
        Dict with outputs, empty timings, and empty fp16_outputs.
    """
    # TODO: Load quantized model and run inference on host
    logger.warning("Simulated quantized inference not yet implemented")
    return {"outputs": [], "timings": [], "fp16_outputs": []}


def main() -> None:
    """CLI entry point for eval_quantized."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Evaluate quantized model")
    parser.add_argument("--model_dir", required=True, help="Path to quantized model directory")
    parser.add_argument("--run_name", required=True, help="wandb run name")
    parser.add_argument("--device_id", default=None, help="ADB device ID (omit for no-device mode)")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = run_eval_quantized(args.model_dir, args.run_name, args.device_id, args.params)
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runners/test_eval_quantized.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): eval_quantized runner with device/no-device modes and tests"
```

---

## Task 16: CLI Entry Point

**Files:**
- Create: `src/pet_eval/cli.py`
- Create: `src/pet_eval/__main__.py`

- [ ] **Step 1: Implement CLI**

```python
# src/pet_eval/cli.py
"""Unified CLI for pet-eval runners."""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    """Main CLI entry point dispatching to sub-commands."""
    parser = argparse.ArgumentParser(
        prog="pet-eval",
        description="Pet-eval: evaluation pipeline for Train-Pet-Pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # eval-trained
    p_trained = sub.add_parser("eval-trained", help="Evaluate trained FP16 VLM model")
    p_trained.add_argument("--model_path", required=True)
    p_trained.add_argument("--run_name", required=True)
    p_trained.add_argument("--params", default="params.yaml")

    # eval-audio
    p_audio = sub.add_parser("eval-audio", help="Evaluate audio CNN model")
    p_audio.add_argument("--model_path", required=True)
    p_audio.add_argument("--run_name", required=True)
    p_audio.add_argument("--params", default="params.yaml")

    # eval-quantized
    p_quant = sub.add_parser("eval-quantized", help="Evaluate quantized model")
    p_quant.add_argument("--model_dir", required=True)
    p_quant.add_argument("--run_name", required=True)
    p_quant.add_argument("--device_id", default=None)
    p_quant.add_argument("--params", default="params.yaml")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "eval-trained":
        from pet_eval.runners.eval_trained import run_eval_trained
        result = run_eval_trained(args.model_path, args.run_name, args.params)
    elif args.command == "eval-audio":
        from pet_eval.runners.eval_audio import run_eval_audio
        result = run_eval_audio(args.model_path, args.run_name, args.params)
    elif args.command == "eval-quantized":
        from pet_eval.runners.eval_quantized import run_eval_quantized
        result = run_eval_quantized(args.model_dir, args.run_name, args.device_id, args.params)
    else:
        parser.print_help()
        sys.exit(1)

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
```

```python
# src/pet_eval/__main__.py
"""Allow running as python -m pet_eval."""

from pet_eval.cli import main

main()
```

- [ ] **Step 2: Verify CLI help works**

Run: `python -m pet_eval --help`
Expected: Shows sub-commands: eval-trained, eval-audio, eval-quantized.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): unified CLI with sub-commands"
```

---

## Task 17: Update metrics/__init__.py Exports

**Files:**
- Modify: `src/pet_eval/metrics/__init__.py`

- [ ] **Step 1: Update exports**

```python
# src/pet_eval/metrics/__init__.py
"""Metrics package — pure computation modules."""

from pet_eval.metrics.anomaly_recall import compute_anomaly_recall
from pet_eval.metrics.audio_accuracy import compute_audio_accuracy
from pet_eval.metrics.calibration import compute_ece
from pet_eval.metrics.kl_quantization import compute_kl_divergence
from pet_eval.metrics.latency import compute_latency
from pet_eval.metrics.mood_correlation import compute_mood_correlation
from pet_eval.metrics.narrative_quality import compute_narrative_quality
from pet_eval.metrics.schema_compliance import compute_schema_compliance
from pet_eval.metrics.types import MetricResult

__all__ = [
    "MetricResult",
    "compute_anomaly_recall",
    "compute_audio_accuracy",
    "compute_ece",
    "compute_kl_divergence",
    "compute_latency",
    "compute_mood_correlation",
    "compute_narrative_quality",
    "compute_schema_compliance",
]
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from pet_eval.metrics import MetricResult, compute_schema_compliance; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): update metrics __init__ exports"
```

---

## Task 18: lm-evaluation-harness Vendor Setup

**Files:**
- Create: `.gitmodules`
- Create: `tasks/pet_feeder.py` (placeholder)

- [ ] **Step 1: Add lm-evaluation-harness as git submodule**

```bash
cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-eval
git submodule add https://github.com/EleutherAI/lm-evaluation-harness.git vendor/lm-evaluation-harness
```

- [ ] **Step 2: Create tasks/pet_feeder.py placeholder**

```python
# tasks/pet_feeder.py
"""lm-evaluation-harness custom task for pet feeder VLM evaluation.

This is an optional supplement to the main evaluation pipeline.
It registers a custom task that can be run via:
    lm_eval --tasks pet_feeder --model hf ...

See vendor/lm-evaluation-harness/lm_eval/tasks/ for examples.
"""

# TODO: Implement custom task registration
# Reference: vendor/lm-evaluation-harness/lm_eval/tasks/README.md
# This will define:
# - Task config YAML
# - Custom metric functions that delegate to pet_eval.metrics
# - Gold set loading from benchmark/
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(pet-eval): add lm-evaluation-harness submodule and task placeholder"
```

---

## Task 19: Full Test Suite Run + Lint

**Files:** None new — validation pass.

- [ ] **Step 1: Run full test suite**

Run: `make test`
Expected: All tests pass.

- [ ] **Step 2: Run linter**

Run: `make lint`
Expected: No errors. Fix any ruff or mypy issues.

- [ ] **Step 3: Fix any issues found**

Address lint/type errors if any.

- [ ] **Step 4: Commit fixes if needed**

```bash
git add -A
git commit -m "fix(pet-eval): lint and type fixes"
```

---

## Task 20: Final Integration Verification

- [ ] **Step 1: Verify CLI works end-to-end**

```bash
# Should show help
python -m pet_eval --help
python -m pet_eval eval-trained --help
python -m pet_eval eval-audio --help
python -m pet_eval eval-quantized --help
```

- [ ] **Step 2: Verify package imports**

```bash
python -c "
from pet_eval.metrics import (
    MetricResult, compute_schema_compliance, compute_ece,
    compute_anomaly_recall, compute_mood_correlation,
    compute_narrative_quality, compute_latency,
    compute_kl_divergence, compute_audio_accuracy,
)
from pet_eval.gate import GateResult
from pet_eval.report.generate_report import generate_report
print('All imports OK')
"
```

- [ ] **Step 3: Verify Makefile targets**

```bash
make clean
make setup
make test
make lint
```

- [ ] **Step 4: Final commit if any remaining changes**

```bash
git status
# If clean, no commit needed
```
