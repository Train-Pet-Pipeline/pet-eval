# Phase 3A v1 Audit — pet-eval

**Date:** 2026-04-21
**Spec:** `pet-infra/docs/superpowers/specs/2026-04-21-phase-3a-training-design.md` §3.1
**Plan:** `pet-infra/docs/superpowers/plans/2026-04-21-phase-3a-training-plan.md` PR #P2-A

## git ls-files (55 files)

```
.github/workflows/ci.yml
.gitignore
.gitmodules
Makefile
benchmark/README.md
docs/superpowers/plans/2026-04-15-pet-eval-implementation.md
docs/superpowers/specs/2026-04-15-pet-eval-design.md
params.yaml
pyproject.toml
src/pet_eval/__init__.py
src/pet_eval/__main__.py
src/pet_eval/cli.py
src/pet_eval/gate/__init__.py
src/pet_eval/gate/checker.py
src/pet_eval/gate/types.py
src/pet_eval/inference/__init__.py
src/pet_eval/inference/constrained.py
src/pet_eval/metrics/__init__.py
src/pet_eval/metrics/anomaly_recall.py
src/pet_eval/metrics/audio_accuracy.py
src/pet_eval/metrics/calibration.py
src/pet_eval/metrics/kl_quantization.py
src/pet_eval/metrics/latency.py
src/pet_eval/metrics/mood_correlation.py
src/pet_eval/metrics/narrative_quality.py
src/pet_eval/metrics/schema_compliance.py
src/pet_eval/metrics/types.py
src/pet_eval/report/__init__.py
src/pet_eval/report/generate_report.py
src/pet_eval/runners/__init__.py
src/pet_eval/runners/eval_audio.py
src/pet_eval/runners/eval_quantized.py
src/pet_eval/runners/eval_trained.py
tasks/pet_feeder.py
tests/__init__.py
tests/conftest.py
tests/test_gate/__init__.py
tests/test_gate/test_checker.py
tests/test_gate/test_types.py
tests/test_metrics/__init__.py
tests/test_metrics/test_anomaly_recall.py
tests/test_metrics/test_audio_accuracy.py
tests/test_metrics/test_calibration.py
tests/test_metrics/test_kl_quantization.py
tests/test_metrics/test_latency.py
tests/test_metrics/test_mood_correlation.py
tests/test_metrics/test_narrative_quality.py
tests/test_metrics/test_schema_compliance.py
tests/test_metrics/test_types.py
tests/test_report/__init__.py
tests/test_report/test_generate_report.py
tests/test_runners/__init__.py
tests/test_runners/test_eval_audio.py
tests/test_runners/test_eval_quantized.py
tests/test_runners/test_eval_trained.py
vendor/lm-evaluation-harness
```

## Classification Table

| Path | Category | Phase 3A Action | Rationale |
|------|----------|-----------------|-----------|
| `.github/workflows/ci.yml` | keep | no change | CI itself is clean; no direct script references |
| `.gitignore` | keep | no change | Infrastructure, no changes needed |
| `.gitmodules` | keep | no change | lm-evaluation-harness submodule definition |
| `Makefile` | keep | no change | No targets reference deleted files; no cleanup needed |
| `benchmark/README.md` | docs | no change | Benchmark documentation |
| `docs/superpowers/plans/2026-04-15-pet-eval-implementation.md` | docs | no change | v1 implementation plan; historical reference |
| `docs/superpowers/specs/2026-04-15-pet-eval-design.md` | docs | no change | v1 design spec; historical reference |
| `docs/phase-3a-audit.md` | docs | new (this file) | Phase 3A audit document |
| `params.yaml` | keep | no change | Central params file; values used by all runners |
| `pyproject.toml` | keep | update — remove `wandb` dep | `wandb` replaced by ClearMLLogger via orchestrator (P0-B/C) |
| `src/pet_eval/__init__.py` | keep | no change | Package init; no v1-only imports |
| `src/pet_eval/__main__.py` | delete-now | delete | Invokes old CLI entry-point; replaced by `pet run` orchestrator |
| `src/pet_eval/cli.py` | delete-now | delete | Argparse v1 CLI; replaced by `pet run` orchestrator |
| `src/pet_eval/gate/__init__.py` | keep | no change | Gate subpackage init |
| `src/pet_eval/gate/checker.py` | migrate-later | keep until P2-C | GateChecker logic moves into VLMEvaluator plugin in P2-C |
| `src/pet_eval/gate/types.py` | migrate-later | keep until P2-C | GateResult/MetricResult types; migrated in P2-C |
| `src/pet_eval/inference/__init__.py` | keep | no change | Inference subpackage init |
| `src/pet_eval/inference/constrained.py` | migrate-later | keep until P2-C | Constrained decoding; migrated to VLMEvaluator plugin in P2-C |
| `src/pet_eval/metrics/__init__.py` | migrate-later | keep until P2-B | Metrics subpackage init; migrated in P2-B |
| `src/pet_eval/metrics/anomaly_recall.py` | migrate-later | keep until P2-B | Migrates to `plugins/metrics/` in P2-B |
| `src/pet_eval/metrics/audio_accuracy.py` | migrate-later | keep until P2-B | Migrates to `plugins/metrics/` in P2-B |
| `src/pet_eval/metrics/calibration.py` | migrate-later | keep until P2-B | Migrates to `plugins/metrics/` in P2-B |
| `src/pet_eval/metrics/kl_quantization.py` | migrate-later | keep until P2-B | Migrates to `plugins/metrics/` in P2-B |
| `src/pet_eval/metrics/latency.py` | migrate-later | keep until P2-B | Migrates to `plugins/metrics/` in P2-B |
| `src/pet_eval/metrics/mood_correlation.py` | migrate-later | keep until P2-B | Migrates to `plugins/metrics/` in P2-B |
| `src/pet_eval/metrics/narrative_quality.py` | migrate-later | keep until P2-B | Migrates to `plugins/metrics/` in P2-B |
| `src/pet_eval/metrics/schema_compliance.py` | migrate-later | keep until P2-B | Migrates to `plugins/metrics/` in P2-B |
| `src/pet_eval/metrics/types.py` | migrate-later | keep until P2-B | MetricResult/GateResult types; migrated in P2-B |
| `src/pet_eval/report/__init__.py` | keep | no change | Report subpackage init |
| `src/pet_eval/report/generate_report.py` | delete-now (partial) | strip wandb inline | Remove wandb import + init/log/finish calls; keep local-JSON fallback as the sole reporting path |
| `src/pet_eval/runners/__init__.py` | keep | no change | Runners subpackage init |
| `src/pet_eval/runners/eval_audio.py` | migrate-later | keep until P2-D | AudioEvaluator plugin migration in P2-D |
| `src/pet_eval/runners/eval_quantized.py` | delete-now | delete | Phase 3B rebuilds as QuantizedModelEvaluator plugin |
| `src/pet_eval/runners/eval_trained.py` | migrate-later | keep until P2-C | VLMEvaluator plugin migration in P2-C |
| `tasks/pet_feeder.py` | keep | no change | lm-evaluation-harness task definition |
| `tests/__init__.py` | keep | no change | Test package init |
| `tests/conftest.py` | keep | no change | Shared fixtures; no imports from deleted modules |
| `tests/test_gate/__init__.py` | keep | no change | |
| `tests/test_gate/test_checker.py` | migrate-later | keep until P2-C | Kept with gate/checker.py |
| `tests/test_gate/test_types.py` | migrate-later | keep until P2-C | Kept with gate/types.py |
| `tests/test_metrics/__init__.py` | migrate-later | keep until P2-B | |
| `tests/test_metrics/test_anomaly_recall.py` | migrate-later | keep until P2-B | P2-B fixture-locks these |
| `tests/test_metrics/test_audio_accuracy.py` | migrate-later | keep until P2-B | P2-B fixture-locks these |
| `tests/test_metrics/test_calibration.py` | migrate-later | keep until P2-B | P2-B fixture-locks these |
| `tests/test_metrics/test_kl_quantization.py` | migrate-later | keep until P2-B | P2-B fixture-locks these |
| `tests/test_metrics/test_latency.py` | migrate-later | keep until P2-B | P2-B fixture-locks these |
| `tests/test_metrics/test_mood_correlation.py` | migrate-later | keep until P2-B | P2-B fixture-locks these |
| `tests/test_metrics/test_narrative_quality.py` | migrate-later | keep until P2-B | P2-B fixture-locks these |
| `tests/test_metrics/test_schema_compliance.py` | migrate-later | keep until P2-B | P2-B fixture-locks these |
| `tests/test_metrics/test_types.py` | migrate-later | keep until P2-B | P2-B fixture-locks these |
| `tests/test_report/__init__.py` | keep | no change | |
| `tests/test_report/test_generate_report.py` | delete-now (full) | delete | All 3 tests mock wandb behavior; entirely wandb-specific |
| `tests/test_runners/__init__.py` | keep | no change | |
| `tests/test_runners/test_eval_audio.py` | migrate-later | keep until P2-D | Kept with eval_audio.py |
| `tests/test_runners/test_eval_quantized.py` | delete-now | delete | Tests for deleted eval_quantized.py |
| `tests/test_runners/test_eval_trained.py` | migrate-later | keep until P2-C | Kept with eval_trained.py |
| `vendor/lm-evaluation-harness` | vendor | no change | Vendored submodule; required by task harness |

## Cross-Import Safety Check

Files being deleted and what imports them in KEEP code:
- `cli.py` — only imported by `__main__.py` (also deleted); no keep-file imports it
- `__main__.py` — not imported by any source file (python -m entry point only)
- `eval_quantized.py` — only invoked by `cli.py` (deleted); no keep-file imports it
- `generate_report.py` (wandb portion) — wandb init/log/finish block removed; local JSON fallback kept; function signature unchanged so callers (`eval_trained.py`, `eval_audio.py`) continue to compile

No keep-file is broken by these deletions.

## Summary Counts

| Category | Count |
|----------|-------|
| delete-now (full file) | 5 |
| delete-now (partial — wandb strip) | 1 |
| migrate-later | 24 |
| keep | 15 |
| vendor | 1 |
| docs | 4 |
| **Total** | **50** |

Files remaining after purge (excluding vendor): ~49 (5 deleted, 1 partially stripped)
