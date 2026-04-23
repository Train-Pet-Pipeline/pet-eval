# Changelog

All notable changes to pet-eval are documented here.
Format follows Keep a Changelog; versions follow SemVer.

## [2.3.0] - 2026-04-23

Phase 6 — ecosystem optimization pass for pet-eval. Single minor bump
rolling up β peer-dep pin refresh, findings ③/⑤/⑥/⑦/⑧/⑨/⑩, dead-code
removal (⑪/⑫/⑬/⑭/⑮), CI guard additions, and `architecture.md`.

### Added
- `docs/architecture.md` (9-章 template per ecosystem-optimization spec §4.1).
- `configs/fusion/weighted.yaml` — shared config for the
  `cross_modal_fusion_eval` recipe so the 3-strategy sweep is actually
  runnable (finding ⑧).
- `.github/workflows/no-wandb-residue.yml` — positive-list CI guard
  scanning first-party code for `\bwandb\b` matches (retro §7 #8).
- README quick-start section + entry-point example (finding ⑩).
- `test_fusion_recipe_config_path_is_loadable_by_all_three_strategies`
  regression test for finding ⑧.
- `test_version_attribute_matches_metadata` parity test (finding ③).
- pet-schema Mode B peer-dep guard in `plugins/_register.py` with
  companion `tests/test_register_peer_guards.py` (finding ⑦).

### Changed
- `pet_eval.plugins.audio_evaluator` now reads `audio.sample_rate` from
  `params.yaml` instead of the 16000 literal (no-hardcode, finding ⑤).
- β peer-dep pins refreshed to Phase 5/6 current versions:
  `pet-schema @ v3.2.1`, `pet-infra @ v2.6.0`, `pet-train @ v2.0.2`,
  `pet-quantize @ v2.0.1` (Phase 6 6A).
- `plugins/_register.py` docstring: stale "matrix 2026.04 / 2026.07"
  strings replaced with "latest matrix row" pointer (finding ⑥).
- Top-level `pet_eval/__init__.py` docstring updated to describe the
  actual surface (metric + evaluator plugins).
- `benchmark/README.md` aligned with `params.yaml` v2 gold-set path
  (finding ⑨).

### Removed
- `src/pet_eval/gate/` (120 LOC) — dead alternate GateResult/check_gate
  abstraction consumed only by `report/generate_report.py`; all
  evaluator plugins use `pet_eval.plugins.gate.apply_gate`
  (findings ⑪/⑭).
- `src/pet_eval/report/generate_report.py` (75 LOC) — zero callers
  across the monorepo; orchestrator + ClearMLLogger is the real
  experiment-tracking path (finding ⑫).
- `src/pet_eval/inference/` (108 LOC) — `build_constrained_generator`
  had no importers and no tests; `params.yaml
  inference.constrained_decoding` flag was never read; `[constrained]`
  pyproject extra + `outlines>=0.1` dep removed with it (finding ⑬).
- `tests/test_gate/` + `tests/test_report/` (115 LOC) — tests targeting
  the removed dead modules (finding ⑮).

## [2.2.0] - 2026-04-22

Phase 4 sub-phase P2-B (cross-modal fusion evaluators + W&B residue removal).

### Added
- 3 rule-based cross-modal fusion evaluator plugins under
  `pet_eval.plugins.fusion`:
  - `single_modal_fusion` — pass-through of a single modality score
  - `and_gate_fusion` — minimum score across all modalities, gated by threshold
  - `weighted_fusion` — normalized weighted sum (missing modalities count as 0)
  All registered via `@EVALUATORS.register_module(force=True)` (P2-B-2).
- `recipes/cross_modal_fusion_eval.yaml` — `ExperimentRecipe` fixture with a
  `fusion_strategy` ablation axis sweeping the 3 strategies (P2-B-3).
- `tests/recipes/test_fusion_recipe.py` — 3 contract tests validating the
  recipe loads, parses, and exposes the expected ablation axis (P2-B-3).

### Changed
- Peer-dep pins bumped to Phase 4 RC matrix (P2-B-1):
  - `pet-schema` `2.4.x` (was `2.3.x`) — installed from `v2.4.0-rc1`
  - `pet-infra` `2.5.x` (was `2.4.x`) — installed from `v2.5.0-rc1`
- CI install order updated to insert pet-schema before pet-infra and to
  re-pin both peer-deps last-wins after `pet-quantize` v2.0.0 is installed
  (pet-quantize transitively pulls older pins) (P2-B-1).
- Peer-dep version assertions tightened from `startswith('2.')` to
  `startswith('2.4')` / `startswith('2.5')` (P2-B-1).

### Removed
- All physical W&B residue (P2-B-4):
  - `wandb_config` parameter dropped from
    `pet_eval.report.generate_report.generate_report()`.
  - `wandb:` block removed from `params.yaml`.
  - `wandb/` entry removed from `.gitignore` and the `Makefile clean` target.
  - `wandb` mock removed from the `sample_params` fixture in `tests/conftest.py`.

  ClearML (orchestrator P0-B/C) is the sole experiment tracker.
  vendor/lm-evaluation-harness retains its upstream wandb code untouched.

## [2.1.0] - 2026-04-21

Phase 3A — initial plugin port. See git history for details.
