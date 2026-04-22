# Changelog

All notable changes to pet-eval are documented here.
Format follows Keep a Changelog; versions follow SemVer.

## [2.2.0-rc1] - 2026-04-22

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
