# pet-eval Architecture

## §1 Repository Responsibility

**pet-eval** is the evaluation stage of the smart pet feeder pipeline.

It ships two registry-backed surfaces consumed by the `pet_infra` orchestrator:

1. **8 metric plugins** (`pet_infra.registry.METRICS`) — schema compliance, anomaly recall,
   mood correlation, narrative quality (BERTScore), latency P95, audio accuracy,
   KL quantization, calibration ECE.
2. **6 evaluator plugins** (`pet_infra.registry.EVALUATORS`) — 3 primary
   (`vlm_evaluator`, `audio_evaluator`, `quantized_vlm_evaluator`) + 3 rule-based cross-modal
   fusion (`single_modal_fusion`, `and_gate_fusion`, `weighted_fusion`).

Pipeline position:

```
pet-train → [pet-eval] → pet-quantize → pet-ota
              ▲
              └─ also consumes pet_quantize.inference.rkllm_runner at runtime
                 (QuantizedVlmEvaluator; cross-repo peer-dep)
```

**Does:**
- Loads VLM / audio / RKLLM-quantized checkpoints, runs inference against a gold set,
  computes metrics, applies gate via `apply_gate(min_*/max_*)`, emits an updated `ModelCard`.
- Registers all plugins under the `pet_infra.plugins` entry point so the orchestrator
  discovers them automatically (`plugins/_register.py:register_all`).
- Reads every numeric threshold / sample rate / max_new_tokens from `params.yaml`.

**Does not:**
- Train checkpoints (pet-train) or convert them to edge formats (pet-quantize).
- Ship its own orchestrator or stage runner — `pet_infra.orchestrator` drives plugin calls.
- Track experiments itself — `pet_infra.orchestrator + ClearMLLogger` is the sole logging path.

---

## §2 I/O Contract

### Upstream dependencies

| Dependency | Mode | Locked version |
|---|---|---|
| pet-schema | β peer-dep (not in pyproject.dependencies) | v3.2.1 (compatibility_matrix 2026.09) |
| pet-infra | β peer-dep (not in pyproject.dependencies) | v2.6.0 (compatibility_matrix 2026.09) |
| pet-train | runtime cross-repo peer-dep (in pyproject.dependencies, no pin) | v2.0.2 |
| pet-quantize | runtime cross-repo peer-dep (in pyproject.dependencies, no pin) | v2.0.1 |

The CI install order (`.github/workflows/ci.yml` / `peer-dep-smoke.yml`) is 6-step:
schema → infra → train → quantize → editable `--no-deps` → editable dev extras,
with a step-7 last-wins re-pin of pet-schema + pet-infra and a step-8 version assert.

### Inputs

| Source | Consumer | Notes |
|---|---|---|
| `input_card: ModelCard` (pet-schema) | all evaluators | must carry `checkpoint_uri` for VLM/Audio or `edge_artifacts[format=rkllm]` for QuantizedVlm |
| `gold_set_path` → JSONL | `VLMEvaluator` / `QuantizedVlmEvaluator` | each record: `image`/`images[]`, `prompt`/`instruction`, `system` |
| `{audio_test_dir}/{class_name}/*.{wav,mp3,flac,ogg}` | `AudioEvaluator` | subdir name = ground-truth label |
| `params.yaml` | every plugin | thresholds, sample_rate, max_new_tokens, prompt_source, retry config |

### Outputs

- `ModelCard` (pet-schema) with merged `metrics` + `gate_status ∈ {passed, failed}`
  + `task ∈ {vlm_eval, audio_eval, quantized_vlm_eval}` + `modality="audio"` for audio.

### Downstream consumers

- **pet-quantize:** consumes ModelCard.metrics + gate_status for INT8 conversion gating.
- **pet-infra orchestrator:** drives stage execution through `stage_executor.execute_stage`
  → `EvaluatorStageRunner` → `plugin_cls(**config).run(input_card, recipe)`.

---

## §3 Architecture Overview

### Directory tree

```
src/pet_eval/
├── __init__.py                            ← __version__ = "2.3.0"
└── plugins/
    ├── __init__.py
    ├── _register.py                       ← register_all; peer-dep guards; entry point target
    ├── gate.py                            ← apply_gate(metrics, thresholds) → GateResult
    ├── vlm_evaluator.py                   ← @EVALUATORS "vlm_evaluator"
    ├── vlm_inference.py                   ← run_inference / _load_model / _FALLBACK_OUTPUT
    ├── audio_evaluator.py                 ← @EVALUATORS "audio_evaluator" (peer-dep pet_train)
    ├── quantized_vlm_evaluator.py         ← @EVALUATORS "quantized_vlm_evaluator"
    ├── quantized_vlm_inference.py         ← lazy import pet_quantize.inference.rkllm_runner
    ├── fusion/
    │   ├── base.py                        ← BaseFusionEvaluator (abstract)
    │   ├── single_modal.py                ← @EVALUATORS "single_modal_fusion"
    │   ├── and_gate.py                    ← @EVALUATORS "and_gate_fusion"
    │   └── weighted.py                    ← @EVALUATORS "weighted_fusion"
    └── metrics/
        ├── types.py                       ← MetricResult dataclass (frozen)
        ├── schema_compliance.py           ← @METRICS "schema_compliance"
        ├── anomaly_recall.py              ← @METRICS "anomaly_recall"
        ├── mood_correlation.py            ← @METRICS "mood_correlation"
        ├── narrative_quality.py           ← @METRICS "narrative_quality"  (BERTScore)
        ├── latency.py                     ← @METRICS "latency"  (P50/P95/P99)
        ├── audio_accuracy.py              ← @METRICS "audio_accuracy"
        ├── kl_quantization.py             ← @METRICS "kl_quantization"
        └── calibration.py                 ← @METRICS "calibration"  (ECE, informational)

configs/fusion/weighted.yaml               ← shared config for cross_modal_fusion_eval recipe
recipes/cross_modal_fusion_eval.yaml       ← ExperimentRecipe fixture sweeping 3 strategies
benchmark/README.md                        ← gold set / anomaly set admission rules
params.yaml                                ← gates / benchmark / inference / audio / device
.github/workflows/
├── ci.yml                                 ← 8-step peer-dep install + ruff + mypy + pytest
├── peer-dep-smoke.yml                     ← isolated install-order contract test
└── no-wandb-residue.yml                   ← positive-list CI guard (Phase 6 6B)
```

### High-level dataflow

```
orchestrator                             pet_eval
─────────────                             ──────────
recipe.yaml  ──► compose_recipe ──► stage ──► EvaluatorStageRunner
                                                 │
                                                 │ _load_stage_kwargs(stage)  ← configs/*.yaml
                                                 ▼
                                           plugin_cls(**kwargs)
                                                 │
                                                 ▼
                                        run(input_card, recipe)
                                                 │
                                                 ├──► run_inference(model_path, ...)
                                                 │        │
                                                 │        ▼
                                                 │   outputs: list[str]
                                                 │
                                                 ├──► _compute_metrics → dict[name → float]
                                                 ├──► apply_gate(metrics, thresholds)
                                                 │
                                                 ▼
                                        return ModelCard(metrics=…, gate_status=…)
```

---

## §4 Core Modules

### 4.1 `plugins/_register.py`

The entry-point target declared in `pyproject.toml` under
`[project.entry-points."pet_infra.plugins"] pet_eval = …`. Imports happen in a specific
order so peer-dep guards surface useful errors:

1. pet-schema → hard `RuntimeError` (Mode B must come first — finding ⑦)
2. pet-infra → hard `RuntimeError`
3. pet-train → hard `RuntimeError` (AudioEvaluator cross-repo runtime)
4. pet-quantize → soft `logging.warning` + version-not-2.x warning (partial installs)
5. Import `audio_evaluator`, `quantized_vlm_evaluator`, `vlm_evaluator`, 3 fusion modules,
   and 8 metric modules — each registers via `@EVALUATORS.register_module` /
   `@METRICS.register_module` side-effects.

### 4.2 `plugins/gate.py` — single gate abstraction

`apply_gate(metrics: dict[str, float], thresholds: dict[str, float]) -> GateResult`.
Threshold keys follow a convention: `min_<metric>` → fail if below; `max_<metric>` →
fail if above; other prefixes are ignored (treated as informational metadata). Missing
metrics default to 0 for `min_` checks and `+inf` for `max_` checks — conservative.

### 4.3 Evaluator plugins (3 primary)

All three return an updated `ModelCard` from `.run(input_card, recipe)` with
`task`, `modality` (audio only), merged `metrics`, and `gate_status`.

| Plugin | Inference path | Cross-repo |
|---|---|---|
| `vlm_evaluator` | `vlm_inference.run_inference` → HF transformers + PEFT LoRA merge | — |
| `audio_evaluator` | lazy `from pet_train.audio.inference import AudioInference` | pet-train runtime |
| `quantized_vlm_evaluator` | lazy `from pet_quantize.inference.rkllm_runner import RKLLMRunner` | pet-quantize runtime |

### 4.4 Fusion evaluators (3 rule-based)

All inherit `BaseFusionEvaluator` and implement `fuse(modality_scores) -> float`.
All three accept `**_` so one YAML config file can satisfy the whole ablation sweep
(`configs/fusion/weighted.yaml` carries `weights` + `threshold` + `modality`).

Learned fusion is deliberately out of scope per `feedback_no_learned_fusion` memory
— rule-based is sufficient for current business needs.

### 4.5 Metric plugins (8)

Each metric module defines a pure function (`compute_*`) plus a registry adapter class
wrapping it for `@METRICS.register_module`. Adapters forward `kwargs` so thresholds and
options can flow from registry build args or call-site kwargs.

- Gated metrics produce `MetricResult(name, value, threshold, passed, details)` where
  `passed` is auto-computed from `operator ∈ {gte, lte}`.
- Informational metrics (e.g. `calibration_ece`) pass `threshold=None` → always `passed=True`.

### 4.6 VLM inference helpers (`plugins/vlm_inference.py`)

Four internal helpers + two public ones:

- `_load_model(model_path, params)` — adapter_config.json → base model → PEFT merge;
  auto-picks cuda / mps / cpu.
- `_load_standard_prompts(schema_version)` — calls `pet_schema.render_prompt` (single
  source of truth; preserves train/infer prompt alignment per `project_prompt_alignment`).
- `_build_generate_kwargs(inference_cfg, temperature_override)` / `_generate_one(...)`.
- `run_inference(model_path, gold_set_path, params)` → `list[str]`.
- `validate_output(raw, schema_version)` → `bool` via `pet_schema.validate_output`.
- `_FALLBACK_OUTPUT` — a 50-line safe JSON used when validation fails after retry
  (see §8.1 for the rationale).

---

## §5 Extension Points

### Adding a metric

1. Drop `src/pet_eval/plugins/metrics/<name>.py` with a pure `compute_<name>`
   returning `MetricResult` (or `list[MetricResult]`).
2. Add a registry adapter class decorated `@METRICS.register_module(name="<name>")`.
3. Append `<name>` to the import list in `_register.py:register_all`.
4. Add a threshold row under `params.yaml:gates.vlm` or `.audio` if gated.

### Adding a primary evaluator

1. Create `src/pet_eval/plugins/<new>_evaluator.py`; decorate class with
   `@EVALUATORS.register_module(name="<new>_evaluator")`.
2. Accept `**cfg` in `__init__`, expose `run(input_card, recipe) -> ModelCard`.
3. Reuse `apply_gate` from `plugins.gate`; compute metrics via `METRICS.build({"type": name})`.
4. Append `<new>_evaluator` to `_register.py` and update `peer-dep-smoke.yml`
   expected-evaluator set.

### Adding a fusion strategy

Rule-based only (per `feedback_no_learned_fusion`). Subclass `BaseFusionEvaluator`,
implement `fuse(modality_scores)`, decorate with `@EVALUATORS.register_module(force=True)`,
and add to the `fusion_strategy` ablation axis in `recipes/cross_modal_fusion_eval.yaml`.

---

## §6 Dependency Management

### Pin style

- pet-schema / pet-infra — **β peer-dep**: NOT in `pyproject.dependencies`; installed
  explicitly by CI / conda env in a pinned order. Keeps pet-eval pip-installable in
  dev without chasing minor-version drift.
- pet-train / pet-quantize — **unpinned runtime peer-dep** listed in
  `pyproject.dependencies`: pip is allowed to resolve them, but the `peer-dep-smoke`
  workflow installs canonical versions explicitly so PRs test against the real matrix.

### Install-order contract (§11.4 of DEVELOPMENT_GUIDE)

CI step 7 re-pins pet-schema + pet-infra after pet-quantize (which transitively pulls
older versions). Step 8 asserts both resolved versions startswith the expected major.
`peer-dep-smoke.yml` is the authoritative producer-side gate for this contract.

### Version bump policy

- **patch** — docstring / doc-only changes; no metric or evaluator surface change.
- **minor** — new metric / evaluator / fusion; new optional params.yaml key; dead-code
  removal that does not change live paths consumed by any other repo.
- **major** — change to `ModelCard` contract, `apply_gate` semantics, or removal of a
  registered plugin name.

`test_version_attribute_matches_metadata` enforces parity between
`pet_eval.__version__` and `importlib.metadata.version("pet-eval")`.

---

## §7 Local Dev and Test

```bash
# Prerequisites: conda env pet-pipeline, peer-deps installed from matrix row
conda activate pet-pipeline

# From repo root:
make setup   # pip install -e ".[dev]" --no-deps
make test    # pytest tests/ -v                  (103 tests)
make lint    # ruff check src/ tests/ && mypy src/
make clean   # drop .pytest_cache / .ruff_cache / .mypy_cache / outputs / benchmark_cache
```

Mini-E2E candidate (for regression after refactor, per T6.3 template):

```bash
pytest tests/recipes/test_fusion_recipe.py \
       tests/test_plugins/test_audio_evaluator.py \
       tests/test_plugins/test_vlm_evaluator.py \
       tests/test_plugins/test_quantized_vlm_evaluator.py \
       tests/plugins/fusion/ -v
```

Skips the heavier `narrative_quality` (loads bert-base-chinese) and `kl_quantization`
(pulls torch heavily); covers the registry shape + 3 primary evaluators + all 3 fusion
strategies.

---

## §8 Known Complex Points (Preserved for Good Reasons)

### 8.1 `_FALLBACK_OUTPUT` — 50-line hardcoded safe JSON

**Why preserved:** When `run_inference` is configured with `retry_on_failure: true`
and the retry still produces invalid JSON, `run_inference` must emit *something* the
downstream metric pipeline can parse. Dropping the record silently would mis-measure
compliance_rate (fewer outputs vs fewer valid outputs); returning `None` would force
every metric to add a `None` branch. A sentinel JSON with `"narrative": "VLM output
could not be parsed"` and zeroed distributions preserves the record count and still
fails the schema gate because `id_confidence=0.0`.

**What would be lost by removing:** Silent event drops or every metric function gaining
null-handling. Current design concentrates the non-obvious behavior in one named constant.

**Condition to revisit:** Moving to constrained decoding that guarantees schema-valid
output at the token level (a different architecture — see spec).

### 8.2 Two-tier peer-dep guard (hard fail vs soft warn)

**Why preserved:** `pet-schema / pet-infra / pet-train` are universally required —
missing any of them means the plugin module cannot even load correctly, so a hard
`RuntimeError` at `register_all()` time is the fastest useful failure. `pet-quantize`
is different: partial-install environments (e.g. CI steps that reinstall subsets) can
transiently lack it, and `peer-dep-smoke.yml` is the real contract gate. A hard fail
there would make the package unimportable during those windows.

**What would be lost by removing:** Either the hard errors lose their early-warning
value (if downgraded to warnings) or transient CI steps break (if pet-quantize is
elevated to hard fail).

**Condition to revisit:** pet-quantize becomes required at every import site (not
just `QuantizedVlmEvaluator.run`), or CI stops reinstalling mid-flow.

### 8.3 Cross-repo runtime imports (pet_train.audio, pet_quantize.rkllm)

**Why preserved:** AudioEvaluator needs the real `AudioInference` class (PANNs model
+ predict signature); QuantizedVlmEvaluator needs `RKLLMRunner` lifecycle
(init / generate / release). Duplicating either contract inside pet-eval would violate
the `pet-schema is single source of truth` spirit and double the maintenance cost on
every upstream change. The imports are deferred (AudioEvaluator delays the import
until `__init__` / `run`; QuantizedVlmEvaluator delays until `run_inference` is called)
to keep module-load fast and tolerant of optional SDK absence.

**What would be lost by removing:** Either reimplementation duplication or the loss of
audio / quantized evaluation entirely.

**Condition to revisit:** pet-train or pet-quantize extracts a stable "inference
interface" package that pet-eval can depend on without pulling full training or
conversion code.

### 8.4 Dual `prompt_source` config (`"gold_set"` vs `"pet_schema"`)

**Why preserved:** sft_v2 training data embedded the full PetFeederEvent prompt inside
each record's `prompt` field. sft_v3+ switched to pet_schema's short prompt rendered
via `pet_schema.render_prompt`. Models trained on either generation must be evaluable
without re-baking the gold set. The flag picks the aligned prompt path at inference
time. (Train/infer prompt alignment is a firm project rule — see
`project_prompt_alignment` memory.)

**What would be lost by removing:** Either backward compatibility with sft_v2-trained
checkpoints or the short-prompt regime for new models — a lose-lose.

**Condition to revisit:** All supported checkpoints have been retrained with the
pet_schema short prompt and sft_v2 models are retired.

### 8.5 Lazy heavy imports (torch / transformers / RKLLMRunner)

**Why preserved:** `plugins/_register.py` is executed at *import* time via the
entry point — a top-level `import torch` there would force every downstream tool
(type checkers, test runners, CLI help) to pay a multi-second torch load. Lazy
imports inside `_load_model` / `run_inference` defer the cost to the first actual
`run()` call. Same reasoning for `pet_quantize.inference.rkllm_runner`, which
requires the RK SDK that is deliberately optional in dev.

**What would be lost by removing:** Slow `pet-eval` entry-point discovery across the
orchestrator, and hard failures in dev environments that deliberately skip RK SDK
installation.

**Condition to revisit:** Torch becomes a universal baseline on every runner *and*
the RK SDK is always available (neither is true today).

### 8.6 Metric signature variation + warn-and-skip in `_compute_metrics`

**Why preserved:** Metrics have genuinely heterogeneous signatures — some take
`(outputs,)` (schema_compliance), some take `(predicted, actual)` (audio_accuracy,
anomaly_recall), some take per-sample paired dicts (mood_correlation). Forcing a
single signature either requires evaluators to know the call pattern per metric
(tight coupling) or invents an over-general marshalling layer. The current
warn-and-skip pattern in `_compute_metrics` lets an evaluator mix metrics whose
shape it can feed and gracefully drop the ones it cannot — each skip logs a
`TypeError` reason for visibility.

**What would be lost by removing:** Either a single-shape straitjacket (and the
ability to configure a VLM evaluator with an audio-only metric by mistake would
become a hard error instead of a log line) or a heavier kwargs-marshalling layer
in every evaluator.

**Condition to revisit:** Phase 4 P2-C's "fuller integration (feeding gold_set
references, per-metric arg marshalling)" follow-up noted in `vlm_evaluator.py`
docstring — still a valid future cleanup if the metric catalog keeps growing.

---

## §9 Phase 7+ Follow-ups

1. **`_compute_metrics` arg marshalling** — every primary evaluator has a
   near-identical `_compute_metrics` with the same `zip + try/except TypeError +
   unpack` shape. When a 4th evaluator is added, extract a `_invoke_metric(name, metric,
   *args)` helper in `plugins/gate.py` or a new `plugins/metric_dispatch.py`.

2. **Benchmark download script** — `benchmark/README.md` promises "downloaded via the
   benchmark download script (to be implemented in a future task)". The gold-set-vN
   admission rules are stable; the script is not yet written.

3. **Thresholds naming convention** — `params.yaml:gates.vlm` uses bare metric
   names (`schema_compliance: 0.99`), but `apply_gate` expects `min_` / `max_`
   prefixes. Somewhere between params.yaml and `apply_gate` a caller must prepend
   the prefix; this is currently a responsibility gap and a source of confusion.
   A thin loader in `plugins/gate.py` that normalizes `params["gates"][gate_type]`
   into prefixed thresholds (based on each metric's `operator`) would close it.

4. **`_FALLBACK_OUTPUT` freshness** — the fallback JSON hardcodes `schema_version: "1.0"`
   and the v1.0 PetFeederEvent shape. When schema v2.0 ships, the constant must be
   regenerated. Consider building the fallback dynamically from `pet_schema` defaults
   at import time (and catching the shape-drift failure in a parity test).

5. **Audio `classes` fixture parity** — `tests/conftest.py:sample_params` hardcodes the
   same 5-class list that `params.yaml:audio.classes` owns. A fixture that loads from
   `params.yaml` would prevent silent drift when the audio taxonomy changes.
