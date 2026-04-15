# pet-eval Design Spec

> Evaluation pipeline for Train-Pet-Pipeline. Independently evaluates VLM and audio CNN models,
> called by both pet-train (post-training) and pet-quantize (post-quantization).

## 1. Architecture Overview

pet-eval is a standalone evaluation repo with two core responsibilities:

1. **Model-level evaluation** — run metrics independently for VLM and audio CNN
2. **Gate checking** — compare metric results against configurable thresholds from params.yaml

### Directory Structure

```
pet-eval/
├── src/pet_eval/
│   ├── __init__.py
│   ├── cli.py                    # Unified CLI entry point
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── schema_compliance.py  # Schema compliance rate (jsonschema + code-level)
│   │   ├── calibration.py        # ECE (Expected Calibration Error)
│   │   ├── anomaly_recall.py     # Recall + false positive rate on anomaly set
│   │   ├── mood_correlation.py   # Spearman correlation vs 72B teacher
│   │   ├── narrative_quality.py  # BERTScore (Chinese BERT)
│   │   ├── kl_quantization.py    # KL divergence: W8A8 vs FP16
│   │   └── audio_accuracy.py     # Audio CNN: per-class P/R/F1, confusion matrix
│   ├── runners/
│   │   ├── __init__.py
│   │   ├── eval_trained.py       # Evaluate FP16 VLM after training
│   │   ├── eval_quantized.py     # Evaluate quantized model (with/without device)
│   │   └── eval_audio.py         # Evaluate audio CNN
│   ├── gate/
│   │   ├── __init__.py
│   │   └── checker.py            # Gate pass/fail logic, reads params.yaml thresholds
│   └── report/
│       ├── __init__.py
│       └── generate_report.py    # Write results to wandb, link to training run
├── tasks/
│   └── pet_feeder.py             # lm-evaluation-harness custom task (optional)
├── benchmark/
│   └── README.md                 # Gold set format + admission rules (data added later)
├── vendor/                       # lm-evaluation-harness source (git submodule)
├── tests/
├── params.yaml
├── pyproject.toml
├── Makefile
└── .gitignore
```

## 2. Core Data Types

### MetricResult

Unified return type for all metric modules.

```python
@dataclass
class MetricResult:
    name: str           # Metric name (e.g. "schema_compliance")
    value: float        # Computed value
    threshold: float    # Gate threshold from params.yaml
    passed: bool        # Whether value meets threshold
    details: dict       # Additional info (per-class breakdown, etc.)
```

### GateResult

Aggregated gate outcome from checker.

```python
@dataclass
class GateResult:
    passed: bool                    # True only if ALL non-skipped metrics pass
    results: list[MetricResult]     # Per-metric details
    skipped: list[str]              # Skipped metric names (no device, no gold set)
    summary: str                    # Human-readable one-line summary
```

## 3. Metrics Module Design

All metric modules are pure computational functions — they receive data, return MetricResult. No I/O.

### 3.1 VLM Metrics (5 modules)

**schema_compliance.py**
- Input: list of model output JSON strings
- Computation: call `pet_schema.validate_output()` for each, compute pass rate + mean |distribution_sum - 1.0|
- Returns: two MetricResults (compliance_rate, distribution_sum_error)

**calibration.py**
- Input: model outputs (with confidence scores) + gold set labels
- Computation: bin predictions by confidence, compute ECE = mean(|accuracy - confidence|) per bin
- Requires: gold set with ground-truth labels
- Skipped when: gold set not available

**anomaly_recall.py**
- Input: model outputs + anomaly_set labels
- Computation: TP/(TP+FN) for recall, FP/(FP+TN) for false positive rate
- Requires: anomaly_set_v1.jsonl
- Skipped when: anomaly set not available

**mood_correlation.py**
- Input: model mood scores + 72B teacher mood scores
- Computation: Spearman rank correlation across mood dimensions
- Requires: teacher reference outputs
- Skipped when: teacher outputs not available

**narrative_quality.py**
- Input: model narrative strings + teacher narrative strings
- Computation: BERTScore using Chinese BERT model (precision/recall/F1)
- Requires: teacher reference narratives
- Skipped when: teacher outputs not available

### 3.2 Audio Metric (1 module)

**audio_accuracy.py**
- Input: predicted labels + ground-truth labels, class names
- Computation: overall accuracy, per-class precision/recall/F1, confusion matrix
- Returns: MetricResult with details containing per-class breakdown
- Classes: eating, drinking, vomiting, ambient, other

### 3.3 Quantization Metric (1 module)

**kl_quantization.py**
- Input: FP16 output distributions + W8A8 output distributions
- Computation: mean KL divergence across samples
- Used by: eval_quantized runner

## 4. Runners Design

### 4.1 eval_trained.py — VLM FP16 Evaluation

**Caller:** `pet-train/scripts/eval_after_train.sh`

**CLI:**
```bash
python -m pet_eval.runners.eval_trained \
    --model_path <hf_model_dir> \
    --run_name <wandb_run_name>
```

**Flow:**
1. Load HF model from `--model_path`
2. Load gold set from `benchmark/` (if exists)
3. Run inference on each gold set sample using pet-schema prompt templates
4. Compute 5 VLM metrics (skip those requiring unavailable data)
5. Run gate check
6. Generate wandb report (project=pet-eval, run=vlm_trained/{run_name})
7. Print pass/fail summary to stdout, exit code 0=pass / 1=fail

**Gold set missing behavior:** skip anomaly_recall, mood_correlation, calibration. Only run schema_compliance + narrative_quality. Log clear warning.

### 4.2 eval_audio.py — Audio CNN Evaluation

**Caller:** pet-train after audio training

**CLI:**
```bash
python -m pet_eval.runners.eval_audio \
    --model_path <pytorch_model> \
    --run_name <wandb_run_name>
```

**Flow:**
1. Load PyTorch audio model
2. Load audio test set (path from params.yaml)
3. Run inference on test samples
4. Compute audio_accuracy metrics
5. Run gate check (loose thresholds)
6. Generate wandb report (project=pet-eval, run=audio/{run_name})
7. Print summary, exit code 0/1

### 4.3 eval_quantized.py — Quantized Model Evaluation

**Caller:** pet-quantize

**Two modes:**

**Without device (default):**
```bash
python -m pet_eval.runners.eval_quantized \
    --model_dir <quantized_model_dir>
```
- Load quantized model on GPU/CPU for simulated inference
- Run schema_compliance + kl_quantization + other VLM metrics
- Skip latency measurement, mark as "skipped (no device)"
- Gate: latency marked skipped, not failed

**With device:**
```bash
python -m pet_eval.runners.eval_quantized \
    --model_dir <quantized_model_dir> \
    --device_id <adb_device_id>
```
- Push model to RK3576 via ADB
- Run inference on device, collect P95 latency
- Pull outputs back, run all metrics including latency
- Full gate check

## 5. Gate Module Design

### gate/checker.py

Reads thresholds from params.yaml `gates` section. Compares each MetricResult against its threshold.

**Rules:**
- Skipped metrics do not count as failures
- ALL non-skipped metrics must pass for overall gate to pass
- Results written to both wandb summary and stdout
- Final release gate (pet-ota check_gate) still requires latency to pass (no skip allowed)

## 6. params.yaml Structure

```yaml
# === Gate Thresholds ===
gates:
  vlm:
    schema_compliance: 0.99          # >99% pass rate
    distribution_sum_error: 0.01     # mean |sum - 1.0| < 0.01
    anomaly_recall: 0.85             # TP/(TP+FN) > 0.85
    anomaly_false_positive: 0.15     # FP/(FP+TN) < 0.15
    mood_spearman: 0.75              # Spearman correlation > 0.75
    narrative_bertscore: 0.80        # BERTScore F1 > 0.80
    latency_p95_ms: 4000             # P95 < 4s on RK3576
    kl_divergence: 0.02              # W8A8 vs FP16 KL < 0.02
  audio:
    overall_accuracy: 0.80           # Loose initial threshold
    vomit_recall: 0.70               # Vomit class recall (safety-critical, still loose)

# === Benchmark Data ===
benchmark:
  gold_set_path: "benchmark/gold_set_v1.jsonl"
  anomaly_set_path: "benchmark/anomaly_set_v1.jsonl"
  audio_test_dir: ""                 # Path to audio test data (set when available)

# === wandb ===
wandb:
  project: "pet-eval"
  entity: ""

# === Model Inference ===
inference:
  schema_version: "1.0"
  max_new_tokens: 1024
  batch_size: 1                      # Sequential for reproducibility

# === Device ===
device:
  adb_timeout: 30                    # ADB command timeout in seconds
  warmup_runs: 3                     # Warmup runs before latency measurement
  latency_runs: 50                   # Number of runs for P95 calculation
```

## 7. Report & wandb Integration

### report/generate_report.py

```python
def generate_report(
    gate_result: GateResult,
    run_name: str,
    eval_type: str,        # "vlm_trained" | "vlm_quantized" | "audio"
    metadata: dict,        # model_path, device_id, schema_version, etc.
) -> None:
```

**wandb organization:**
- Project: `pet-eval`
- Run name: `{eval_type}/{run_name}` (e.g. `vlm_trained/sft_lora_r16_lr2e4_ep3`)
- Tags: eval_type, schema_version, gate pass/fail
- Summary: each metric value + threshold + pass/fail
- Config: metadata dict (model_path, params.yaml snapshot) for traceability back to training run

## 8. Benchmark / Gold Set Management

### benchmark/README.md contents:

**Gold set format** (`gold_set_v1.jsonl`, one JSON per line):
```json
{
  "gold_id": "gold_001",
  "frame_path": "benchmark/frames/gold_001.jpg",
  "expected_output": { ... },
  "annotator": "human_expert",
  "annotation_date": "2024-01-15",
  "difficulty": "normal",
  "notes": "Typical normal eating, baseline verification"
}
```

**Admission rules:**
- Every entry must be confirmed by a human expert; no VLM-only annotations accepted
- Existing entries are immutable; new samples append to new version files
- `anomaly_set_v1.jsonl` must contain >= 70% real anomaly samples (not synthetic)
- Gold set samples must NEVER appear in any training set

**Current state:** Data files not yet created. Runners gracefully skip gold-set-dependent metrics when files are absent.

## 9. lm-evaluation-harness Integration

- Source code in `vendor/` as git submodule
- `tasks/pet_feeder.py` registers a custom task for standardized evaluation
- **Not part of the main evaluation flow** — optional supplement for community benchmark comparison
- Reusable ecosystem components (logging, result formatting) evaluated during implementation

## 10. Cross-Repo Interface

### Called by pet-train:
```bash
# eval_after_train.sh
python -m pet_eval.runners.eval_trained \
    --model_path "$MODEL_PATH" \
    --run_name "$RUN_NAME"
```

### Called by pet-quantize:
```bash
# Without device
python -m pet_eval.runners.eval_quantized --model_dir "$MODEL_DIR"

# With device
python -m pet_eval.runners.eval_quantized \
    --model_dir "$MODEL_DIR" \
    --device_id "$DEVICE_ID"
```

### Audio evaluation:
```bash
python -m pet_eval.runners.eval_audio \
    --model_path "$AUDIO_MODEL" \
    --run_name "$RUN_NAME"
```

## 11. Dependencies

- `pet-schema` (pinned to version tag) — schema validation
- `torch` + `transformers` — model loading and inference
- `torchaudio` — audio model inference
- `bert-score` — narrative quality evaluation
- `scipy` — Spearman correlation
- `wandb` — experiment tracking and reporting
- `jsonschema` — via pet-schema
- `tenacity` — retry for external calls (ADB, wandb)

## 12. Testing Strategy

- **Unit tests per metric module:** synthetic inputs, verify computation correctness
- **Unit tests for gate checker:** verify threshold comparison logic, skip handling
- **Integration tests for runners:** mock model inference, verify end-to-end flow
- **Test fixtures:** synthetic gold set samples for testing only (NOT in benchmark/)
- **Makefile targets:** setup, test, lint, clean
