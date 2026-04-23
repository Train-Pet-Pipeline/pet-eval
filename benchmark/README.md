# Benchmark Gold Set

## Gold Set Format

Each entry in `gold_set_v2.jsonl` and `anomaly_set_v1.jsonl` is a single JSON object per line with the following structure:

```json
{
  "id": "<unique-string>",
  "source": "<source-identifier>",
  "annotator": "<human-expert-id>",
  "created_at": "<ISO-8601-date>",
  "vlm_output": { /* PetFeederEvent JSON matching schema v1.0 */ },
  "ground_truth": {
    "action": "<ActionLabel>",
    "anomaly_present": true | false,
    "anomaly_type": "<string or null>",
    "mood_alertness": 0.0,
    "mood_anxiety": 0.0,
    "mood_engagement": 0.0
  },
  "narrative_reference": "<human-written reference narrative>"
}
```

## Admission Rules

1. **Human expert only** — entries must be annotated by a qualified human expert; no automated or model-generated ground truth is permitted.
2. **Immutable entries** — once an entry is committed to the gold set, it must not be modified. If a correction is needed, the entry must be retired (marked `"retired": true`) and replaced by a new entry with a new `id`.
3. **Anomaly composition** — the anomaly set must maintain at least 70% real (hardware-confirmed or veterinarian-confirmed) anomaly cases; synthetic edge cases may not exceed 30%.
4. **No training overlap** — no gold set entry may originate from data used in any training or fine-tuning run. The `source` field must be verifiable against the training data registry in `pet-data`.

## Current State

The gold set files (`benchmark/gold_set_v2.jsonl`, `benchmark/anomaly_set_v1.jsonl`, `benchmark/teacher_references_v1.jsonl`) are **not yet populated**. Path versions are authoritative in `params.yaml` (`benchmark.*_path`); this document tracks them.

These files are excluded from git via `.gitignore` (`*.jsonl`) and must be obtained separately from the annotation team or downloaded via the benchmark download script (to be implemented in a future task).

Audio test data directory is configured via `params.yaml` (`benchmark.audio_test_dir`).
