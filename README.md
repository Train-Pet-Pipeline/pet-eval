# pet-eval

Evaluation pipeline for Train-Pet-Pipeline. Provides metric plugins, evaluators, and harness integration for VLM, audio, and quantized model evaluation.

## Install

pet-eval is a peer-dep consumer: `pet-schema`, `pet-infra`, `pet-train`, and `pet-quantize` must be installed first with versions from the current matrix row in `pet-infra/docs/compatibility_matrix.yaml`. See `.github/workflows/ci.yml` for the canonical 6-step install order. For local dev in the shared `pet-pipeline` conda env:

```bash
make setup   # pip install -e ".[dev]" — peer-deps expected to be already installed
make test    # pytest tests/ -v
make lint    # ruff + mypy
```

## Architecture & extension points

See `docs/architecture.md` for the full module map, 8 metric plugins, 3 evaluators + 3 fusion evaluators, peer-dep contract, and extension points (METRICS / EVALUATORS registries).

## Plugin entry point

pet-eval registers 8 metric plugins + 6 evaluator plugins (3 primary + 3 fusion) via the `pet_infra.plugins` entry point. The orchestrator discovers them automatically; direct registration is also supported for tests:

```python
from pet_eval.plugins._register import register_all
from pet_infra.registry import METRICS, EVALUATORS

register_all()
print(sorted(METRICS.module_dict))     # 8 metric names
print(sorted(EVALUATORS.module_dict))  # 6 evaluator names
```

## Running an evaluator

Evaluators are instantiated via the registry and invoked by the orchestrator (see `pet-infra/src/pet_infra/orchestrator/runner.py`). A minimal direct call:

```python
from pet_infra.registry import EVALUATORS

evaluator = EVALUATORS.build({
    "type": "vlm_evaluator",
    "metrics": ["schema_compliance", "narrative_quality"],
    "thresholds": {"min_compliance_rate": 0.99},
    "gold_set_path": "benchmark/gold_set_v2.jsonl",
    "params": {"inference": {"max_new_tokens": 1024}},
})
updated_card = evaluator.run(input_card, recipe)
```

Recipe-driven runs: see `recipes/cross_modal_fusion_eval.yaml` for a working fixture that sweeps the three fusion strategies.

## License

This project is licensed under the [Business Source License 1.1](LICENSE) (BSL 1.1).
On **2030-04-22** it converts automatically to the Apache License, Version 2.0.

> Note: BSL 1.1 is **source-available**, not OSI-approved open source.
> Production / commercial use requires a separate commercial license.

![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-blue.svg)
