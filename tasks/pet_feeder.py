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
