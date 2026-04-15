"""pet_eval.metrics — metric computation and result types."""
from pet_eval.metrics.narrative_quality import compute_narrative_quality
from pet_eval.metrics.types import MetricResult

__all__ = ["MetricResult", "compute_narrative_quality"]
