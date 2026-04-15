"""pet_eval.metrics — metric computation and result types."""
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
