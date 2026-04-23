"""AudioEvaluator plugin — EVALUATORS-registered adapter using pet_train.audio cross-repo import."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pet_infra.registry import EVALUATORS, METRICS
from pet_schema.model_card import ModelCard

from pet_eval.plugins.gate import apply_gate

_PARAMS_PATH = Path(__file__).parent.parent.parent.parent / "params.yaml"


def _default_sample_rate() -> int:
    """Read audio.sample_rate from params.yaml; fall back to 16000 if unavailable.

    Returns:
        Sample rate as an integer (e.g. 16000).
    """
    try:
        with open(_PARAMS_PATH) as f:
            params = yaml.safe_load(f)
        return int(params["audio"]["sample_rate"])
    except Exception:  # noqa: BLE001
        return 16000

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


@EVALUATORS.register_module(name="audio_evaluator")
class AudioEvaluator:
    """Evaluate audio CNN against a directory of labeled audio clips.

    Iterates a benchmark directory structured as::

        {audio_test_dir}/{class_name}/*.wav

    Each subdirectory name is the ground-truth label. Runs zero-shot
    classification via pet_train.audio.inference.AudioInference, computes
    audio_accuracy (and optionally vomit_recall), applies gate.

    Expected kwargs (via Registry.build):
      audio_test_dir: str — root with {class_name}/ subdirs
      metrics: list[str] — metric registry names to build
      thresholds: dict[str, float] — min_*/max_* gate config
      pretrained_path: str | None — PANNs checkpoint path
      sample_rate: int — default from params.yaml audio.sample_rate (16000 if unavailable)
      device: str | None — override torch device; None = auto-detect
    """

    def __init__(self, **cfg: Any) -> None:
        """Initialise evaluator from registry build kwargs.

        Args:
            **cfg: Keyword arguments from Registry.build.  See class docstring
                for the expected keys.
        """
        self._cfg: dict[str, Any] = dict(cfg)
        # Import CLASSES here (not lazy) so metric build can inject them as defaults.
        # pet_train is a required peer-dep (guarded in _register.py), so this is safe.
        from pet_train.audio.inference import CLASSES as _AUDIO_CLASSES

        metric_names: list[str] = cfg.get("metrics", [])
        self._metrics: list[Any] = [
            METRICS.build({"type": name, "classes": _AUDIO_CLASSES}) for name in metric_names
        ]
        self._metric_names: list[str] = metric_names
        self._thresholds: dict[str, float] = cfg.get("thresholds", {})
        self._audio_test_dir: str | None = cfg.get("audio_test_dir")
        self._pretrained_path: str | None = cfg.get("pretrained_path")
        self._sample_rate: int = int(cfg.get("sample_rate", _default_sample_rate()))
        self._device: str | None = cfg.get("device")

    def run(self, input_card: ModelCard | None, recipe: Any) -> ModelCard:
        """Run evaluation and return an updated ModelCard with metrics + gate_status.

        Args:
            input_card: ModelCard from the preceding training stage.  Must not
                be None; the evaluator needs the checkpoint_uri to locate the
                model weights.
            recipe: Recipe object forwarded by the orchestrator (not used
                directly; kept for interface compatibility).

        Returns:
            Updated ModelCard with metrics merged and gate_status set.

        Raises:
            ValueError: If input_card is None.
        """
        if input_card is None:
            raise ValueError("AudioEvaluator.run requires a trained model_card; got None.")

        # Lazy cross-repo import — only resolved when plugin actually runs.
        # Registry registration via _register.py already loaded the module;
        # this ensures we don't pay the torch import cost at plugin build time.
        from pet_train.audio.inference import AudioInference

        checkpoint = (
            self._pretrained_path or input_card.checkpoint_uri.replace("file://", "") or None
        )
        predicted, actual = self._collect_predictions_and_labels(
            AudioInference(
                pretrained_path=checkpoint,
                device=self._device,
                sample_rate=self._sample_rate,
            )
        )

        metrics_out: dict[str, float] = self._compute_metrics(predicted, actual)
        gate = apply_gate(metrics_out, self._thresholds)

        updated = input_card.metrics.copy()
        updated.update(metrics_out)

        return input_card.model_copy(
            update={
                "metrics": updated,
                "gate_status": "passed" if gate.passed else "failed",
                "task": "audio_eval",
                "modality": "audio",
                "notes": gate.reason if not gate.passed else input_card.notes,
            }
        )

    def _collect_predictions_and_labels(self, inference: Any) -> tuple[list[str], list[str]]:
        """Walk audio_test_dir/{class_name}/*.wav and collect predictions vs ground truth.

        Args:
            inference: AudioInference instance with a predict(audio_path) method.

        Returns:
            Tuple of (predicted_labels, actual_labels) lists.
        """
        if not self._audio_test_dir:
            log.warning("audio_test_dir not configured; returning empty lists")
            return ([], [])
        test_dir = Path(self._audio_test_dir)
        if not test_dir.exists():
            log.warning("audio_test_dir does not exist: %s", test_dir)
            return ([], [])

        predicted: list[str] = []
        actual: list[str] = []
        for class_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            class_name = class_dir.name
            for audio_path in sorted(class_dir.iterdir()):
                if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                try:
                    pred = inference.predict(str(audio_path))
                except Exception as e:
                    log.warning("inference failed on %s: %s", audio_path, e)
                    continue
                predicted.append(pred.label)
                actual.append(class_name)
        return predicted, actual

    def _compute_metrics(self, predicted: list[str], actual: list[str]) -> dict[str, float]:
        """Invoke each registered metric with (predicted, actual); unpack list[MetricResult].

        Args:
            predicted: List of predicted class labels.
            actual: List of ground-truth class labels.

        Returns:
            Dict mapping metric name to float value.
        """
        results: dict[str, float] = {}
        for name, metric in zip(self._metric_names, self._metrics, strict=True):
            try:
                out = metric(predicted, actual)
            except TypeError as e:
                log.warning("metric %s skipped: signature mismatch (%s)", name, e)
                continue
            for mr in out if isinstance(out, list) else [out]:
                if hasattr(mr, "name") and hasattr(mr, "value"):
                    results[mr.name] = float(mr.value)
                else:
                    results[name] = float(mr)
        return results
