from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from pet_eval.plugins.audio_evaluator import AudioEvaluator, _default_sample_rate


@pytest.fixture
def sample_card():
    from pet_schema.model_card import ModelCard

    return ModelCard(
        id="sft-card-1",
        version="1.0.0",
        modality="audio",
        task="sft",
        arch="mobilenetv2_audioset",
        training_recipe="dummy",
        hydra_config_sha="a" * 64,
        git_shas={},
        dataset_versions={},
        checkpoint_uri="file:///tmp/fake_audio_model.pth",
        metrics={},
        gate_status="pending",
        trained_at=datetime.now(UTC),
        trained_by="ci",
    )


def test_cross_repo_import_succeeds():
    # Proves pet-train is installed and audio module is importable
    from pet_train.audio.inference import CLASSES, AudioInference, AudioPrediction

    assert AudioInference is not None
    assert AudioPrediction is not None
    assert len(CLASSES) == 5


def test_registers_to_evaluators():
    from pet_eval.plugins._register import register_all

    register_all()
    from pet_infra.registry import EVALUATORS

    assert "audio_evaluator" in EVALUATORS.module_dict


def test_registry_build_produces_evaluator():
    from pet_eval.plugins._register import register_all

    register_all()
    from pet_infra.registry import EVALUATORS

    evaluator = EVALUATORS.build(
        {
            "type": "audio_evaluator",
            "metrics": ["audio_accuracy"],
            "thresholds": {},
        }
    )
    assert isinstance(evaluator, AudioEvaluator)


def test_init_builds_metrics_from_config():
    from pet_eval.plugins._register import register_all

    register_all()
    evaluator = AudioEvaluator(metrics=["audio_accuracy"], thresholds={})
    assert len(evaluator._metrics) == 1


def test_run_raises_without_input_card():
    evaluator = AudioEvaluator(metrics=[], thresholds={})
    with pytest.raises(ValueError, match="requires a trained model_card"):
        evaluator.run(input_card=None, recipe=SimpleNamespace())


def test_run_returns_updated_card_when_dir_missing(sample_card, tmp_path):
    """When audio_test_dir doesn't exist, the evaluator returns empty metrics
    and applies gate against empty dict (conservative: min_* thresholds would fail)."""
    evaluator = AudioEvaluator(
        metrics=[],
        thresholds={},
        audio_test_dir=str(tmp_path / "nonexistent"),
    )
    # F026: default backend is PANNs — patch PANNsAudioInference (was legacy AudioInference)
    with patch("pet_train.audio.panns_inference_plugin.PANNsAudioInference") as mock_inf:
        mock_inf.return_value = MagicMock()
        card = evaluator.run(input_card=sample_card, recipe=SimpleNamespace())
    assert card.gate_status == "passed"  # no thresholds to violate
    assert card.task == "audio_eval"
    assert card.modality == "audio"


def test_run_uses_panns_backend_by_default(sample_card, tmp_path):
    """F026: AudioEvaluator default backend MUST be PANNs (not the broken legacy class)."""
    evaluator = AudioEvaluator(
        metrics=[],
        thresholds={},
        audio_test_dir=str(tmp_path / "nonexistent"),
    )
    # If we patch the LEGACY class and PANNs is the default, legacy should NOT be called.
    with patch(
        "pet_train.audio.panns_inference_plugin.PANNsAudioInference"
    ) as mock_panns, patch("pet_train.audio.inference.AudioInference") as mock_legacy:
        mock_panns.return_value = MagicMock()
        mock_legacy.return_value = MagicMock()
        evaluator.run(input_card=sample_card, recipe=SimpleNamespace())
    assert mock_panns.called
    assert not mock_legacy.called


def test_run_legacy_backend_opt_in(sample_card, tmp_path):
    """F026: cfg `inference_backend: legacy_mobilenetv2` selects the F008-broken class."""
    evaluator = AudioEvaluator(
        metrics=[],
        thresholds={},
        audio_test_dir=str(tmp_path / "nonexistent"),
        inference_backend="legacy_mobilenetv2",
    )
    with patch(
        "pet_train.audio.panns_inference_plugin.PANNsAudioInference"
    ) as mock_panns, patch("pet_train.audio.inference.AudioInference") as mock_legacy:
        mock_panns.return_value = MagicMock()
        mock_legacy.return_value = MagicMock()
        evaluator.run(input_card=sample_card, recipe=SimpleNamespace())
    assert mock_legacy.called
    assert not mock_panns.called


def test_run_unknown_backend_raises(sample_card, tmp_path):
    """F026: unknown inference_backend → ValueError listing valid options."""
    evaluator = AudioEvaluator(
        metrics=[],
        thresholds={},
        audio_test_dir=str(tmp_path / "nonexistent"),
        inference_backend="bogus",
    )
    with pytest.raises(ValueError, match="unknown inference_backend"):
        evaluator.run(input_card=sample_card, recipe=SimpleNamespace())


def test_run_iterates_audio_files_and_computes_metrics(sample_card, tmp_path):
    """Build a tiny fake audio dir, mock predict() deterministically, verify metrics emerge."""
    from pet_train.audio.inference import AudioPrediction

    audio_root = tmp_path / "audio_bench"
    (audio_root / "eating").mkdir(parents=True)
    (audio_root / "drinking").mkdir(parents=True)
    eating_clip = audio_root / "eating" / "clip1.wav"
    drinking_clip = audio_root / "drinking" / "clip2.wav"
    eating_clip.write_bytes(b"fake wav header")
    drinking_clip.write_bytes(b"fake wav header")

    predicted_labels = {
        str(eating_clip): AudioPrediction(
            label="eating",
            confidence=0.9,
            class_scores={
                "eating": 0.9,
                "drinking": 0.05,
                "vomiting": 0.01,
                "ambient": 0.02,
                "other": 0.02,
            },
        ),
        str(drinking_clip): AudioPrediction(
            label="drinking",
            confidence=0.85,
            class_scores={
                "eating": 0.1,
                "drinking": 0.85,
                "vomiting": 0.01,
                "ambient": 0.02,
                "other": 0.02,
            },
        ),
    }

    def fake_predict(path):
        return predicted_labels[path]

    fake_inference_instance = MagicMock()
    fake_inference_instance.predict.side_effect = fake_predict

    from pet_eval.plugins._register import register_all

    register_all()
    evaluator = AudioEvaluator(
        metrics=["audio_accuracy"],
        thresholds={},
        audio_test_dir=str(audio_root),
    )

    # F026: default backend is now PANNs; mock the upstream class
    with patch(
        "pet_train.audio.panns_inference_plugin.PANNsAudioInference",
        return_value=fake_inference_instance,
    ):
        card = evaluator.run(input_card=sample_card, recipe=SimpleNamespace())

    # Both predictions match ground truth → 100% accuracy.
    # AudioAccuracyMetric emits "audio_overall_accuracy" and "audio_vomit_recall".
    assert "audio_overall_accuracy" in card.metrics
    assert card.metrics["audio_overall_accuracy"] == pytest.approx(1.0, abs=0.01)
    assert card.gate_status == "passed"


def test_default_sample_rate_reads_from_params_yaml(tmp_path: Path) -> None:
    """_default_sample_rate() must return params.yaml audio.sample_rate, not a hardcode."""
    import pet_eval.plugins.audio_evaluator as mod

    fake_params = tmp_path / "params.yaml"
    fake_params.write_text(yaml.dump({"audio": {"sample_rate": 22050}}))

    original_path = mod._PARAMS_PATH
    try:
        mod._PARAMS_PATH = fake_params
        assert _default_sample_rate() == 22050
    finally:
        mod._PARAMS_PATH = original_path


def test_audio_evaluator_uses_sample_rate_from_params(tmp_path: Path) -> None:
    """AudioEvaluator without explicit sample_rate cfg should pick up params.yaml value."""
    import pet_eval.plugins.audio_evaluator as mod

    fake_params = tmp_path / "params.yaml"
    fake_params.write_text(yaml.dump({"audio": {"sample_rate": 8000}}))

    original_path = mod._PARAMS_PATH
    try:
        mod._PARAMS_PATH = fake_params
        evaluator = AudioEvaluator(metrics=[], thresholds={})
        assert evaluator._sample_rate == 8000
    finally:
        mod._PARAMS_PATH = original_path


def test_audio_evaluator_cfg_sample_rate_overrides_params(tmp_path: Path) -> None:
    """Explicit sample_rate in cfg must take precedence over params.yaml value."""
    import pet_eval.plugins.audio_evaluator as mod

    fake_params = tmp_path / "params.yaml"
    fake_params.write_text(yaml.dump({"audio": {"sample_rate": 8000}}))

    original_path = mod._PARAMS_PATH
    try:
        mod._PARAMS_PATH = fake_params
        evaluator = AudioEvaluator(metrics=[], thresholds={}, sample_rate=44100)
        assert evaluator._sample_rate == 44100
    finally:
        mod._PARAMS_PATH = original_path
