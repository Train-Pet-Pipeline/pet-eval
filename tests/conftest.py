"""Shared pytest fixtures for pet-eval test suite."""

from __future__ import annotations

import pathlib

import pytest


@pytest.fixture()
def tmp_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return a temporary directory for test output files."""
    return tmp_path


@pytest.fixture()
def sample_params() -> dict:
    """Return a full params.yaml configuration dict matching params.yaml."""
    return {
        "gates": {
            "vlm": {
                "schema_compliance": 0.99,
                "distribution_sum_error": 0.01,
                "anomaly_recall": 0.85,
                "anomaly_false_positive": 0.15,
                "mood_spearman": 0.75,
                "narrative_bertscore": 0.80,
                "latency_p95_ms": 4000,
                "kl_divergence": 0.02,
            },
            "audio": {
                "overall_accuracy": 0.80,
                "vomit_recall": 0.70,
            },
        },
        "benchmark": {
            "gold_set_path": "benchmark/gold_set_v1.jsonl",
            "anomaly_set_path": "benchmark/anomaly_set_v1.jsonl",
            "audio_test_dir": "",
        },
        "inference": {
            "schema_version": "1.0",
            "max_new_tokens": 1024,
            "batch_size": 1,
        },
        "audio": {
            "classes": ["eating", "drinking", "vomiting", "ambient", "other"],
        },
        "device": {
            "adb_timeout": 30,
            "warmup_runs": 3,
            "latency_runs": 50,
        },
    }


@pytest.fixture()
def sample_vlm_output_valid() -> dict:
    """Return a valid PetFeederEvent JSON dict (schema v1.0 compliant).

    Action distribution sums to exactly 1.0.
    All required fields present and within valid ranges.
    """
    return {
        "schema_version": "1.0",
        "pet_present": True,
        "pet_count": 1,
        "pet": {
            "species": "cat",
            "breed_estimate": "domestic shorthair",
            "id_tag": "cat_001",
            "id_confidence": 0.92,
            "action": {
                "primary": "eating",
                "distribution": {
                    "eating": 0.75,
                    "drinking": 0.05,
                    "sniffing_only": 0.10,
                    "leaving_bowl": 0.05,
                    "sitting_idle": 0.03,
                    "other": 0.02,
                },
            },
            "eating_metrics": {
                "speed": {
                    "fast": 0.20,
                    "normal": 0.60,
                    "slow": 0.20,
                },
                "engagement": 0.85,
                "abandoned_midway": 0.05,
            },
            "mood": {
                "alertness": 0.70,
                "anxiety": 0.10,
                "engagement": 0.85,
            },
            "body_signals": {
                "posture": "relaxed",
                "ear_position": "forward",
            },
            "anomaly_signals": {
                "vomit_gesture": 0.02,
                "food_rejection": 0.05,
                "excessive_sniffing": 0.10,
                "lethargy": 0.03,
                "aggression": 0.01,
            },
        },
        "bowl": {
            "food_fill_ratio": 0.65,
            "water_fill_ratio": None,
            "food_type_visible": "dry",
        },
        "scene": {
            "lighting": "bright",
            "image_quality": "clear",
            "confidence_overall": 0.94,
        },
        "narrative": "Cat eating dry food at normal pace, relaxed posture.",
    }


@pytest.fixture()
def sample_vlm_output_invalid() -> dict:
    """Return an invalid PetFeederEvent JSON dict with bad distribution sum.

    The action distribution sums to 1.25 (other=0.50 instead of valid value),
    which violates the schema constraint of sum == 1.0 +/- 0.01.
    """
    return {
        "schema_version": "1.0",
        "pet_present": True,
        "pet_count": 1,
        "pet": {
            "species": "cat",
            "breed_estimate": "domestic shorthair",
            "id_tag": "cat_001",
            "id_confidence": 0.90,
            "action": {
                "primary": "eating",
                "distribution": {
                    "eating": 0.75,
                    "drinking": 0.05,
                    "sniffing_only": 0.10,
                    "leaving_bowl": 0.05,
                    "sitting_idle": 0.30,
                    "other": 0.50,  # sum = 1.75, invalid
                },
            },
            "eating_metrics": {
                "speed": {
                    "fast": 0.20,
                    "normal": 0.60,
                    "slow": 0.20,
                },
                "engagement": 0.85,
                "abandoned_midway": 0.05,
            },
            "mood": {
                "alertness": 0.70,
                "anxiety": 0.10,
                "engagement": 0.85,
            },
            "body_signals": {
                "posture": "relaxed",
                "ear_position": "forward",
            },
            "anomaly_signals": {
                "vomit_gesture": 0.02,
                "food_rejection": 0.05,
                "excessive_sniffing": 0.10,
                "lethargy": 0.03,
                "aggression": 0.01,
            },
        },
        "bowl": {
            "food_fill_ratio": 0.65,
            "water_fill_ratio": None,
            "food_type_visible": "dry",
        },
        "scene": {
            "lighting": "bright",
            "image_quality": "clear",
            "confidence_overall": 0.94,
        },
        "narrative": "Cat eating dry food at normal pace.",
    }
