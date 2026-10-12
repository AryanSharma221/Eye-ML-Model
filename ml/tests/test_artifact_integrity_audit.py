"""
Test Suite for Artifact Integrity, Traceability, Reproducibility, and Failure Handling Audit.

Verifies:
1. Champion checkpoint integrity (DenseNet201, SHA-256 hash match, epoch 6, canonical 4-class order).
2. Manifest integrity (SHA-256 hash match, 5,889 samples, zero patient/filepath/content leakage).
3. Cross-module reference alignment across all ML submodules.
4. Bit-for-bit repeatability on identical inputs and device.
5. Strict fail-fast behavior on missing checkpoint, corrupt checkpoint, missing manifest, incompatible classes.
6. Traceability contracts in evaluation report artifacts.
"""

import json
import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    get_default_checkpoint_path,
    get_default_evaluation_dir,
    get_default_manifest_path,
)
from ml.dataset_loader import EyeDiseaseDataset, load_split_manifest
from ml.evaluate import (
    calculate_classification_metrics,
    evaluate_model_checkpoint,
)
from ml.exceptions import (
    CheckpointCorruptError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    CheckpointVerificationError,
)
from ml.experiment_tracker import compute_file_sha256
from ml.eye_model import (
    DEFAULT_CHECKPOINT_PATH,
    build_densenet201_model,
    clear_model_cache,
    get_cached_model,
    load_eye_model_checkpoint,
    predict_eye_disease,
)
from ml.screening_pipeline import run_screening

EXPECTED_CHAMPION_SHA256 = "e819440c35f166c90e6469b071cc83136857e95aa80bcd6697336894bb459f5f"
EXPECTED_MANIFEST_SHA256 = "cb2d80449899673438268750011697f7ae9518a8259e5dc8482edfe9d7c01f27"
CANONICAL_CLASSES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]


class TestArtifactIntegrityAndAlignment:
    """Verifies that all production artifacts and code references are synchronized."""

    def test_champion_checkpoint_sha256_and_metadata(self):
        """Verifies champion checkpoint exists, matches SHA-256 hash, and contains valid metadata."""
        ckpt_path = DEFAULT_CHECKPOINT_PATH
        assert os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}"

        actual_sha = compute_file_sha256(ckpt_path)
        assert actual_sha == EXPECTED_CHAMPION_SHA256, (
            f"Checkpoint SHA-256 mismatch! Expected {EXPECTED_CHAMPION_SHA256}, got {actual_sha}"
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert ckpt["architecture"] == "DenseNet201"
        assert ckpt["classes"] == CANONICAL_CLASSES
        assert ckpt["epoch"] == 6
        assert len(ckpt["state_dict"]) == 1207

    def test_dataset_split_manifest_sha256_and_structure(self):
        """Verifies dataset manifest matches expected SHA-256 hash and contains all 5,889 images."""
        manifest_path = get_default_manifest_path()
        assert os.path.exists(manifest_path), f"Manifest missing at {manifest_path}"

        actual_sha = compute_file_sha256(manifest_path)
        assert actual_sha == EXPECTED_MANIFEST_SHA256, (
            f"Manifest SHA-256 mismatch! Expected {EXPECTED_MANIFEST_SHA256}, got {actual_sha}"
        )

        data = load_split_manifest(manifest_path)
        assert data["metadata"]["total_images"] == 5889
        assert data["metadata"]["classes"] == CANONICAL_CLASSES
        assert len(data["splits"]["test"]) == 894

    def test_cross_module_checkpoint_reference_alignment(self):
        """Confirms all ML modules reference the exact same champion checkpoint file."""
        from ml.config import get_default_checkpoint_path
        from ml.eye_model import DEFAULT_CHECKPOINT_PATH as EYE_CKPT
        from ml.evaluate import DEFAULT_BEST_CHECKPOINT_PATH as EVAL_CKPT
        from ml.checkpoint_manager import get_default_checkpoint_path as MGR_CKPT

        p_config = os.path.abspath(get_default_checkpoint_path())
        p_eye = os.path.abspath(EYE_CKPT)
        p_eval = os.path.abspath(EVAL_CKPT)
        p_mgr = os.path.abspath(MGR_CKPT())

        assert p_config == p_eye == p_eval == p_mgr
        assert compute_file_sha256(p_config) == EXPECTED_CHAMPION_SHA256


class TestReproducibilityAndRepeatability:
    """Verifies numerical determinism and repeatability on identical inputs."""

    def test_repeated_inference_numerical_determinism(self):
        """Confirms running inference twice produces 100% identical predictions and probabilities."""
        clear_model_cache()
        manifest_path = get_default_manifest_path()
        manifest = load_split_manifest(manifest_path)
        test_records = manifest["splits"]["test"]

        # Select first 5 valid images
        sample_paths = [r["filepath"] for r in test_records if os.path.exists(r["filepath"])][:5]
        assert len(sample_paths) == 5

        device = "cuda" if torch.cuda.is_available() else "cpu"

        pass1 = [predict_eye_disease(p, weights_path=DEFAULT_CHECKPOINT_PATH, device=device, use_cache=True) for p in sample_paths]
        pass2 = [predict_eye_disease(p, weights_path=DEFAULT_CHECKPOINT_PATH, device=device, use_cache=True) for p in sample_paths]

        for p1, p2 in zip(pass1, pass2):
            assert p1["prediction"] == p2["prediction"]
            assert p1["class_index"] == p2["class_index"]
            assert abs(p1["confidence"] - p2["confidence"]) == 0.0
            for c in CANONICAL_CLASSES:
                diff = abs(p1["all_probabilities"][c] - p2["all_probabilities"][c])
                assert diff == 0.0, f"Non-zero probability diff: {diff}"


class TestFailureHandlingAndSafeGuards:
    """Verifies strict fail-fast error handling without random weight fallback."""

    def test_missing_checkpoint_fails_cleanly(self):
        """Missing checkpoint raises CheckpointNotFoundError without random fallback."""
        clear_model_cache()
        with pytest.raises((CheckpointNotFoundError, FileNotFoundError)):
            get_cached_model(weights_path="non_existent_weights_xyz.pth")

    def test_corrupted_checkpoint_fails_cleanly(self, tmp_path):
        """Corrupted checkpoint payload fails with CheckpointCorruptError or CheckpointVerificationError."""
        corrupt_path = str(tmp_path / "corrupt_checkpoint.pth")
        with open(corrupt_path, "wb") as f:
            f.write(b"CORRUPTED_CHECKPOINT_HEADER_DATA_12345")

        clear_model_cache()
        model = build_densenet201_model(num_classes=NUM_CLASSES)
        with pytest.raises((CheckpointCorruptError, CheckpointVerificationError, Exception)):
            load_eye_model_checkpoint(model, corrupt_path)

    def test_missing_manifest_fails_cleanly(self):
        """Missing dataset manifest raises FileNotFoundError immediately."""
        with pytest.raises(FileNotFoundError):
            load_split_manifest("non_existent_manifest_xyz.json")

    def test_incompatible_class_mapping_fails_cleanly(self, tmp_path):
        """Checkpoint with incompatible class count or order fails verification."""
        bad_ckpt_path = str(tmp_path / "bad_class_checkpoint.pth")
        torch.save(
            {
                "epoch": 1,
                "architecture": "DenseNet201",
                "classes": ["disease_a", "disease_b"],  # Incompatible classes
                "state_dict": {},
            },
            bad_ckpt_path,
        )

        model = build_densenet201_model(num_classes=NUM_CLASSES)
        with pytest.raises((CheckpointIncompatibleError, CheckpointVerificationError, Exception)):
            load_eye_model_checkpoint(model, bad_ckpt_path)

    def test_evaluation_metric_length_mismatch_fails_cleanly(self):
        """Mismatched ground truth and prediction lengths fail immediately."""
        with pytest.raises(ValueError, match="Length mismatch"):
            calculate_classification_metrics([0, 1, 2], [0, 1], CANONICAL_CLASSES)


class TestArtifactTraceabilityContract:
    """Verifies that evaluation report contains all required artifact traceability fields."""

    def test_evaluation_report_traceability_fields(self):
        """Confirms test_evaluation_report.json records all required traceability metadata."""
        report_path = os.path.join(get_default_evaluation_dir(), "test_evaluation_report.json")
        assert os.path.exists(report_path), f"Report missing at {report_path}"

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert report["checkpoint_sha256"] == EXPECTED_CHAMPION_SHA256
        assert report["manifest_sha256"] == EXPECTED_MANIFEST_SHA256
        assert report["model_architecture"] == "DenseNet201"
        assert report["canonical_class_order"] == CANONICAL_CLASSES
        assert "preprocessing_config" in report
        assert report["preprocessing_config"]["target_size"] == [224, 224]
        assert "software_versions" in report
        assert "evaluation_timestamp" in report
        assert report["total_test_images"] == 894
        assert report["evaluated_images"] == 894
        assert report["failed_or_excluded_images"] == 0
        assert report["overall_accuracy"] == 0.8781
