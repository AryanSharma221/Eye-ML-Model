"""
Tests for Experiment Tracking, Safe Checkpoint Versioning, and Provenance (ml/experiment_tracker.py).

Verifies:
1. Cryptographic hashing (SHA-256) and checkpoint integrity verification.
2. Rejection of corrupted, incomplete, shape-mismatched, or non-finite checkpoints.
3. Isolated run directory structure and metadata JSON schema conformance.
4. Safe checkpoint versioning: baseline production checkpoints are never overwritten without explicit flag.
5. Resumable training: state restoration, epoch counting, and rejection of corrupt recovery checkpoints.
6. Traceable evaluation linkage between test set evaluations and experiment runs.
7. Experiment registry discovery, legacy checkpoint auditing, and summary CLI integration.
"""

import hashlib
import json
import os
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.config import get_default_experiments_dir
from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES, save_split_manifest
from ml.evaluate import evaluate_model_checkpoint
from ml.experiment_tracker import (
    ExperimentRun,
    ExperimentTracker,
    compute_file_sha256,
    verify_checkpoint_integrity,
)
from ml.eye_model import (
    DEFAULT_CHECKPOINT_PATH,
    build_densenet201_model,
    load_eye_model_checkpoint,
)
from ml.train_eye import (
    DEFAULT_BEST_CHECKPOINT_PATH,
    TrainConfig,
    build_training_model,
    save_atomic_checkpoint,
    train_eye_model,
)


@pytest.fixture
def temp_experiments_dir(tmp_path) -> str:
    """Provides an isolated directory for experiment runs during testing."""
    exp_dir = str(tmp_path / "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


@pytest.fixture
def synthetic_manifest_and_images(tmp_path) -> str:
    """Creates a tiny synthetic dataset manifest for testing."""
    import cv2

    dataset_dir = str(tmp_path / "test_data")
    os.makedirs(dataset_dir, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}
    for split in ["train", "val", "test"]:
        for i, cls_name in enumerate(CLASS_NAMES):
            img_path = os.path.join(dataset_dir, f"{split}_{cls_name}_{i}.png")
            arr = np.full((224, 224, 3), 120 + i * 20, dtype=np.uint8)
            cv2.circle(arr, (112, 112), 45, (180, 140, 100), -1)
            cv2.imwrite(img_path, arr)

            splits[split].append(
                {
                    "filepath": img_path,
                    "filename": os.path.basename(img_path),
                    "class_name": cls_name,
                    "class_index": i,
                    "patient_id": f"p_{split}_{i}",
                    "split": split,
                }
            )

    manifest_path = str(tmp_path / "split_manifest.json")
    save_split_manifest(splits, manifest_path)
    return manifest_path


@pytest.fixture
def mock_densenet_checkpoint(tmp_path) -> str:
    """Creates a valid lightweight DenseNet201 checkpoint file for fast testing."""
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    payload = {
        "state_dict": model.state_dict(),
        "classes": list(CLASS_NAMES),
        "epoch": 5,
        "best_acc": 0.9029,
        "architecture": "DenseNet201",
    }
    ckpt_path = str(tmp_path / "valid_test_checkpoint.pth")
    torch.save(payload, ckpt_path)
    return ckpt_path


class TestFileHashingAndIntegrity:
    """Verifies SHA-256 calculation and checkpoint integrity verification."""

    def test_compute_file_sha256_success(self, tmp_path):
        sample_file = tmp_path / "sample.txt"
        sample_file.write_text("multimodal-health-screening-sha256-test", encoding="utf-8")

        expected_hash = hashlib.sha256("multimodal-health-screening-sha256-test".encode("utf-8")).hexdigest()
        computed_hash = compute_file_sha256(str(sample_file))

        assert computed_hash == expected_hash
        assert len(computed_hash) == 64

    def test_compute_file_sha256_missing_file(self, tmp_path):
        non_existent = str(tmp_path / "non_existent_file.pth")
        with pytest.raises(FileNotFoundError):
            compute_file_sha256(non_existent)

    def test_verify_checkpoint_integrity_valid(self, mock_densenet_checkpoint):
        result = verify_checkpoint_integrity(mock_densenet_checkpoint, num_classes=NUM_CLASSES)
        assert result["valid"] is True
        assert len(result["sha256"]) == 64
        assert result["file_size_bytes"] > 0
        assert result["epoch"] == 5
        assert result["best_val_accuracy"] == 0.9029
        assert result["classes"] == list(CLASS_NAMES)

    def test_verify_checkpoint_integrity_corrupted_file(self, tmp_path):
        corrupt_path = str(tmp_path / "corrupt.pth")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_PYTORCH_ZIP_ARCHIVE_HEADER_CORRUPTED_BYTES")

        result = verify_checkpoint_integrity(corrupt_path, num_classes=NUM_CLASSES)
        assert result["valid"] is False
        assert "error" in result
        assert len(result["sha256"]) == 64

    def test_verify_checkpoint_integrity_missing_state_dict(self, tmp_path):
        invalid_path = str(tmp_path / "no_weights.pth")
        torch.save({"classes": list(CLASS_NAMES), "epoch": 1}, invalid_path)

        result = verify_checkpoint_integrity(invalid_path, num_classes=NUM_CLASSES)
        assert result["valid"] is False
        assert "state_dict" in result["error"].lower()

    def test_verify_checkpoint_integrity_wrong_class_shape(self, tmp_path):
        # Create a checkpoint with only 2 output classes instead of 4
        wrong_model = build_densenet201_model(num_classes=2)
        wrong_path = str(tmp_path / "wrong_classes.pth")
        torch.save(
            {"state_dict": wrong_model.state_dict(), "classes": ["cataract", "normal"]},
            wrong_path,
        )

        result = verify_checkpoint_integrity(wrong_path, num_classes=NUM_CLASSES)
        assert result["valid"] is False
        assert "shape" in result["error"] or "mismatch" in result["error"]

    def test_verify_checkpoint_integrity_nan_logits(self, tmp_path):
        model = build_densenet201_model(num_classes=NUM_CLASSES)
        state_dict = model.state_dict()
        # Inject NaN into classifier weights
        state_dict["classifier.weight"][0, 0] = float("nan")
        nan_path = str(tmp_path / "nan_weights.pth")
        torch.save({"state_dict": state_dict, "classes": list(CLASS_NAMES)}, nan_path)

        result = verify_checkpoint_integrity(nan_path, num_classes=NUM_CLASSES)
        assert result["valid"] is False
        assert "non-finite logits" in result["error"]


class TestExperimentRunLifecycle:
    """Verifies ExperimentRun creation, directory isolation, metadata schema, and lifecycle events."""

    def test_isolated_directory_structure(self, temp_experiments_dir):
        tracker = ExperimentTracker(experiments_dir=temp_experiments_dir)
        run = tracker.create_run(experiment_name="test_experiment", tags=["unit_test"])

        assert os.path.exists(run.run_dir)
        assert os.path.exists(run.checkpoints_dir)
        assert os.path.exists(run.evaluations_dir)
        assert os.path.exists(run.logs_dir)
        assert run.run_id in os.listdir(os.path.join(temp_experiments_dir, "runs"))

    def test_metadata_schema_conformance(self, temp_experiments_dir, synthetic_manifest_and_images):
        tracker = ExperimentTracker(experiments_dir=temp_experiments_dir)
        run = tracker.create_run(
            experiment_name="densenet201_baseline",
            description="Testing schema conformance",
            tags=["schema_test", "densenet201"],
        )

        dummy_cfg = {
            "batch_size": 4,
            "num_epochs": 2,
            "learning_rate": 0.0005,
            "weight_decay": 1e-4,
            "patience": 3,
            "use_class_weights": True,
            "use_local_pretrained": False,
        }

        run.start(
            config_dict=dummy_cfg,
            manifest_path=synthetic_manifest_and_images,
            device="cpu",
            seed=123,
        )

        assert os.path.exists(run.metadata_path)
        with open(run.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Validate mandatory schema top-level fields
        required_keys = [
            "schema_version",
            "run_id",
            "experiment_name",
            "description",
            "tags",
            "status",
            "timestamps",
            "reproducibility",
            "device",
            "dataset",
            "model",
            "hyperparameters",
            "preprocessing",
            "environment",
            "epoch_history",
            "best_metrics",
            "checkpoints",
            "evaluations",
        ]
        for key in required_keys:
            assert key in metadata, f"Missing key '{key}' in run metadata"

        assert metadata["status"] == "RUNNING"
        assert metadata["reproducibility"]["seed"] == 123
        assert metadata["model"]["num_classes"] == 4
        assert metadata["model"]["class_names"] == list(CLASS_NAMES)
        assert metadata["dataset"]["total_images"] == 12
        assert len(metadata["dataset"]["split_manifest_sha256"]) == 64

    def test_log_epoch_records(self, temp_experiments_dir, synthetic_manifest_and_images):
        tracker = ExperimentTracker(experiments_dir=temp_experiments_dir)
        run = tracker.create_run(experiment_name="epoch_test")
        run.start(
            config_dict={},
            manifest_path=synthetic_manifest_and_images,
            device="cpu",
            seed=42,
        )

        run.log_epoch(
            epoch=1,
            train_loss=0.8521,
            train_acc=0.6500,
            val_loss=0.7412,
            val_acc=0.7500,
            lr=0.001,
            duration_seconds=12.34,
        )
        run.log_epoch(
            epoch=2,
            train_loss=0.6214,
            train_acc=0.8000,
            val_loss=0.5123,
            val_acc=0.8500,
            lr=0.001,
            duration_seconds=11.98,
        )

        with open(run.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert len(meta["epoch_history"]) == 2
        assert meta["epoch_history"][0]["epoch"] == 1
        assert meta["epoch_history"][0]["train_loss"] == 0.8521
        assert meta["epoch_history"][1]["val_acc"] == 0.85

    def test_save_run_checkpoint_atomic_and_hashed(
        self, temp_experiments_dir, synthetic_manifest_and_images
    ):
        tracker = ExperimentTracker(experiments_dir=temp_experiments_dir)
        run = tracker.create_run(experiment_name="checkpoint_save_test")
        run.start(
            config_dict={},
            manifest_path=synthetic_manifest_and_images,
            device="cpu",
            seed=42,
        )

        model = build_densenet201_model(num_classes=NUM_CLASSES)
        saved_best_path = run.save_run_checkpoint(
            model=model,
            epoch=1,
            is_best=True,
            best_val_acc=0.875,
            val_loss=0.45,
        )

        assert os.path.exists(saved_best_path)
        assert os.path.dirname(saved_best_path) == run.checkpoints_dir

        with open(run.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        best_meta = meta["checkpoints"]["best_checkpoint"]
        assert best_meta is not None
        assert best_meta["verified"] is True
        assert len(best_meta["sha256"]) == 64
        assert best_meta["epoch"] == 1
        assert best_meta["best_val_accuracy"] == 0.875

    def test_run_lifecycle_status_transitions(
        self, temp_experiments_dir, synthetic_manifest_and_images
    ):
        tracker = ExperimentTracker(experiments_dir=temp_experiments_dir)

        # Test completion
        run1 = tracker.create_run(experiment_name="complete_test")
        run1.start(config_dict={}, manifest_path=synthetic_manifest_and_images, device="cpu")
        run1.complete()
        assert run1.metadata["status"] == "COMPLETED"
        assert run1.metadata["timestamps"]["completion_utc"] is not None
        assert run1.metadata["timestamps"]["total_duration_seconds"] is not None

        # Test interruption
        run2 = tracker.create_run(experiment_name="interrupt_test")
        run2.start(config_dict={}, manifest_path=synthetic_manifest_and_images, device="cpu")
        run2.interrupt(reason="KeyboardInterrupt simulated")
        assert run2.metadata["status"] == "INTERRUPTED"
        assert "KeyboardInterrupt" in run2.metadata["failure_reason"]

        # Test failure
        run3 = tracker.create_run(experiment_name="fail_test")
        run3.start(config_dict={}, manifest_path=synthetic_manifest_and_images, device="cpu")
        run3.fail(error_message="Simulated CUDA Out Of Memory")
        assert run3.metadata["status"] == "FAILED"
        assert "Out Of Memory" in run3.metadata["failure_reason"]


class TestSafeCheckpointVersioning:
    """Verifies that production baseline checkpoints are never overwritten without explicit flag."""

    def test_training_does_not_overwrite_production_weights_by_default(
        self, temp_experiments_dir, synthetic_manifest_and_images
    ):
        # Capture baseline production checkpoint hash before training
        assert os.path.exists(DEFAULT_CHECKPOINT_PATH), "Production eye_model.pth missing"
        original_prod_hash = compute_file_sha256(DEFAULT_CHECKPOINT_PATH)
        assert os.path.exists(DEFAULT_BEST_CHECKPOINT_PATH), "eye_densenet201_best.pth missing"
        original_best_hash = compute_file_sha256(DEFAULT_BEST_CHECKPOINT_PATH)

        # Run 1 epoch dry-run training with default promote_to_production=False
        cfg = TrainConfig(
            manifest_path=synthetic_manifest_and_images,
            experiments_dir=temp_experiments_dir,
            batch_size=2,
            num_epochs=1,
            num_workers=0,
            use_local_pretrained=False,
            device=torch.device("cpu"),
            eval_test_after_train=False,
            promote_to_production=False,  # Explicitly default
        )

        result = train_eye_model(cfg)

        # 1. Verify production baseline checkpoint hash is 100% UNCHANGED
        post_prod_hash = compute_file_sha256(DEFAULT_CHECKPOINT_PATH)
        post_best_hash = compute_file_sha256(DEFAULT_BEST_CHECKPOINT_PATH)
        assert post_prod_hash == original_prod_hash, "CRITICAL: eye_model.pth was mutated!"
        assert post_best_hash == original_best_hash, "CRITICAL: eye_densenet201_best.pth was mutated!"

        # 2. Verify run-isolated checkpoint was created instead
        run_id = result.get("run_id")
        assert run_id is not None
        run_ckpt_path = os.path.join(temp_experiments_dir, "runs", run_id, "checkpoints", "best_model.pth")
        assert os.path.exists(run_ckpt_path)


class TestResumptionAndRecovery:
    """Verifies resilient training resumption from valid checkpoints and rejection of corrupted checkpoints."""

    def test_resumption_from_verified_recovery_checkpoint(
        self, temp_experiments_dir, synthetic_manifest_and_images, tmp_path
    ):
        # Create a mock valid recovery checkpoint
        model = build_training_model(num_classes=NUM_CLASSES, use_local_pretrained=False, device=torch.device("cpu"))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        recovery_path = str(tmp_path / "valid_recovery.pth")
        payload = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "classes": list(CLASS_NAMES),
            "epoch": 2,
            "best_acc": 0.65,
            "best_val_accuracy": 0.65,
            "history": {
                "train_loss": [1.2, 0.9],
                "train_acc": [0.4, 0.6],
                "val_loss": [1.1, 0.8],
                "val_acc": [0.5, 0.65],
                "lr": [0.001, 0.001],
            },
            "seed": 42,
            "architecture": "DenseNet201",
        }
        save_atomic_checkpoint(payload, recovery_path)

        # Run 1 epoch resumed training (epochs 2 -> 3)
        cfg = TrainConfig(
            manifest_path=synthetic_manifest_and_images,
            experiments_dir=temp_experiments_dir,
            recovery_checkpoint_path=recovery_path,
            batch_size=2,
            num_epochs=3,
            resume=True,
            resume_from=recovery_path,
            use_local_pretrained=False,
            device=torch.device("cpu"),
            eval_test_after_train=False,
        )

        res = train_eye_model(cfg)

        # Verify history resumed from epoch 2 and added epoch 3
        assert len(res["history"]["train_loss"]) == 3
        assert len(res["history"]["val_acc"]) == 3

        # Verify metadata recorded resumption linkage
        tracker = ExperimentTracker(experiments_dir=temp_experiments_dir)
        run = tracker.get_run(res["run_id"])
        assert run is not None
        assert "resumed_from" in run.metadata
        assert run.metadata["resumed_from"]["resumed_epoch"] == 2
        assert len(run.metadata["resumed_from"]["sha256"]) == 64

    def test_resumption_rejects_corrupted_checkpoint(
        self, temp_experiments_dir, synthetic_manifest_and_images, tmp_path
    ):
        corrupt_path = str(tmp_path / "broken_recovery.pth")
        with open(corrupt_path, "wb") as f:
            f.write(b"CORRUPTED_BYTES_NOT_A_VALID_CHECKPOINT")

        cfg = TrainConfig(
            manifest_path=synthetic_manifest_and_images,
            experiments_dir=temp_experiments_dir,
            recovery_checkpoint_path=corrupt_path,
            resume=True,
            resume_from=corrupt_path,
            device=torch.device("cpu"),
        )

        with pytest.raises(ValueError) as exc_info:
            train_eye_model(cfg)

        assert "integrity check failed" in str(exc_info.value)


class TestEvaluationLinkage:
    """Verifies that test evaluation results are linked to experiment runs and checkpoints."""

    def test_link_evaluation_to_experiment_run(
        self, temp_experiments_dir, synthetic_manifest_and_images
    ):
        tracker = ExperimentTracker(experiments_dir=temp_experiments_dir)
        run = tracker.create_run(experiment_name="eval_link_test")
        run.start(config_dict={}, manifest_path=synthetic_manifest_and_images, device="cpu")

        sample_eval_report = {
            "evaluation_timestamp": "2026-10-12T02:00:00Z",
            "dataset_split": "test",
            "evaluated_images": 894,
            "checkpoint_sha256": "e819440c35f166c90e6469b071cc83136857e95aa80bcd6697336894bb459f5f",
            "metrics": {
                "overall_accuracy": 0.9029,
                "macro_f1": 0.9021,
                "weighted_f1": 0.9030,
            },
            "saved_report_path": "ml/experiments/evaluations/report.json",
        }

        run.link_evaluation(sample_eval_report)

        with open(run.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert len(meta["evaluations"]) == 1
        linked = meta["evaluations"][0]
        assert linked["evaluated_images"] == 894
        assert linked["overall_accuracy"] == 0.9029
        assert linked["checkpoint_sha256"] == "e819440c35f166c90e6469b071cc83136857e95aa80bcd6697336894bb459f5f"

    def test_evaluate_model_checkpoint_with_run_id_integration(
        self, temp_experiments_dir, synthetic_manifest_and_images, tmp_path
    ):
        # Create a valid checkpoint
        model = build_densenet201_model(num_classes=NUM_CLASSES)
        ckpt_path = str(tmp_path / "eval_model.pth")
        torch.save({"state_dict": model.state_dict(), "classes": list(CLASS_NAMES)}, ckpt_path)

        # Create experiment run
        tracker = ExperimentTracker(experiments_dir=temp_experiments_dir)
        run = tracker.create_run(experiment_name="eval_integration_test")
        run.start(config_dict={}, manifest_path=synthetic_manifest_and_images, device="cpu")

        # Run evaluate_model_checkpoint linking run_id
        eval_out_dir = str(tmp_path / "eval_out")
        report = evaluate_model_checkpoint(
            checkpoint_path=ckpt_path,
            manifest_path=synthetic_manifest_and_images,
            split="test",
            batch_size=2,
            output_dir=eval_out_dir,
            run_id=run.run_id,
            experiments_dir=temp_experiments_dir,
        )

        assert report["overall_accuracy"] >= 0.0
        assert "checkpoint_sha256" in report
        assert len(report["checkpoint_sha256"]) == 64

        # Verify run metadata received evaluation link
        refreshed_run = tracker.get_run(run.run_id)
        assert len(refreshed_run.metadata["evaluations"]) == 1
        assert refreshed_run.metadata["evaluations"][0]["dataset_split"] == "test"


class TestExperimentTrackerRegistryAndCLI:
    """Verifies registry queries, baseline audits, and summary CLI."""

    def test_tracker_list_and_get_runs(self, temp_experiments_dir, synthetic_manifest_and_images):
        tracker = ExperimentTracker(experiments_dir=temp_experiments_dir)

        run_a = tracker.create_run(experiment_name="exp_a", run_id="run_alpha")
        run_a.start(config_dict={}, manifest_path=synthetic_manifest_and_images, device="cpu")
        run_a.complete()

        run_b = tracker.create_run(experiment_name="exp_b", run_id="run_beta")
        run_b.start(config_dict={}, manifest_path=synthetic_manifest_and_images, device="cpu")
        run_b.fail("Intentional fail")

        runs = tracker.list_runs()
        assert len(runs) == 2
        run_ids = [r["run_id"] for r in runs]
        assert "run_alpha" in run_ids
        assert "run_beta" in run_ids

        fetched_a = tracker.get_run("run_alpha")
        assert fetched_a is not None
        assert fetched_a.metadata["status"] == "COMPLETED"

    def test_audit_legacy_checkpoints(self):
        tracker = ExperimentTracker()
        audits = tracker.audit_legacy_checkpoints()

        # Must find baseline checkpoints in ml/weights/
        assert len(audits) >= 4
        filenames = [a["filename"] for a in audits]
        assert "eye_model.pth" in filenames
        assert "eye_densenet201_best.pth" in filenames

        for item in audits:
            assert item["integrity"]["valid"] is True, f"Legacy checkpoint failed integrity: {item['filename']}"
            assert len(item["sha256"]) == 64
            assert item["file_size_bytes"] > 0

    def test_experiment_summary_cli_json_execution(self):
        res = subprocess.run(
            [sys.executable, "ml/experiment_summary.py", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0, f"experiment_summary.py failed: {res.stderr}"

        parsed = json.loads(res.stdout)
        assert "experiments_root" in parsed
        assert "total_runs" in parsed
        assert "runs" in parsed
        assert "legacy_checkpoints_audit" in parsed
        assert len(parsed["legacy_checkpoints_audit"]) >= 4

    def test_experiment_summary_cli_legacy_audit_execution(self):
        res = subprocess.run(
            [sys.executable, "ml/experiment_summary.py", "--legacy"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0, f"experiment_summary.py --legacy failed: {res.stderr}"
        assert "eye_model.pth" in res.stdout
        assert "SHA-256" in res.stdout
        assert "VALID" in res.stdout
