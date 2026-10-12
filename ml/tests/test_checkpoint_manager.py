"""
Tests for Checkpoint Selection, Eligibility Validation, Candidate Comparison, and Safe Rollback.

Verifies:
1. Eligibility verification (production loader compatibility, tensor dimensions, class order,
   inference smoke tests, preprocessing match, and documented provenance).
2. Rejection of missing, corrupted, shape-mismatched, class-mismatched, and non-finite checkpoints.
3. Candidate comparison with equivalence checks and clinical regression alerts.
4. Safe promotion: preserves underlying binary files, updates active pointer, archives history,
   and verifies post-promotion production loading.
5. Reversible rollback: restores previous verified checkpoint, verifies integrity, rejects missing
   or modified historical files, and reports when no historical state is available.
6. Interruption recovery: cleans orphaned .tmp files and auto-repairs corrupted selection files.
7. CLI entry point execution for status, list, check, and compare commands.
"""

import json
import os
import subprocess
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.checkpoint_manager import (
    audit_all_checkpoints,
    check_checkpoint_eligibility,
    compare_checkpoints,
    get_active_checkpoint_status,
    load_selection_file,
    promote_checkpoint,
    recover_selection_state,
    rollback_checkpoint,
    save_atomic_selection_file,
)
from ml.config import (
    CLASS_NAMES,
    ML_DIR,
    NUM_CLASSES,
    get_default_checkpoint_path,
    get_ml_config,
)
from ml.eye_model import clear_model_cache
from ml.exceptions import (
    CheckpointCorruptError,
    CheckpointNotFoundError,
    CheckpointVerificationError,
    IneligibleCheckpointError,
    RollbackUnavailableError,
)
from ml.experiment_tracker import compute_file_sha256
from ml.eye_model import build_densenet201_model


@pytest.fixture
def temp_selection_file(tmp_path) -> str:
    """Provides a temporary, isolated checkpoint selection file."""
    sel_path = str(tmp_path / "checkpoint_selection.json")
    return sel_path


@pytest.fixture
def valid_mock_checkpoint_a(tmp_path) -> str:
    """Creates a valid lightweight DenseNet201 checkpoint representing Model A."""
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    payload = {
        "state_dict": model.state_dict(),
        "classes": list(CLASS_NAMES),
        "epoch": 5,
        "best_acc": 0.8950,
        "architecture": "DenseNet201",
        "seed": 42,
    }
    ckpt_path = str(tmp_path / "model_a.pth")
    torch.save(payload, ckpt_path)
    return ckpt_path


@pytest.fixture
def valid_mock_checkpoint_b(tmp_path) -> str:
    """Creates a valid lightweight DenseNet201 checkpoint representing Model B."""
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    payload = {
        "state_dict": model.state_dict(),
        "classes": list(CLASS_NAMES),
        "epoch": 6,
        "best_acc": 0.9029,
        "architecture": "DenseNet201",
        "seed": 42,
    }
    ckpt_path = str(tmp_path / "model_b.pth")
    torch.save(payload, ckpt_path)
    return ckpt_path


class TestCheckpointEligibility:
    """Verifies eligibility checks against valid, corrupt, and incompatible checkpoints."""

    def test_real_baseline_checkpoints_eligible(self):
        """Verifies that the repository's real eye_model.pth is 100% eligible."""
        baseline_path = os.path.join(ML_DIR, "weights", "eye_model.pth")
        assert os.path.exists(baseline_path), "Baseline eye_model.pth missing"

        res = check_checkpoint_eligibility(baseline_path, run_smoke_test=True)
        assert res["eligible"] is True
        assert res["sha256"] == "e819440c35f166c90e6469b071cc83136857e95aa80bcd6697336894bb459f5f"
        assert res["architecture"] == "DenseNet201"
        assert res["classes"] == list(CLASS_NAMES)
        assert res["checks"]["file_exists"]["passed"] is True
        assert res["checks"]["safe_load"]["passed"] is True
        assert res["checks"]["architecture_dimensions"]["passed"] is True
        assert res["checks"]["class_mapping"]["passed"] is True
        assert res["checks"]["inference_smoke_test"]["passed"] is True
        assert len(res["rejection_reasons"]) == 0

    def test_missing_checkpoint_rejected(self, tmp_path):
        missing_path = str(tmp_path / "non_existent.pth")
        res = check_checkpoint_eligibility(missing_path)
        assert res["eligible"] is False
        assert any("File not found" in r for r in res["rejection_reasons"])

    def test_directory_path_rejected(self, tmp_path):
        res = check_checkpoint_eligibility(str(tmp_path))
        assert res["eligible"] is False
        assert any("directory" in r for r in res["rejection_reasons"])

    def test_empty_checkpoint_rejected(self, tmp_path):
        empty_path = str(tmp_path / "empty.pth")
        with open(empty_path, "wb") as f:
            pass
        res = check_checkpoint_eligibility(empty_path)
        assert res["eligible"] is False
        assert any("0 bytes" in r for r in res["rejection_reasons"])

    def test_corrupted_binary_rejected(self, tmp_path):
        corrupt_path = str(tmp_path / "corrupt.pth")
        with open(corrupt_path, "wb") as f:
            f.write(b"CORRUPTED_NON_PYTORCH_BYTES_DATA")
        res = check_checkpoint_eligibility(corrupt_path)
        assert res["eligible"] is False
        assert any("weights_only=True" in r for r in res["rejection_reasons"])

    def test_wrong_classifier_dimensions_rejected(self, tmp_path):
        # DenseNet201 with only 2 output classes instead of 4
        wrong_model = build_densenet201_model(num_classes=2)
        wrong_path = str(tmp_path / "wrong_dim.pth")
        torch.save({"state_dict": wrong_model.state_dict(), "classes": ["cataract", "normal"]}, wrong_path)

        res = check_checkpoint_eligibility(wrong_path)
        assert res["eligible"] is False
        assert any("dimension mismatch" in r.lower() or "shape" in r.lower() for r in res["rejection_reasons"])

    def test_wrong_class_mapping_rejected(self, tmp_path):
        model = build_densenet201_model(num_classes=4)
        wrong_order_path = str(tmp_path / "wrong_order.pth")
        # Inverted class order: glaucoma first instead of cataract
        inverted_classes = ["glaucoma", "diabetic_retinopathy", "cataract", "normal"]
        torch.save({"state_dict": model.state_dict(), "classes": inverted_classes}, wrong_order_path)

        res = check_checkpoint_eligibility(wrong_order_path)
        assert res["eligible"] is False
        assert any("class order mismatch" in r.lower() for r in res["rejection_reasons"])

    def test_nan_weights_fail_smoke_test(self, tmp_path):
        model = build_densenet201_model(num_classes=4)
        sd = model.state_dict()
        sd["classifier.weight"][0, 0] = float("nan")
        nan_path = str(tmp_path / "nan_model.pth")
        torch.save({"state_dict": sd, "classes": list(CLASS_NAMES)}, nan_path)

        res = check_checkpoint_eligibility(nan_path, run_smoke_test=True)
        assert res["eligible"] is False
        assert any("non-finite logits" in r.lower() for r in res["rejection_reasons"])


class TestCandidateComparison:
    """Verifies candidate comparison, evaluation equivalence checking, and regression detection."""

    def test_compare_candidates_equivalent_and_regression_flagged(self):
        c1 = os.path.join(ML_DIR, "weights", "eye_model.pth")
        c2 = os.path.join(ML_DIR, "weights", "eye_candidate_model.pth")

        comp = compare_checkpoints([c1, c2])
        assert len(comp["candidates"]) == 2
        assert comp["equivalent_conditions"] is True
        assert any("EQUIVALENT" in n for n in comp["comparison_notes"])

        # Must detect and highlight the 7.88% glaucoma recall regression in eye_candidate_model.pth
        assert any("glaucoma recall dropped" in n.lower() for n in comp["comparison_notes"])

        # Check per-class comparison records
        glaucoma_rec = next((p for p in comp["per_class_comparisons"] if p["class_name"] == "glaucoma"), None)
        assert glaucoma_rec is not None
        assert glaucoma_rec["recall_diff"] < -0.05
        assert glaucoma_rec["clinical_warning"] is not None

    def test_compare_candidates_non_equivalent_detected(self, valid_mock_checkpoint_a):
        # Real model has held-out test evaluation, mock model does not
        c_real = os.path.join(ML_DIR, "weights", "eye_model.pth")
        comp = compare_checkpoints([c_real, valid_mock_checkpoint_a])

        assert comp["equivalent_conditions"] is False
        assert any("NON-EQUIVALENT" in n for n in comp["comparison_notes"])


class TestSafePromotionWorkflow:
    """Verifies safe promotion, atomic selection updates, and file preservation."""

    def test_promote_ineligible_candidate_raises_error(self, temp_selection_file, tmp_path):
        bad_path = str(tmp_path / "broken.pth")
        with open(bad_path, "wb") as f:
            f.write(b"NOT_A_VALID_MODEL")

        with pytest.raises(IneligibleCheckpointError):
            promote_checkpoint(bad_path, reason="Promote corrupted model", selection_file=temp_selection_file)

    def test_promote_requires_non_empty_reason(self, valid_mock_checkpoint_a, temp_selection_file):
        with pytest.raises(ValueError) as exc:
            promote_checkpoint(valid_mock_checkpoint_a, reason="   ", selection_file=temp_selection_file)
        assert "documented reason" in str(exc.value).lower()

    def test_promote_valid_candidate_updates_active_and_records_history(
        self, valid_mock_checkpoint_a, valid_mock_checkpoint_b, temp_selection_file
    ):
        # 1. First promotion: Model A
        res_a = promote_checkpoint(
            valid_mock_checkpoint_a,
            reason="Initial champion Model A",
            selection_file=temp_selection_file,
            actor="test_runner",
        )
        assert res_a["status"] == "PROMOTED"
        assert res_a["active_checkpoint"] == os.path.abspath(valid_mock_checkpoint_a)

        status_a = get_active_checkpoint_status(temp_selection_file)
        assert status_a["active_checkpoint_path"] == os.path.abspath(valid_mock_checkpoint_a)
        assert status_a["history_count"] == 1  # Contains auto-initialized baseline

        # 2. Second promotion: Model B
        res_b = promote_checkpoint(
            valid_mock_checkpoint_b,
            reason="Upgraded champion Model B",
            selection_file=temp_selection_file,
            actor="test_runner",
        )
        assert res_b["status"] == "PROMOTED"
        assert res_b["previous_checkpoint"] == os.path.abspath(valid_mock_checkpoint_a)
        assert res_b["active_checkpoint"] == os.path.abspath(valid_mock_checkpoint_b)

        status_b = get_active_checkpoint_status(temp_selection_file)
        assert status_b["active_checkpoint_path"] == os.path.abspath(valid_mock_checkpoint_b)
        assert status_b["history_count"] == 2

        # Verify historical entry preserved Model A details
        last_hist = status_b["history"][-1]
        assert last_hist["checkpoint_path"] == os.path.abspath(valid_mock_checkpoint_a)
        assert "Initial champion Model A" in last_hist["promotion_reason"]

    def test_promote_preserves_underlying_binary_files(
        self, valid_mock_checkpoint_a, valid_mock_checkpoint_b, temp_selection_file
    ):
        """Verifies that promotion NEVER mutates or overwrites the .pth files on disk."""
        sha_a_before = compute_file_sha256(valid_mock_checkpoint_a)
        sha_b_before = compute_file_sha256(valid_mock_checkpoint_b)

        promote_checkpoint(valid_mock_checkpoint_a, reason="Promote A", selection_file=temp_selection_file)
        promote_checkpoint(valid_mock_checkpoint_b, reason="Promote B", selection_file=temp_selection_file)

        sha_a_after = compute_file_sha256(valid_mock_checkpoint_a)
        sha_b_after = compute_file_sha256(valid_mock_checkpoint_b)

        assert sha_a_before == sha_a_after, "CRITICAL: Model A was modified on disk!"
        assert sha_b_before == sha_b_after, "CRITICAL: Model B was modified on disk!"

    def test_promote_dry_run_leaves_selection_unchanged(
        self, valid_mock_checkpoint_a, valid_mock_checkpoint_b, temp_selection_file
    ):
        promote_checkpoint(valid_mock_checkpoint_a, reason="Promote A", selection_file=temp_selection_file)
        status_before = get_active_checkpoint_status(temp_selection_file)

        dry = promote_checkpoint(valid_mock_checkpoint_b, reason="Promote B dry run", selection_file=temp_selection_file, dry_run=True)
        assert dry["status"] == "DRY_RUN"

        status_after = get_active_checkpoint_status(temp_selection_file)
        assert status_after["active_checkpoint_path"] == status_before["active_checkpoint_path"]

    def test_promote_already_active_checkpoint(
        self, valid_mock_checkpoint_a, temp_selection_file
    ):
        promote_checkpoint(valid_mock_checkpoint_a, reason="Promote A", selection_file=temp_selection_file)
        res = promote_checkpoint(valid_mock_checkpoint_a, reason="Promote A again", selection_file=temp_selection_file)
        assert res["status"] == "ALREADY_ACTIVE"


class TestReversibleRollbackWorkflow:
    """Verifies reversible rollback, historical restoration, and integrity checks."""

    def test_rollback_without_history_raises_error(self, tmp_path):
        sel_path = str(tmp_path / "empty_selection.json")
        save_atomic_selection_file(
            {
                "schema_version": "1.0.0",
                "active_checkpoint_path": "ml/weights/eye_model.pth",
                "active_checkpoint_sha256": "e819440c35f166c90e6469b071cc83136857e95aa80bcd6697336894bb459f5f",
                "history": [],
            },
            sel_path,
        )

        with pytest.raises(RollbackUnavailableError):
            rollback_checkpoint(selection_file=sel_path)

    def test_successful_rollback_restores_previous_checkpoint(
        self, valid_mock_checkpoint_a, valid_mock_checkpoint_b, temp_selection_file
    ):
        # Promote A then promote B
        promote_checkpoint(valid_mock_checkpoint_a, reason="Promote A", selection_file=temp_selection_file)
        promote_checkpoint(valid_mock_checkpoint_b, reason="Promote B", selection_file=temp_selection_file)

        # Active is now B
        assert get_active_checkpoint_status(temp_selection_file)["active_checkpoint_path"] == os.path.abspath(valid_mock_checkpoint_b)

        # Rollback -> should restore A
        roll_res = rollback_checkpoint(reason="Regression observed in B", selection_file=temp_selection_file)
        assert roll_res["status"] == "ROLLED_BACK"
        assert roll_res["active_checkpoint"] == os.path.abspath(valid_mock_checkpoint_a)
        assert roll_res["rolled_back_from"] == os.path.abspath(valid_mock_checkpoint_b)

        # Verify active selection is now A
        status = get_active_checkpoint_status(temp_selection_file)
        assert status["active_checkpoint_path"] == os.path.abspath(valid_mock_checkpoint_a)

    def test_rollback_dry_run(
        self, valid_mock_checkpoint_a, valid_mock_checkpoint_b, temp_selection_file
    ):
        promote_checkpoint(valid_mock_checkpoint_a, reason="Promote A", selection_file=temp_selection_file)
        promote_checkpoint(valid_mock_checkpoint_b, reason="Promote B", selection_file=temp_selection_file)

        dry = rollback_checkpoint(selection_file=temp_selection_file, dry_run=True)
        assert dry["status"] == "DRY_RUN"
        assert dry["target_restore"] == os.path.abspath(valid_mock_checkpoint_a)

        # Verify state is still B
        status = get_active_checkpoint_status(temp_selection_file)
        assert status["active_checkpoint_path"] == os.path.abspath(valid_mock_checkpoint_b)

    def test_rollback_rejects_missing_target_file(
        self, valid_mock_checkpoint_a, valid_mock_checkpoint_b, temp_selection_file
    ):
        promote_checkpoint(valid_mock_checkpoint_a, reason="Promote A", selection_file=temp_selection_file)
        promote_checkpoint(valid_mock_checkpoint_b, reason="Promote B", selection_file=temp_selection_file)

        # Simulate historical artifact being deleted
        os.remove(valid_mock_checkpoint_a)

        with pytest.raises(CheckpointNotFoundError):
            rollback_checkpoint(selection_file=temp_selection_file)

    def test_rollback_rejects_altered_checksum(
        self, valid_mock_checkpoint_a, valid_mock_checkpoint_b, temp_selection_file
    ):
        promote_checkpoint(valid_mock_checkpoint_a, reason="Promote A", selection_file=temp_selection_file)
        promote_checkpoint(valid_mock_checkpoint_b, reason="Promote B", selection_file=temp_selection_file)

        # Alter historical checkpoint file A
        with open(valid_mock_checkpoint_a, "ab") as f:
            f.write(b"MUTATED_EXTRA_BYTES")

        with pytest.raises(CheckpointCorruptError):
            rollback_checkpoint(selection_file=temp_selection_file)


class TestRecoveryFromInterruption:
    """Verifies recovery from interrupted operations and orphaned temporary files."""

    def test_recovery_cleans_orphaned_tmp_file(self, tmp_path):
        sel_path = str(tmp_path / "checkpoint_selection.json")
        tmp_file = f"{sel_path}.tmp"
        with open(tmp_file, "w") as f:
            f.write("UNFINISHED_PARTIAL_JSON_WRITE")

        assert os.path.exists(tmp_file)
        recover_selection_state(sel_path)
        assert not os.path.exists(tmp_file)

    def test_recovery_repairs_corrupted_selection_file(self, tmp_path):
        sel_path = str(tmp_path / "checkpoint_selection.json")
        with open(sel_path, "w") as f:
            f.write("{ INVALID_JSON_DATA_CORRUPT")

        rec = recover_selection_state(sel_path)
        assert "active_checkpoint_path" in rec
        assert "eye_model.pth" in rec["active_checkpoint_path"]


class TestCLIIntegration:
    """Verifies CLI execution of checkpoint management commands."""

    def test_cli_status_command(self):
        res = subprocess.run(
            [sys.executable, "ml/checkpoint_manager.py", "status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0
        assert "ACTIVE PRODUCTION CHECKPOINT STATUS" in res.stdout
        assert "eye_model.pth" in res.stdout

    def test_cli_list_command(self):
        res = subprocess.run(
            [sys.executable, "ml/checkpoint_manager.py", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0
        assert "CHECKPOINT ARTIFACT AUDIT" in res.stdout
        assert "eye_model.pth" in res.stdout
        assert "eye_candidate_model.pth" in res.stdout

    def test_cli_check_command(self):
        res = subprocess.run(
            [sys.executable, "ml/checkpoint_manager.py", "check", "ml/weights/eye_model.pth"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0
        assert "ELIGIBILITY AUDIT" in res.stdout
        assert "Status:   ELIGIBLE" in res.stdout

    def test_cli_compare_command(self):
        res = subprocess.run(
            [
                sys.executable,
                "ml/checkpoint_manager.py",
                "compare",
                "ml/weights/eye_model.pth",
                "ml/weights/eye_candidate_model.pth",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0
        assert "CHECKPOINT CANDIDATE COMPARISON" in res.stdout
        assert "glaucoma recall dropped" in res.stdout
