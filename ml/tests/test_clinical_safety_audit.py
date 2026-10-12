"""
Test Suite for Part 53 — Clinical-Safety and Performance Audit Verification.

Verifies:
1. Data split independence and zero patient/image/hash leakage.
2. Checkpoint selection isolation from held-out test data.
3. Comprehensive metric computation correctness (balanced accuracy, ROC-AUC, PR-AUC, ECE, Brier).
4. Quality-gate cohort partitioning (available, passed, rejected).
5. Existence and schema compliance of clinical audit artifacts.
"""

import json
import os
import numpy as np
import pytest
import torch

from ml.clinical_safety_audit import (
    audit_evaluation_protocol,
    compute_comprehensive_metrics,
)
from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES, load_split_manifest
from ml.eye_model import DEFAULT_CHECKPOINT_PATH

MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "dataset", "split_manifest.json"
)
REPORT_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "evaluation_results", "clinical_safety_audit_report.json"
)
REPORT_MD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "evaluation_results", "CLINICAL_SAFETY_AUDIT.md"
)


class TestEvaluationProtocolAndDataIntegrity:
    """Verifies that dataset partitioning and model selection adhere to clinical evaluation standards."""

    def test_zero_leakage_and_strict_disjointness(self):
        """Confirms 0 patient, filepath, and SHA256 overlap between all split pairs."""
        audit = audit_evaluation_protocol(MANIFEST_PATH, DEFAULT_CHECKPOINT_PATH)
        assert audit["is_strictly_disjoint"] is True
        assert audit["patient_overlap"]["train_val"] == 0
        assert audit["patient_overlap"]["train_test"] == 0
        assert audit["patient_overlap"]["val_test"] == 0
        assert audit["filepath_overlap"]["train_val"] == 0
        assert audit["filepath_overlap"]["train_test"] == 0
        assert audit["filepath_overlap"]["val_test"] == 0
        assert audit["sha256_overlap"]["train_val"] == 0
        assert audit["sha256_overlap"]["train_test"] == 0
        assert audit["sha256_overlap"]["val_test"] == 0
        assert audit["missing_files"]["train"] == 0
        assert audit["missing_files"]["val"] == 0
        assert audit["missing_files"]["test"] == 0

    def test_checkpoint_selection_isolated_from_test_split(self):
        """Confirms champion checkpoint was selected on validation accuracy without test tuning."""
        audit = audit_evaluation_protocol(MANIFEST_PATH, DEFAULT_CHECKPOINT_PATH)
        ckpt_meta = audit["checkpoint_selection_audit"]
        assert ckpt_meta["architecture"] == "DenseNet201"
        assert ckpt_meta["epoch"] == 6
        assert ckpt_meta["classes"] == CLASS_NAMES
        assert abs(ckpt_meta["best_val_accuracy"] - (781 / 865)) < 1e-5


class TestMetricCalculatorsAndNumericalInvariants:
    """Verifies correctness of statistical and calibration metrics."""

    def test_metrics_calculator_perfect_prediction(self):
        """Perfect predictions yield 1.0 accuracy, 0.0 ECE, and clean 4x4 matrix."""
        y_true = [0, 1, 2, 3] * 10
        y_pred = list(y_true)
        probs = np.zeros((40, 4))
        for i, t in enumerate(y_true):
            probs[i, t] = 1.0

        res = compute_comprehensive_metrics(y_true, y_pred, probs, latencies=[5.0] * 40)
        assert res["accuracy"] == 1.0
        assert res["balanced_accuracy"] == 1.0
        assert res["macro_f1"] == 1.0
        assert res["calibration"]["brier_score"] == 0.0
        assert res["calibration"]["expected_calibration_error_ece"] == 0.0
        assert res["latency_ms"]["median_ms"] == 5.0

    def test_metrics_calculator_zero_division_safety(self):
        """Calculator handles unpredicted classes gracefully with 0.0 metrics rather than crashing."""
        y_true = [0, 1, 2, 3]
        y_pred = [0, 0, 0, 0]  # Only predicts class 0
        probs = np.zeros((4, 4))
        probs[:, 0] = 1.0

        res = compute_comprehensive_metrics(y_true, y_pred, probs)
        assert res["accuracy"] == 0.25
        assert 0.0 <= res["macro_f1"] <= 1.0
        assert res["per_class"]["diabetic_retinopathy"]["precision"] == 0.0


class TestClinicalAuditReportArtifacts:
    """Verifies that persistent JSON and Markdown audit reports exist and contain required sections."""

    def test_clinical_safety_audit_report_json_validity(self):
        """Validates schema, metrics, and non-empty sections of clinical_safety_audit_report.json."""
        assert os.path.exists(REPORT_JSON_PATH), f"Report missing at {REPORT_JSON_PATH}"
        with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "evaluation_protocol_verification" in data
        assert "sample_cohort_breakdown" in data
        assert "metrics_comparison" in data
        assert "error_and_uncertainty_audit" in data
        assert "clinical_safety_boundaries" in data
        assert "final_verdict" in data

        cohort = data["sample_cohort_breakdown"]
        assert cohort["total_test_images_available"] == 894
        assert cohort["total_test_images_evaluated"] == 894
        assert cohort["quality_gate_passed_evaluated"] == 378
        assert cohort["quality_gate_rejected_count"] == 516

        full_m = data["metrics_comparison"]["full_held_out_test_set_n894"]
        assert 0.85 <= full_m["accuracy"] <= 0.95
        assert 0.85 <= full_m["balanced_accuracy"] <= 0.95
        assert 0.85 <= full_m["macro_f1"] <= 0.95
        assert full_m["macro_roc_auc"] >= 0.95
        assert full_m["macro_pr_auc"] >= 0.90

    def test_clinical_safety_audit_markdown_report_validity(self):
        """Validates that CLINICAL_SAFETY_AUDIT.md exists and contains clinical disclaimer and tables."""
        assert os.path.exists(REPORT_MD_PATH), f"Markdown missing at {REPORT_MD_PATH}"
        with open(REPORT_MD_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        assert "DenseNet201 Eye Disease Classifier" in content
        assert "Evaluation Protocol & Data Integrity Verification" in content
        assert "Quantitative Performance & Generalization Metrics" in content
        assert "Canonical 4x4 Confusion Matrix" in content
        assert "Error Analysis, Uncertainty & Failure Inspection" in content
        assert "Clinical Safety Boundaries & Operational Guardrails" in content
        assert "INVESTIGATIONAL" in content
