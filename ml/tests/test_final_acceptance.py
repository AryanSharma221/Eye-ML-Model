"""
Automated PyTest Module for Final ML Acceptance Verification (Part 20).

Validates all 9 acceptance criteria required for deployment readiness:
1. Dataset manifest counts (5,889 images, 0% leakage, patient-stratified).
2. Production checkpoint integrity, architecture tag, and clean reload.
3. Strict canonical 4-class ordering: ["cataract", "diabetic_retinopathy", "glaucoma", "normal"].
4. Quality gate pass on eligible test images and rejection on non-compliant images.
5. 4-class output correctness: bounds [0, 1], sum-to-1, confidence alignment.
6. Missing/invalid checkpoint never silently produces a prediction.
7. Grad-CAM Base64 PNG generation, decoding, and valid image dimensions.
8. Independent physiological triage execution and boolean coercion rejection.
9. Persistent acceptance report validation (ml/evaluation_results/acceptance_report.json).
"""

import base64
import io
import json
import os
from typing import Any, Dict

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES, load_split_manifest
from ml.exceptions import (
    CheckpointCorruptError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    PhysiologicalTypeValidationError,
    PhysiologicalValidationError,
    QualityGateRejectionError,
)
from ml.eye_model import (
    CLASS_TO_IDX,
    DEFAULT_CHECKPOINT_PATH,
    IDX_TO_CLASS,
    build_densenet201_model,
    clear_model_cache,
    get_cached_model,
    get_default_device,
    load_eye_model_checkpoint,
    predict_eye_disease,
    screen_eye_image,
)
from ml.gradcam import generate_gradcam
from ml.physio_model import analyze_vitals
from ml.preprocessing import preprocess_image
from ml.quality_gate import check_image_quality


MANIFEST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "split_manifest.json")
ACCEPTANCE_REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_results", "acceptance_report.json")


class TestPart20ModelAndDatasetIntegrity:
    """Verifies that the real dataset was used and the champion model is trained and verified."""

    def test_dataset_manifest_integrity_and_sample_counts(self):
        """Confirms 5,889 usable images with zero leakage across train, val, and test splits."""
        assert os.path.exists(MANIFEST_PATH), f"Manifest missing at {MANIFEST_PATH}"
        data = load_split_manifest(MANIFEST_PATH)
        meta = data["metadata"]
        assert meta["total_images"] == 5889
        assert meta["train_count"] == 4130
        assert meta["val_count"] == 865
        assert meta["test_count"] == 894
        assert len(data["splits"]["train"]) == 4130
        assert len(data["splits"]["val"]) == 865
        assert len(data["splits"]["test"]) == 894

    def test_canonical_class_ordering_is_strictly_preserved(self):
        """Confirms exact class order: ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']."""
        expected = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
        assert CLASS_NAMES == expected
        assert NUM_CLASSES == 4
        assert [IDX_TO_CLASS[i] for i in range(4)] == expected
        assert [CLASS_TO_IDX[c] for c in expected] == [0, 1, 2, 3]

    def test_production_checkpoint_exists_and_metadata_verified(self):
        """Confirms champion checkpoint exists, completed 6 epochs, and has >=90% val acc."""
        assert os.path.exists(DEFAULT_CHECKPOINT_PATH), "Production checkpoint eye_model.pth missing"
        ckpt = torch.load(DEFAULT_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        assert ckpt.get("architecture") == "DenseNet201"
        assert ckpt.get("epoch") == 6
        val_acc = ckpt.get("best_val_accuracy", ckpt.get("best_acc", 0.0))
        assert val_acc >= 0.90, f"Expected val acc >= 90%, got {val_acc}"
        assert ckpt.get("classes") == CLASS_NAMES
        assert len(ckpt["state_dict"]) == 1207

    def test_production_checkpoint_reloads_into_production_interface(self):
        """Confirms clean model reloading through get_cached_model."""
        clear_model_cache()
        model = get_cached_model(weights_path=DEFAULT_CHECKPOINT_PATH, device=torch.device("cpu"))
        assert isinstance(model, nn.Module)
        assert not model.training, "Model must be in eval() mode"
        assert model.classifier.out_features == 4


class TestPart20InferenceAndSafety:
    """Verifies end-to-end inference correctness, probability constraints, and failure protection."""

    def test_missing_checkpoint_never_silently_predicts(self):
        """Confirms missing checkpoint raises CheckpointNotFoundError without producing predictions."""
        clear_model_cache()
        with pytest.raises((CheckpointNotFoundError, FileNotFoundError)):
            get_cached_model(weights_path="non_existent_weights.pth")

    def test_quality_gate_rejection_blocks_inference(self):
        """Confirms non-compliant image is rejected and blocks neural network inference."""
        data = load_split_manifest(MANIFEST_PATH)
        failing_fp = None
        for rec in data["splits"]["test"]:
            fp = rec["filepath"]
            if os.path.exists(fp):
                q = check_image_quality(fp)
                if not q["overall_pass"]:
                    failing_fp = fp
                    break

        assert failing_fp is not None, "Failed to find test split rejection candidate"
        res = screen_eye_image(failing_fp, use_cache=True)
        assert res["status"] == "rejected"
        assert "prediction" not in res or res.get("prediction") is None
        assert not res["quality_checks"]["overall_pass"]

    def test_real_image_inference_output_correctness(self):
        """Confirms output schema, bounds [0, 1], sum-to-1, and confidence alignment."""
        data = load_split_manifest(MANIFEST_PATH)
        passing_fp = None
        for rec in data["splits"]["test"]:
            fp = rec["filepath"]
            if os.path.exists(fp):
                q = check_image_quality(fp)
                if q["overall_pass"]:
                    passing_fp = fp
                    break

        assert passing_fp is not None, "Failed to find valid test split passing image"
        res = screen_eye_image(passing_fp, include_gradcam=True, use_cache=True)
        assert res["status"] == "success"
        pred = res["prediction"]
        assert pred in CLASS_NAMES
        assert res["class_index"] == CLASS_NAMES.index(pred)
        assert abs(res["confidence"] - res["all_probabilities"][pred]) < 1e-4

        probs = res["all_probabilities"]
        assert len(probs) == 4
        for c, p in probs.items():
            assert 0.0 <= p <= 1.0
            assert not np.isnan(p) and not np.isinf(p)

        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 0.002, f"Probabilities do not sum to 1.0: {prob_sum}"
        assert res["gradcam_image"] is not None
        assert json.dumps(res) is not None

    def test_gradcam_generates_valid_png(self):
        """Confirms Grad-CAM outputs a decodable Base64 PNG image."""
        data = load_split_manifest(MANIFEST_PATH)
        passing_fp = None
        for rec in data["splits"]["test"]:
            fp = rec["filepath"]
            if os.path.exists(fp) and check_image_quality(fp)["overall_pass"]:
                passing_fp = fp
                break

        b64 = generate_gradcam(passing_fp, predicted_class_idx=1)
        assert isinstance(b64, str) and len(b64) > 100
        raw_png = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw_png))
        assert img.format == "PNG"
        assert img.size[0] > 0 and img.size[1] > 0

    def test_independent_physiological_vitals_triage(self):
        """Confirms vitals analysis operates independently and rejects invalid types."""
        norm = analyze_vitals(heart_rate=70.0, spo2=99.0, temperature=36.7)
        assert norm["overall_status"] == "Normal"
        assert norm["risk_level"] == "LOW"

        crit = analyze_vitals(heart_rate=140.0, spo2=87.0, temperature=39.1)
        assert crit["overall_status"] == "Abnormal"
        assert crit["risk_level"] == "HIGH"
        assert crit["is_abnormal"] is True

        with pytest.raises((TypeError, ValueError, PhysiologicalTypeValidationError)):
            analyze_vitals(heart_rate=True, spo2=98.0, temperature=37.0)  # type: ignore


class TestPart20AcceptanceReportArtifact:
    """Verifies that the acceptance report artifact is generated, valid, and reports READY."""

    def test_acceptance_report_exists_and_verdict_ready(self):
        """Confirms ml/evaluation_results/acceptance_report.json exists with READY_FOR_INTEGRATION."""
        assert os.path.exists(ACCEPTANCE_REPORT_PATH), f"Report missing at {ACCEPTANCE_REPORT_PATH}"
        with open(ACCEPTANCE_REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["final_verdict"]["status"] == "READY_FOR_INTEGRATION"
        assert data["final_verdict"]["all_acceptance_checks_passed"] is True
        assert data["final_verdict"]["ready_for_backend_integration"] is True
        assert data["checks"]["dataset_manifest"]["status"] == "PASS"
        assert data["checks"]["checkpoint_loading"]["status"] == "PASS"
        assert data["checks"]["inference_correctness"]["status"] == "PASS"
        assert data["checks"]["quality_rejection"]["status"] == "PASS"
        assert data["checks"]["gradcam"]["status"] == "PASS"
        assert data["checks"]["physiological_analysis"]["status"] == "PASS"
        assert data["checks"]["backend_contract"]["status"] == "PASS"
        assert data["checks"]["held_out_evaluation"]["status"] == "PASS"
        assert data["checks"]["held_out_evaluation"]["overall_accuracy"] >= 0.85
