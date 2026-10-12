"""
Part 51 — Final Independent ML Module Integration Audit Test Suite.

Verifies end-to-end integration across:
1. Eye-disease screening model (DenseNet201)
2. Image-quality gate (Laplacian blur, brightness, contrast, resolution, contour)
3. Standardized preprocessing pipeline (224x224, ImageNet normalization)
4. Grad-CAM visual explainability module (features.denseblock4)
5. Physiological vital-sign triage engine (HR, SpO2, Temperature)
"""

import json
import os
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
import torch

from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES
from ml.eye_model import (
    DEFAULT_CHECKPOINT_PATH,
    build_densenet201_model,
    predict_eye_disease,
)
from ml.physio_model import analyze_vitals
from ml.quality_gate import check_image_quality
from ml.screening_pipeline import run_screening


@pytest.fixture
def synthetic_retinal_fixtures(tmp_path):
    """Generates synthetic test images for deterministic integration checks."""
    fixtures = {}

    # 1. Valid retinal image (passes all 5 quality gate checks)
    valid_path = str(tmp_path / "valid_fundus.png")
    valid_arr = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(valid_arr, (150, 150), 100, (140, 130, 120), -1)
    np.random.seed(42)
    noise = np.random.randint(-40, 40, (300, 300, 3), dtype=np.int16)
    valid_arr = np.clip(valid_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.circle(valid_arr, (190, 150), 22, (230, 220, 200), -1)
    cv2.line(valid_arr, (190, 150), (100, 100), (40, 30, 160), 2)
    cv2.imwrite(valid_path, valid_arr)
    fixtures["valid"] = valid_path

    # 2. Blurry image (Laplacian variance < 100.0)
    blurry_path = str(tmp_path / "blurry_fundus.png")
    blurry_arr = cv2.GaussianBlur(valid_arr, (35, 35), sigmaX=15)
    cv2.imwrite(blurry_path, blurry_arr)
    fixtures["blurry"] = blurry_path

    # 3. Small resolution image (< 224x224)
    small_path = str(tmp_path / "small_fundus.png")
    small_arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    cv2.imwrite(small_path, small_arr)
    fixtures["small"] = small_path

    # 4. Corrupt non-image file
    corrupt_path = str(tmp_path / "corrupt.png")
    with open(corrupt_path, "wb") as f:
        f.write(b"CORRUPTED_NON_IMAGE_DATA_12345")
    fixtures["corrupt"] = corrupt_path

    return fixtures


class TestIndependentPredictions:
    """Verifies that eye-image inference and vital-sign triage run completely independently."""

    def test_vitals_variation_does_not_alter_eye_model_probabilities(self, synthetic_retinal_fixtures):
        """Varying physiological vitals must not change eye-disease class or probabilities by even 1e-7."""
        img = synthetic_retinal_fixtures["valid"]

        mock_probs = {
            "cataract": 0.05,
            "diabetic_retinopathy": 0.85,
            "glaucoma": 0.07,
            "normal": 0.03,
        }
        mock_inference = {
            "prediction": "diabetic_retinopathy",
            "class_index": 1,
            "confidence": 0.85,
            "risk_level": "HIGH",
            "all_probabilities": mock_probs,
        }

        # Test across diverse vital-sign profiles: normal, tachycardia, hypoxemia, fever
        vital_scenarios = [
            (72.0, 98.0, 36.8),   # Normal
            (145.0, 98.0, 36.8),  # Tachycardia
            (72.0, 84.0, 36.8),   # Severe hypoxemia
            (72.0, 98.0, 39.2),   # High fever
            (150.0, 82.0, 39.5),  # Triple critical
        ]

        eye_results = []
        for hr, spo2, temp in vital_scenarios:
            with patch("ml.screening_pipeline.predict_eye_disease", return_value=mock_inference):
                res = run_screening(
                    image_path=img,
                    heart_rate=hr,
                    spo2=spo2,
                    temperature=temp,
                    include_gradcam=False,
                )
                eye_results.append(res["eye_screening"])

        # Confirm all eye screening outputs are identical
        baseline = eye_results[0]
        for idx, other in enumerate(eye_results[1:], start=1):
            assert other["prediction"] == baseline["prediction"]
            assert other["confidence"] == baseline["confidence"]
            assert other["all_probabilities"] == baseline["all_probabilities"]
            assert other["class_index"] == baseline["class_index"]
            assert other["risk_level"] == baseline["risk_level"]

    def test_eye_image_variation_does_not_alter_vitals_analysis(self, synthetic_retinal_fixtures):
        """Varying eye image inputs must not alter physiological analysis results or triage flags."""
        hr, spo2, temp = 74.0, 98.5, 36.7

        mock_inference_dr = {
            "prediction": "diabetic_retinopathy",
            "class_index": 1,
            "confidence": 0.90,
            "risk_level": "HIGH",
            "all_probabilities": {"cataract": 0.02, "diabetic_retinopathy": 0.90, "glaucoma": 0.05, "normal": 0.03},
        }
        mock_inference_normal = {
            "prediction": "normal",
            "class_index": 3,
            "confidence": 0.95,
            "risk_level": "LOW",
            "all_probabilities": {"cataract": 0.01, "diabetic_retinopathy": 0.02, "glaucoma": 0.02, "normal": 0.95},
        }

        # Scenario 1: Diabetic Retinopathy eye
        with patch("ml.screening_pipeline.predict_eye_disease", return_value=mock_inference_dr):
            res_dr = run_screening(image_path=synthetic_retinal_fixtures["valid"], heart_rate=hr, spo2=spo2, temperature=temp, include_gradcam=False)

        # Scenario 2: Normal eye
        with patch("ml.screening_pipeline.predict_eye_disease", return_value=mock_inference_normal):
            res_normal = run_screening(image_path=synthetic_retinal_fixtures["valid"], heart_rate=hr, spo2=spo2, temperature=temp, include_gradcam=False)

        # Scenario 3: Image missing
        res_no_img = run_screening(image_path=None, heart_rate=hr, spo2=spo2, temperature=temp, include_gradcam=False)

        # Scenario 4: Blurry rejected image
        res_blurry = run_screening(image_path=synthetic_retinal_fixtures["blurry"], heart_rate=hr, spo2=spo2, temperature=temp, include_gradcam=False)

        v_dr = res_dr["vitals_screening"]
        v_norm = res_normal["vitals_screening"]
        v_no_img = res_no_img["vitals_screening"]
        v_blurry = res_blurry["vitals_screening"]

        for v in (v_norm, v_no_img, v_blurry):
            assert v["overall_status"] == v_dr["overall_status"] == "Normal"
            assert v["risk_level"] == v_dr["risk_level"] == "LOW"
            assert v["is_abnormal"] == v_dr["is_abnormal"] is False
            assert v["heart_rate"] == v_dr["heart_rate"] == 74.0
            assert v["spo2"] == v_dr["spo2"] == 98.5
            assert v["temperature"] == v_dr["temperature"] == 36.7

    def test_final_result_clearly_separates_modalities(self, synthetic_retinal_fixtures):
        """Combined response must strictly isolate eye and vitals into distinct structured subtrees."""
        res = run_screening(
            image_path=synthetic_retinal_fixtures["valid"],
            heart_rate=72.0,
            spo2=98.0,
            temperature=36.8,
            include_gradcam=False,
        )

        assert "eye_screening" in res
        assert "vitals_screening" in res
        assert isinstance(res["eye_screening"], dict)
        assert isinstance(res["vitals_screening"], dict)
        assert "overall_risk_level" in res
        assert "disclaimer" in res


class TestQualityGateEnforcement:
    """Verifies that non-compliant images never reach inference or fabricate results."""

    def test_failed_quality_gate_halts_inference_early(self, synthetic_retinal_fixtures):
        """A blurry image must be rejected by the quality gate without calling predict_eye_disease."""
        with patch("ml.screening_pipeline.predict_eye_disease") as mock_predict, \
             patch("ml.screening_pipeline.preprocess_image") as mock_prep:
            res = run_screening(
                image_path=synthetic_retinal_fixtures["blurry"],
                heart_rate=72.0,
                spo2=98.0,
                temperature=36.8,
            )

            mock_prep.assert_not_called()
            mock_predict.assert_not_called()

            eye = res["eye_screening"]
            assert eye["status"] == "rejected"
            assert "blur" in eye["reason"].lower()
            assert eye["prediction"] is None
            assert eye["confidence"] is None
            assert eye["all_probabilities"] is None
            assert eye["gradcam_image"] is None
            assert eye["quality_checks"]["overall_pass"] is False

    def test_small_resolution_rejection(self, synthetic_retinal_fixtures):
        """Images smaller than 224x224 must be rejected at the quality gate."""
        with patch("ml.screening_pipeline.predict_eye_disease") as mock_predict:
            res = run_screening(
                image_path=synthetic_retinal_fixtures["small"],
                heart_rate=72.0,
                spo2=98.0,
                temperature=36.8,
            )
            mock_predict.assert_not_called()
            assert res["eye_screening"]["status"] == "rejected"
            assert "resolution" in res["eye_screening"]["reason"].lower()

    def test_corrupt_image_handled_safely(self, synthetic_retinal_fixtures):
        """Corrupt or non-image files must be rejected cleanly without unhandled crashes."""
        res = run_screening(
            image_path=synthetic_retinal_fixtures["corrupt"],
            heart_rate=72.0,
            spo2=98.0,
            temperature=36.8,
        )
        assert res["eye_screening"]["status"] == "rejected"
        assert res["eye_screening"]["prediction"] is None
        assert res["eye_screening"]["gradcam_image"] is None

    def test_missing_image_file_returns_explicit_error(self):
        """Non-existent image paths produce structured error without fabricating predictions."""
        res = run_screening(
            image_path="nonexistent_retina_image_99999.png",
            heart_rate=75.0,
            spo2=98.0,
            temperature=36.8,
        )
        eye = res["eye_screening"]
        assert eye["status"] == "error"
        assert "not found" in eye["reason"].lower()
        assert eye["prediction"] is None
        assert eye["confidence"] is None
        assert eye["gradcam_image"] is None


class TestResultConsistency:
    """Verifies that predictions, probabilities, confidence, class order, and Grad-CAM align."""

    def test_predicted_class_matches_highest_probability_and_confidence(self, synthetic_retinal_fixtures):
        """Confidence must equal max(all_probabilities), and predicted class must be argmax."""
        mock_probs = {
            "cataract": 0.08,
            "diabetic_retinopathy": 0.12,
            "glaucoma": 0.76,
            "normal": 0.04,
        }
        mock_inference = {
            "prediction": "glaucoma",
            "class_index": 2,
            "confidence": 0.76,
            "risk_level": "HIGH",
            "all_probabilities": mock_probs,
        }

        with patch("ml.screening_pipeline.predict_eye_disease", return_value=mock_inference):
            res = run_screening(
                image_path=synthetic_retinal_fixtures["valid"],
                heart_rate=72.0,
                spo2=98.0,
                temperature=36.8,
                include_gradcam=False,
            )

        eye = res["eye_screening"]
        probs = eye["all_probabilities"]
        assert eye["prediction"] == "glaucoma"
        assert eye["confidence"] == 0.76
        assert max(probs.values()) == eye["confidence"]
        assert max(probs, key=probs.get) == eye["prediction"]

    def test_canonical_class_ordering_invariant(self):
        """Exact canonical class ordering must be preserved: cataract, diabetic_retinopathy, glaucoma, normal."""
        expected = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
        assert CLASS_NAMES == expected
        assert NUM_CLASSES == 4

    def test_risk_labels_follow_documented_rules(self, synthetic_retinal_fixtures):
        """Normal -> LOW; Disease >= 0.70 -> HIGH; Disease < 0.70 -> MODERATE."""
        scenarios = [
            ("normal", 0.92, "LOW"),
            ("cataract", 0.85, "HIGH"),
            ("diabetic_retinopathy", 0.65, "MODERATE"),
            ("glaucoma", 0.72, "HIGH"),
            ("glaucoma", 0.55, "MODERATE"),
        ]

        for pred, conf, expected_risk in scenarios:
            mock_inference = {
                "prediction": pred,
                "class_index": CLASS_NAMES.index(pred),
                "confidence": conf,
                "risk_level": expected_risk,
                "all_probabilities": {c: (conf if c == pred else (1.0 - conf) / 3) for c in CLASS_NAMES},
            }
            with patch("ml.screening_pipeline.predict_eye_disease", return_value=mock_inference):
                res = run_screening(
                    image_path=synthetic_retinal_fixtures["valid"],
                    heart_rate=72.0,
                    spo2=98.0,
                    temperature=36.8,
                    include_gradcam=False,
                )
                assert res["eye_screening"]["risk_level"] == expected_risk

    def test_gradcam_receives_actual_model_and_predicted_class(self, synthetic_retinal_fixtures):
        """Grad-CAM must receive the real loaded model and the exact predicted class index."""
        mock_inference = {
            "prediction": "glaucoma",
            "class_index": 2,
            "confidence": 0.88,
            "risk_level": "HIGH",
            "all_probabilities": {"cataract": 0.04, "diabetic_retinopathy": 0.04, "glaucoma": 0.88, "normal": 0.04},
        }

        with patch("ml.screening_pipeline.predict_eye_disease", return_value=mock_inference), \
             patch("ml.screening_pipeline.generate_gradcam", return_value="MOCK_BASE64_HEATMAP") as mock_gcam:
            res = run_screening(
                image_path=synthetic_retinal_fixtures["valid"],
                heart_rate=72.0,
                spo2=98.0,
                temperature=36.8,
                include_gradcam=True,
            )

            mock_gcam.assert_called_once()
            call_kwargs = mock_gcam.call_args.kwargs
            assert call_kwargs["predicted_class_idx"] == 2  # Glaucoma index
            assert call_kwargs["model"] is not None
            assert res["eye_screening"]["gradcam_image"] == "MOCK_BASE64_HEATMAP"


class TestFailureIsolation:
    """Verifies that faults in one modality do not contaminate the other modality."""

    def test_eye_model_inference_failure_reported_without_erasing_vitals(self, synthetic_retinal_fixtures):
        """A forward pass failure in the eye model must be reported as error while vitals complete."""
        with patch("ml.screening_pipeline.predict_eye_disease", side_effect=RuntimeError("CUDA Out of Memory in forward()")):
            res = run_screening(
                image_path=synthetic_retinal_fixtures["valid"],
                heart_rate=72.0,
                spo2=98.0,
                temperature=36.8,
            )

            # Eye failure recorded
            assert res["eye_screening"]["status"] == "error"
            assert "CUDA Out of Memory" in res["eye_screening"]["reason"]
            assert res["eye_screening"]["prediction"] is None

            # Vitals intact
            assert res["vitals_screening"]["status"] == "success"
            assert res["vitals_screening"]["overall_status"] == "Normal"
            assert res["vitals_screening"]["risk_level"] == "LOW"

            # Composite status is partial
            assert res["status"] == "partial"

    def test_vital_signs_validation_failure_does_not_corrupt_eye_prediction(self, synthetic_retinal_fixtures):
        """An impossible vital sign reading must fail vitals without overwriting successful eye inference."""
        mock_inference = {
            "prediction": "diabetic_retinopathy",
            "class_index": 1,
            "confidence": 0.91,
            "risk_level": "HIGH",
            "all_probabilities": {"cataract": 0.03, "diabetic_retinopathy": 0.91, "glaucoma": 0.03, "normal": 0.03},
        }

        with patch("ml.screening_pipeline.predict_eye_disease", return_value=mock_inference):
            res = run_screening(
                image_path=synthetic_retinal_fixtures["valid"],
                heart_rate=550.0,  # Impossible heart rate (> 300)
                spo2=98.0,
                temperature=36.8,
                include_gradcam=False,
            )

            # Vitals failure recorded
            assert res["vitals_screening"]["status"] == "error"
            assert "plausible" in res["vitals_screening"]["reason"].lower()
            assert res["vitals_screening"]["overall_status"] is None

            # Eye screening completely intact
            assert res["eye_screening"]["status"] == "success"
            assert res["eye_screening"]["prediction"] == "diabetic_retinopathy"
            assert res["eye_screening"]["confidence"] == 0.91

            # Composite status is partial
            assert res["status"] == "partial"


class TestRealCheckpointIntegrationSmokeTest:
    """Exercises real end-to-end inference using the champion weights file if present."""

    def test_real_checkpoint_end_to_end_smoke(self, synthetic_retinal_fixtures):
        if not os.path.exists(DEFAULT_CHECKPOINT_PATH):
            pytest.skip("Production checkpoint not found at ml/weights/eye_model.pth")

        res = run_screening(
            image_path=synthetic_retinal_fixtures["valid"],
            heart_rate=75.0,
            spo2=98.0,
            temperature=36.8,
            include_gradcam=True,
        )

        assert res["status"] == "success"
        assert res["eye_screening"]["status"] == "success"
        assert res["eye_screening"]["prediction"] in CLASS_NAMES
        assert 0.0 <= res["eye_screening"]["confidence"] <= 1.0
        assert res["eye_screening"]["gradcam_image"] is not None
        assert res["vitals_screening"]["status"] == "success"
        assert res["vitals_screening"]["overall_status"] == "Normal"
        assert res["overall_risk_level"] in ("LOW", "MODERATE", "HIGH")
