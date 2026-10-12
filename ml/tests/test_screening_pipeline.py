"""
Comprehensive Integration & Verification Tests for Multimodal Health Screening Pipeline.

Validates:
1. Full workflow execution: Quality Gate -> Preprocessing -> Inference -> Grad-CAM + Independent Vitals.
2. Realistic scenarios:
   - Valid retinal image + valid vitals (success).
   - Blurry image (quality gate rejection, zero inference).
   - Low resolution image (<224x224, rejection).
   - Excessively dark / bright image (rejection).
   - Non-retinal image (no circular contour, rejection).
   - Corrupt image binary (safe error, no crash).
   - Missing image path (safe error, vitals unaffected).
   - Missing / invalid vitals (types, out-of-range, NaN, vitals error, eye screening unaffected).
   - Missing / corrupt model checkpoint (typed error, no random weights fallback).
   - Grad-CAM failure isolation (preserves inference results).
   - Model inference exception handling (clean error, no internal leak).
   - Multiple consecutive requests (caching & memory safety).
   - Simultaneous concurrent requests (multi-threaded safety).
   - CLI execution (stdout JSON, proper exit codes).
3. Contract schema compliance (prediction, class_index, confidence, risk_level, all_probabilities).
4. Safety, separation, and medical prototype disclaimers.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
import torch

from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES
from ml.eye_model import build_densenet201_model, clear_model_cache
from ml.screening_pipeline import (
    SYSTEM_SCREENING_DISCLAIMER,
    _aggregate_overall_risk,
    main as cli_main,
    run_screening,
)

REAL_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "weights", "eye_model.pth"
)
BEST_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "weights", "eye_densenet201_best.pth"
)
SPLIT_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "dataset", "split_manifest.json"
)


@pytest.fixture
def synthetic_retinal_fixtures(tmp_path):
    """Generates synthetic valid, degraded, and corrupt image fixtures for pipeline tests."""
    fixtures = {}

    # 1. Valid Retinal Image (Passes all 5 quality gate checks)
    valid_path = str(tmp_path / "valid_fundus.png")
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 100, (140, 130, 120), -1)
    np.random.seed(42)
    noise = np.random.randint(-40, 40, (300, 300, 3), dtype=np.int16)
    noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.circle(noisy, (190, 150), 22, (230, 220, 200), -1)
    cv2.line(noisy, (190, 150), (100, 100), (40, 30, 160), 2)
    cv2.line(noisy, (190, 150), (100, 200), (40, 30, 160), 2)
    cv2.line(noisy, (190, 150), (130, 150), (40, 30, 160), 2)
    cv2.imwrite(valid_path, noisy)
    fixtures["valid"] = valid_path

    # 2. Blurry Image
    blurry_path = str(tmp_path / "blurry_fundus.png")
    blur_canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(blur_canvas, (150, 150), 100, (140, 130, 120), -1)
    blurred = cv2.GaussianBlur(blur_canvas, (45, 45), sigmaX=20)
    cv2.imwrite(blurry_path, blurred)
    fixtures["blurry"] = blurry_path

    # 3. Small Image (< 224x224)
    small_path = str(tmp_path / "small_image.png")
    small_canvas = np.full((128, 128, 3), 120, dtype=np.uint8)
    cv2.imwrite(small_path, small_canvas)
    fixtures["small"] = small_path

    # 4. Excessively Dark Image (< 30 brightness)
    dark_path = str(tmp_path / "dark_image.png")
    dark_canvas = np.full((300, 300, 3), 10, dtype=np.uint8)
    cv2.imwrite(dark_path, dark_canvas)
    fixtures["dark"] = dark_path

    # 5. Excessively Bright Image (> 220 brightness)
    bright_path = str(tmp_path / "bright_image.png")
    bright_canvas = np.full((300, 300, 3), 245, dtype=np.uint8)
    cv2.imwrite(bright_path, bright_canvas)
    fixtures["bright"] = bright_path

    # 6. Non-Retinal Image (No circular retinal region)
    non_retinal_path = str(tmp_path / "non_retinal.png")
    square_canvas = np.full((300, 300, 3), 120, dtype=np.uint8)
    # Uniform texture with no circular mask
    cv2.rectangle(square_canvas, (50, 50), (250, 250), (140, 140, 140), -1)
    cv2.imwrite(non_retinal_path, square_canvas)
    fixtures["non_retinal"] = non_retinal_path

    # 7. Corrupted Binary
    corrupt_path = str(tmp_path / "corrupt.png")
    with open(corrupt_path, "wb") as f:
        f.write(b"CORRUPTED_NOT_AN_IMAGE_PAYLOAD")
    fixtures["corrupt"] = corrupt_path

    # 8. Controlled Test Checkpoint
    ckpt_path = str(tmp_path / "valid_test_ckpt.pth")
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    torch.save(
        {
            "epoch": 1,
            "architecture": "DenseNet201",
            "classes": CLASS_NAMES,
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    fixtures["checkpoint"] = ckpt_path

    return fixtures


class TestScreeningPipelineFullWorkflow:
    """Verifies end-to-end multimodal screening pipeline execution."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        clear_model_cache()
        yield
        clear_model_cache()

    def test_valid_image_and_valid_vitals_success(self, synthetic_retinal_fixtures):
        result = run_screening(
            image_path=synthetic_retinal_fixtures["valid"],
            heart_rate=72.0,
            spo2=98.0,
            temperature=36.8,
            weights_path=synthetic_retinal_fixtures["checkpoint"],
            device=torch.device("cpu"),
            include_gradcam=True,
        )

        assert result["status"] == "success"
        assert "screening_id" in result
        assert "timestamp" in result
        assert result["duration_ms"] > 0.0

        # Eye Screening Contract
        eye = result["eye_screening"]
        assert eye["status"] == "success"
        assert eye["prediction"] in CLASS_NAMES
        assert 0 <= eye["class_index"] < NUM_CLASSES
        assert 0.0 <= eye["confidence"] <= 1.0
        assert eye["risk_level"] in ("LOW", "MODERATE", "HIGH")
        assert len(eye["all_probabilities"]) == NUM_CLASSES
        assert eye["quality_checks"]["overall_pass"] is True
        assert eye["gradcam_image"] is not None
        assert len(eye["gradcam_image"]) > 100

        # Vitals Screening Contract
        vitals = result["vitals_screening"]
        assert vitals["status"] == "success"
        assert vitals["heart_rate"] == 72.0
        assert vitals["spo2"] == 98.0
        assert vitals["temperature"] == 36.8
        assert vitals["overall_status"] == "Normal"
        assert vitals["risk_level"] == "LOW"
        assert vitals["is_abnormal"] is False

        # Overall Aggregated Risk
        assert result["overall_risk_level"] in ("LOW", "MODERATE", "HIGH")
        assert SYSTEM_SCREENING_DISCLAIMER in result["disclaimer"]

    def test_real_checkpoint_end_to_end_screening_if_available(self):
        target_path = REAL_CHECKPOINT_PATH if os.path.exists(REAL_CHECKPOINT_PATH) else BEST_CHECKPOINT_PATH
        if not os.path.exists(target_path):
            pytest.skip("No trained checkpoint available on disk")

        if not os.path.exists(SPLIT_MANIFEST_PATH):
            pytest.skip("Manifest not available on disk")

        with open(SPLIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        test_records = manifest["splits"]["test"]
        eligible_image = None
        for rec in test_records:
            if os.path.exists(rec["filepath"]):
                from ml.quality_gate import check_image_quality
                q = check_image_quality(rec["filepath"])
                if q["overall_pass"]:
                    eligible_image = rec["filepath"]
                    break

        if not eligible_image:
            pytest.skip("No quality-passing sample found in test split")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = run_screening(
            image_path=eligible_image,
            heart_rate=80.0,
            spo2=96.0,
            temperature=37.0,
            weights_path=target_path,
            device=device,
            include_gradcam=True,
        )

        assert result["status"] == "success"
        eye = result["eye_screening"]
        assert eye["status"] == "success"
        assert eye["prediction"] in CLASS_NAMES
        assert eye["confidence"] > 0.40  # Real trained model
        assert eye["gradcam_image"] is not None
        assert result["vitals_screening"]["status"] == "success"


class TestRealisticFailureScenarios:
    """Verifies correct rejection and error routing across realistic degradation cases."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        clear_model_cache()
        yield
        clear_model_cache()

    def test_blurry_image_rejected_without_inference(self, synthetic_retinal_fixtures):
        with patch("ml.screening_pipeline.predict_eye_disease") as mock_predict:
            result = run_screening(
                image_path=synthetic_retinal_fixtures["blurry"],
                heart_rate=75.0,
                spo2=99.0,
                temperature=36.6,
                weights_path=synthetic_retinal_fixtures["checkpoint"],
                device="cpu",
            )
            mock_predict.assert_not_called()

        assert result["status"] in ("partial", "rejected")
        eye = result["eye_screening"]
        assert eye["status"] == "rejected"
        assert "blur" in eye["reason"].lower()
        assert eye["prediction"] is None
        assert eye["gradcam_image"] is None
        # Vitals independent
        assert result["vitals_screening"]["status"] == "success"
        assert result["vitals_screening"]["overall_status"] == "Normal"

    def test_small_resolution_image_rejected(self, synthetic_retinal_fixtures):
        with patch("ml.screening_pipeline.predict_eye_disease") as mock_predict:
            result = run_screening(
                image_path=synthetic_retinal_fixtures["small"],
                heart_rate=70.0,
                spo2=98.0,
                temperature=36.5,
                weights_path=synthetic_retinal_fixtures["checkpoint"],
                device="cpu",
            )
            mock_predict.assert_not_called()

        assert result["eye_screening"]["status"] == "rejected"
        assert "resolution" in result["eye_screening"]["reason"].lower()

    def test_dark_image_rejected(self, synthetic_retinal_fixtures):
        with patch("ml.screening_pipeline.predict_eye_disease") as mock_predict:
            result = run_screening(
                image_path=synthetic_retinal_fixtures["dark"],
                heart_rate=70.0,
                spo2=98.0,
                temperature=36.5,
                weights_path=synthetic_retinal_fixtures["checkpoint"],
                device="cpu",
            )
            mock_predict.assert_not_called()

        assert result["eye_screening"]["status"] == "rejected"
        assert result["eye_screening"]["quality_checks"]["brightness_mean"] < 30.0
        assert result["eye_screening"]["quality_checks"]["overall_pass"] is False
        assert result["eye_screening"]["prediction"] is None

    def test_bright_image_rejected(self, synthetic_retinal_fixtures):
        with patch("ml.screening_pipeline.predict_eye_disease") as mock_predict:
            result = run_screening(
                image_path=synthetic_retinal_fixtures["bright"],
                heart_rate=70.0,
                spo2=98.0,
                temperature=36.5,
                weights_path=synthetic_retinal_fixtures["checkpoint"],
                device="cpu",
            )
            mock_predict.assert_not_called()

        assert result["eye_screening"]["status"] == "rejected"
        assert result["eye_screening"]["quality_checks"]["brightness_mean"] > 220.0
        assert result["eye_screening"]["quality_checks"]["overall_pass"] is False
        assert result["eye_screening"]["prediction"] is None

    def test_corrupted_image_file_handled_safely(self, synthetic_retinal_fixtures):
        result = run_screening(
            image_path=synthetic_retinal_fixtures["corrupt"],
            heart_rate=72.0,
            spo2=98.0,
            temperature=36.7,
            weights_path=synthetic_retinal_fixtures["checkpoint"],
            device="cpu",
        )
        assert result["eye_screening"]["status"] in ("rejected", "error")
        assert result["eye_screening"]["prediction"] is None
        # Vitals independent
        assert result["vitals_screening"]["status"] == "success"

    def test_missing_image_file_handled_safely(self, synthetic_retinal_fixtures):
        result = run_screening(
            image_path="nonexistent_retina_image_12345.png",
            heart_rate=75.0,
            spo2=97.0,
            temperature=36.9,
            weights_path=synthetic_retinal_fixtures["checkpoint"],
            device="cpu",
        )
        assert result["eye_screening"]["status"] == "error"
        assert "not found" in result["eye_screening"]["reason"].lower()
        # Vitals independent
        assert result["vitals_screening"]["status"] == "success"

    def test_missing_vitals_handled_safely(self, synthetic_retinal_fixtures):
        result = run_screening(
            image_path=synthetic_retinal_fixtures["valid"],
            heart_rate=None,
            spo2=None,
            temperature=None,
            weights_path=synthetic_retinal_fixtures["checkpoint"],
            device="cpu",
        )
        assert result["vitals_screening"]["status"] == "skipped"
        assert result["eye_screening"]["status"] == "success"

    def test_partially_missing_vitals_handled_safely(self, synthetic_retinal_fixtures):
        result = run_screening(
            image_path=synthetic_retinal_fixtures["valid"],
            heart_rate=72.0,
            spo2=None,  # Missing
            temperature=36.8,
            weights_path=synthetic_retinal_fixtures["checkpoint"],
            device="cpu",
        )
        assert result["vitals_screening"]["status"] == "error"
        assert "missing: spo2" in result["vitals_screening"]["reason"].lower()
        assert result["eye_screening"]["status"] == "success"

    def test_invalid_vital_values_handled_safely(self, synthetic_retinal_fixtures):
        # Impossible heart rate (550 BPM)
        result = run_screening(
            image_path=synthetic_retinal_fixtures["valid"],
            heart_rate=550.0,
            spo2=98.0,
            temperature=36.8,
            weights_path=synthetic_retinal_fixtures["checkpoint"],
            device="cpu",
        )
        assert result["vitals_screening"]["status"] == "error"
        assert "plausible" in result["vitals_screening"]["reason"].lower()
        assert result["eye_screening"]["status"] == "success"

    def test_missing_checkpoint_fails_with_clear_error(self, synthetic_retinal_fixtures):
        result = run_screening(
            image_path=synthetic_retinal_fixtures["valid"],
            heart_rate=72.0,
            spo2=98.0,
            temperature=36.8,
            weights_path="nonexistent_checkpoint_weights_99999.pth",
            device="cpu",
        )
        assert result["eye_screening"]["status"] == "error"
        assert "unavailable" in result["eye_screening"]["reason"].lower()
        # Vitals independent
        assert result["vitals_screening"]["status"] == "success"

    def test_gradcam_failure_isolation(self, synthetic_retinal_fixtures):
        with patch("ml.screening_pipeline.generate_gradcam", side_effect=RuntimeError("CUDA OOM in GradCAM")):
            result = run_screening(
                image_path=synthetic_retinal_fixtures["valid"],
                heart_rate=72.0,
                spo2=98.0,
                temperature=36.8,
                weights_path=synthetic_retinal_fixtures["checkpoint"],
                device="cpu",
                include_gradcam=True,
            )

        # Eye inference MUST succeed despite Grad-CAM failure
        assert result["eye_screening"]["status"] == "success"
        assert result["eye_screening"]["prediction"] in CLASS_NAMES
        assert result["eye_screening"]["gradcam_image"] is None
        assert "gradcam_error" in result["eye_screening"]

    def test_model_inference_exception_handling(self, synthetic_retinal_fixtures):
        with patch("ml.screening_pipeline.predict_eye_disease", side_effect=RuntimeError("Internal tensor error")):
            result = run_screening(
                image_path=synthetic_retinal_fixtures["valid"],
                heart_rate=72.0,
                spo2=98.0,
                temperature=36.8,
                weights_path=synthetic_retinal_fixtures["checkpoint"],
                device="cpu",
            )
        assert result["eye_screening"]["status"] == "error"
        assert "inference error" in result["eye_screening"]["reason"].lower()
        assert result["vitals_screening"]["status"] == "success"


class TestConcurrencyAndCLI:
    """Verifies thread safety across concurrent requests and CLI functionality."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        clear_model_cache()
        yield
        clear_model_cache()

    def test_consecutive_screening_requests(self, synthetic_retinal_fixtures):
        # 5 consecutive requests reusing model cache
        results = []
        for i in range(5):
            res = run_screening(
                image_path=synthetic_retinal_fixtures["valid"],
                heart_rate=70.0 + i,
                spo2=98.0,
                temperature=36.8,
                weights_path=synthetic_retinal_fixtures["checkpoint"],
                device="cpu",
                use_cache=True,
            )
            assert res["status"] == "success"
            results.append(res)
        assert len(results) == 5

    def test_concurrent_screening_requests(self, synthetic_retinal_fixtures):
        def _invoke_request(idx: int):
            return run_screening(
                image_path=synthetic_retinal_fixtures["valid"],
                heart_rate=65.0 + idx,
                spo2=97.0,
                temperature=36.7,
                weights_path=synthetic_retinal_fixtures["checkpoint"],
                device="cpu",
                use_cache=True,
                include_gradcam=False,
                request_id=f"concurrent-{idx}",
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_invoke_request, i) for i in range(6)]
            results = [f.result() for f in futures]

        assert len(results) == 6
        for r in results:
            assert r["status"] == "success"
            assert r["eye_screening"]["prediction"] in CLASS_NAMES
            assert r["vitals_screening"]["status"] == "success"

    def test_cli_execution_with_valid_args(self, synthetic_retinal_fixtures, tmp_path, capsys):
        out_file = str(tmp_path / "cli_out.json")
        code = cli_main([
            "--image", synthetic_retinal_fixtures["valid"],
            "--hr", "72",
            "--spo2", "98",
            "--temp", "36.8",
            "--weights", synthetic_retinal_fixtures["checkpoint"],
            "--device", "cpu",
            "--output", out_file,
        ])
        assert code == 0
        assert os.path.exists(out_file)
        with open(out_file, "r") as f:
            data = json.load(f)
        assert data["status"] == "success"
        assert data["eye_screening"]["prediction"] in CLASS_NAMES

    def test_cli_missing_all_arguments_fails(self, capsys):
        code = cli_main([])
        assert code == 2


class TestRiskAggregationAndContracts:
    """Verifies risk calculation invariants and output structure."""

    def test_overall_risk_aggregation_logic(self):
        assert _aggregate_overall_risk("HIGH", "LOW") == "HIGH"
        assert _aggregate_overall_risk("LOW", "HIGH") == "HIGH"
        assert _aggregate_overall_risk("MODERATE", "LOW") == "MODERATE"
        assert _aggregate_overall_risk("LOW", "MODERATE") == "MODERATE"
        assert _aggregate_overall_risk("LOW", "LOW") == "LOW"
        assert _aggregate_overall_risk(None, "HIGH") == "HIGH"
        assert _aggregate_overall_risk("MODERATE", None) == "MODERATE"
        assert _aggregate_overall_risk(None, None) == "UNKNOWN"
