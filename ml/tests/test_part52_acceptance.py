"""
Part 52 — Final ML Pipeline Acceptance Test Suite.

Performs rigorous acceptance testing of the complete standalone multimodal health-screening
ML pipeline across all 10 mandated operational scenarios and verifies output correctness invariants.

Mandated Test Cases:
1. Valid image and valid vital signs.
2. Blurry or poor-quality image (quality gate rejection).
3. Missing image file (status: "error").
4. Corrupt image file (status: "rejected").
5. Missing or invalid model checkpoint (status: "error").
6. Invalid heart rate, SpO2, or temperature (out of range/bad type).
7. Eye-model inference failure (e.g. injected forward exception).
8. Grad-CAM generation failure (safe isolation).
9. Successful eye analysis with vital-sign validation failure (partial status).
10. Successful execution of both independent analyses (success status).

Output Correctness Invariants:
- Output structure invariants (all required keys present, JSON serializable).
- Class probabilities sum to 1.0 +- 1e-4.
- Argmax matches predicted_class.
- Quality gate failure halts eye inference and Grad-CAM early.
- Grad-CAM failure does not invalidate prediction.
- Modality isolation (vital signs do not leak into eye predictions and vice versa).
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
import torch

from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES, load_split_manifest
from ml.eye_model import (
    DEFAULT_CHECKPOINT_PATH,
    build_densenet201_model,
    clear_model_cache,
    get_cached_model,
    predict_eye_disease,
)
from ml.physio_model import analyze_vitals
from ml.quality_gate import check_image_quality
from ml.screening_pipeline import (
    SYSTEM_SCREENING_DISCLAIMER,
    main as cli_main,
    run_screening,
)

MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "dataset", "split_manifest.json"
)


@pytest.fixture
def test_fixtures(tmp_path):
    """Generates synthetic valid, degraded, corrupt, and checkpoint fixtures."""
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
    cv2.imwrite(valid_path, noisy)
    fixtures["valid"] = valid_path

    # 2. Blurry Image (Fails blur threshold)
    blurry_path = str(tmp_path / "blurry_fundus.png")
    blurred = cv2.GaussianBlur(canvas, (45, 45), sigmaX=20)
    cv2.imwrite(blurry_path, blurred)
    fixtures["blurry"] = blurry_path

    # 3. Corrupt Binary (Not a decodable image)
    corrupt_path = str(tmp_path / "corrupt.png")
    with open(corrupt_path, "wb") as f:
        f.write(b"CORRUPT_NON_IMAGE_DATA_PAYLOAD_ABC123")
    fixtures["corrupt"] = corrupt_path

    # 4. Valid Test Checkpoint
    ckpt_path = str(tmp_path / "test_checkpoint.pth")
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

    # 5. Invalid / Incompatible Architecture Checkpoint
    bad_arch_ckpt = str(tmp_path / "bad_arch_checkpoint.pth")
    torch.save(
        {
            "epoch": 1,
            "architecture": "ResNet50",  # Incompatible architecture
            "classes": CLASS_NAMES,
            "state_dict": {},
        },
        bad_arch_ckpt,
    )
    fixtures["bad_arch_checkpoint"] = bad_arch_ckpt

    return fixtures


class TestPart52TenRealisticScenarios:
    """Explicitly verifies the 10 mandated operational scenarios."""

    @pytest.fixture(autouse=True)
    def clean_cache(self):
        clear_model_cache()
        yield
        clear_model_cache()

    # Scenario 1: Valid image and valid vital signs
    def test_scenario_01_valid_image_and_valid_vitals(self, test_fixtures):
        """Scenario 1: Valid image and valid vitals succeed end-to-end."""
        res = run_screening(
            image_path=test_fixtures["valid"],
            heart_rate=72.0,
            spo2=98.0,
            temperature=36.8,
            weights_path=test_fixtures["checkpoint"],
            device="cpu",
            include_gradcam=True,
        )

        assert res["status"] == "success"
        assert res["eye_screening"]["status"] == "success"
        assert res["eye_screening"]["prediction"] in CLASS_NAMES
        assert 0.0 <= res["eye_screening"]["confidence"] <= 1.0
        assert res["eye_screening"]["gradcam_image"] is not None
        assert res["vitals_screening"]["status"] == "success"
        assert res["vitals_screening"]["overall_status"] == "Normal"
        assert res["vitals_screening"]["risk_level"] == "LOW"
        assert res["overall_risk_level"] in ("LOW", "MODERATE", "HIGH")

    # Scenario 2: Blurry or poor-quality image (quality gate rejection)
    def test_scenario_02_blurry_image_quality_gate_rejection(self, test_fixtures):
        """Scenario 2: Blurry image is rejected early without calling neural network inference."""
        with patch("ml.screening_pipeline.predict_eye_disease") as mock_predict:
            res = run_screening(
                image_path=test_fixtures["blurry"],
                heart_rate=70.0,
                spo2=98.0,
                temperature=36.6,
                weights_path=test_fixtures["checkpoint"],
                device="cpu",
            )
            mock_predict.assert_not_called()

        assert res["status"] == "partial"
        assert res["eye_screening"]["status"] == "rejected"
        assert "blur" in res["eye_screening"]["reason"].lower()
        assert res["eye_screening"]["prediction"] is None
        assert res["eye_screening"]["confidence"] is None
        assert res["eye_screening"]["gradcam_image"] is None
        assert res["eye_screening"]["quality_checks"]["overall_pass"] is False
        assert res["vitals_screening"]["status"] == "success"

    # Scenario 3: Missing image file (status: "error")
    def test_scenario_03_missing_image_file(self, test_fixtures):
        """Scenario 3: Missing image file returns status 'error' without fabricating predictions."""
        res = run_screening(
            image_path="nonexistent_retina_image_part52_xyz.png",
            heart_rate=75.0,
            spo2=98.0,
            temperature=36.7,
            weights_path=test_fixtures["checkpoint"],
            device="cpu",
        )

        assert res["eye_screening"]["status"] == "error"
        assert "not found" in res["eye_screening"]["reason"].lower()
        assert res["eye_screening"]["prediction"] is None
        assert res["eye_screening"]["confidence"] is None
        assert res["eye_screening"]["gradcam_image"] is None
        # Vitals succeed independently
        assert res["vitals_screening"]["status"] == "success"
        assert res["status"] == "partial"

    # Scenario 4: Corrupt image file (status: "rejected")
    def test_scenario_04_corrupt_image_file(self, test_fixtures):
        """Scenario 4: Corrupt image file is rejected by quality gate without crashing."""
        with patch("ml.screening_pipeline.predict_eye_disease") as mock_predict:
            res = run_screening(
                image_path=test_fixtures["corrupt"],
                heart_rate=72.0,
                spo2=98.0,
                temperature=36.8,
                weights_path=test_fixtures["checkpoint"],
                device="cpu",
            )
            mock_predict.assert_not_called()

        assert res["eye_screening"]["status"] == "rejected"
        assert "corrupt" in res["eye_screening"]["reason"].lower() or "decode" in res["eye_screening"]["reason"].lower()
        assert res["eye_screening"]["prediction"] is None
        assert res["eye_screening"]["gradcam_image"] is None
        assert res["vitals_screening"]["status"] == "success"
        assert res["status"] == "partial"

    # Scenario 5: Missing or invalid model checkpoint (status: "error")
    def test_scenario_05_missing_or_invalid_model_checkpoint(self, test_fixtures):
        """Scenario 5: Missing or incompatible checkpoint results in status 'error' with no random fallback."""
        # 5a. Missing checkpoint file
        res_missing = run_screening(
            image_path=test_fixtures["valid"],
            heart_rate=72.0,
            spo2=98.0,
            temperature=36.8,
            weights_path="nonexistent_checkpoint_weights_52.pth",
            device="cpu",
        )
        assert res_missing["eye_screening"]["status"] == "error"
        assert "unavailable" in res_missing["eye_screening"]["reason"].lower()
        assert res_missing["eye_screening"]["prediction"] is None
        assert res_missing["vitals_screening"]["status"] == "success"

        # 5b. Incompatible architecture checkpoint
        res_bad = run_screening(
            image_path=test_fixtures["valid"],
            heart_rate=72.0,
            spo2=98.0,
            temperature=36.8,
            weights_path=test_fixtures["bad_arch_checkpoint"],
            device="cpu",
        )
        assert res_bad["eye_screening"]["status"] == "error"
        assert "invalid" in res_bad["eye_screening"]["reason"].lower()
        assert res_bad["eye_screening"]["prediction"] is None
        assert res_bad["vitals_screening"]["status"] == "success"

    # Scenario 6: Invalid heart rate, SpO2, or temperature (out of range/bad type)
    def test_scenario_06_invalid_vitals_values_or_types(self, test_fixtures):
        """Scenario 6: Invalid or unphysiological vitals result in vitals error while eye is unaffected."""
        # 6a. Impossible heart rate (550 BPM)
        res_hr = run_screening(
            image_path=test_fixtures["valid"],
            heart_rate=550.0,
            spo2=98.0,
            temperature=36.8,
            weights_path=test_fixtures["checkpoint"],
            device="cpu",
        )
        assert res_hr["vitals_screening"]["status"] == "error"
        assert "plausible" in res_hr["vitals_screening"]["reason"].lower()
        assert res_hr["eye_screening"]["status"] == "success"
        assert res_hr["status"] == "partial"

        # 6b. Impossible SpO2 (115%)
        res_spo2 = run_screening(
            image_path=test_fixtures["valid"],
            heart_rate=72.0,
            spo2=115.0,
            temperature=36.8,
            weights_path=test_fixtures["checkpoint"],
            device="cpu",
        )
        assert res_spo2["vitals_screening"]["status"] == "error"
        assert res_spo2["eye_screening"]["status"] == "success"

        # 6c. Impossible temperature (55.0 °C > 45.0 °C max plausible)
        res_temp = run_screening(
            image_path=test_fixtures["valid"],
            heart_rate=72.0,
            spo2=98.0,
            temperature=55.0,
            weights_path=test_fixtures["checkpoint"],
            device="cpu",
        )
        assert res_temp["vitals_screening"]["status"] == "error"
        assert "plausible" in res_temp["vitals_screening"]["reason"].lower()
        assert res_temp["eye_screening"]["status"] == "success"

        # 6d. Boolean injection type rejection
        res_bool = run_screening(
            image_path=test_fixtures["valid"],
            heart_rate=True,  # type: ignore
            spo2=98.0,
            temperature=36.8,
            weights_path=test_fixtures["checkpoint"],
            device="cpu",
        )
        assert res_bool["vitals_screening"]["status"] == "error"
        assert "bool" in res_bool["vitals_screening"]["reason"].lower()
        assert res_bool["eye_screening"]["status"] == "success"

    # Scenario 7: Eye-model inference failure (e.g. injected forward exception)
    def test_scenario_07_eye_model_inference_failure(self, test_fixtures):
        """Scenario 7: Inference exception is captured safely as error while vitals complete."""
        with patch("ml.screening_pipeline.predict_eye_disease", side_effect=RuntimeError("CUDA Forward Pass Crash")):
            res = run_screening(
                image_path=test_fixtures["valid"],
                heart_rate=72.0,
                spo2=98.0,
                temperature=36.8,
                weights_path=test_fixtures["checkpoint"],
                device="cpu",
            )

        assert res["eye_screening"]["status"] == "error"
        assert "CUDA Forward Pass Crash" in res["eye_screening"]["reason"]
        assert res["eye_screening"]["prediction"] is None
        assert res["eye_screening"]["gradcam_image"] is None
        assert res["vitals_screening"]["status"] == "success"
        assert res["status"] == "partial"

    # Scenario 8: Grad-CAM generation failure (safe isolation)
    def test_scenario_08_gradcam_generation_failure_safe_isolation(self, test_fixtures):
        """Scenario 8: Grad-CAM failure does not corrupt or invalidate an otherwise successful prediction."""
        with patch("ml.screening_pipeline.generate_gradcam", side_effect=RuntimeError("Grad-CAM Hook Fault")):
            res = run_screening(
                image_path=test_fixtures["valid"],
                heart_rate=72.0,
                spo2=98.0,
                temperature=36.8,
                weights_path=test_fixtures["checkpoint"],
                device="cpu",
                include_gradcam=True,
            )

        assert res["status"] == "success"
        assert res["eye_screening"]["status"] == "success"
        assert res["eye_screening"]["prediction"] in CLASS_NAMES
        assert res["eye_screening"]["gradcam_image"] is None
        assert "gradcam_error" in res["eye_screening"]
        assert "Grad-CAM Hook Fault" in res["eye_screening"]["gradcam_error"]
        assert res["vitals_screening"]["status"] == "success"

    # Scenario 9: Successful eye analysis with vital-sign validation failure (partial status)
    def test_scenario_09_successful_eye_with_vitals_failure_is_partial(self, test_fixtures):
        """Scenario 9: Valid eye inference + vital validation failure yields overall 'partial' status."""
        res = run_screening(
            image_path=test_fixtures["valid"],
            heart_rate=350.0,  # Invalid
            spo2=98.0,
            temperature=36.8,
            weights_path=test_fixtures["checkpoint"],
            device="cpu",
        )

        assert res["status"] == "partial"
        assert res["eye_screening"]["status"] == "success"
        assert res["eye_screening"]["prediction"] in CLASS_NAMES
        assert res["vitals_screening"]["status"] == "error"
        assert res["vitals_screening"]["overall_status"] is None

    # Scenario 10: Successful execution of both independent analyses (success status)
    def test_scenario_10_successful_execution_of_both_analyses(self, test_fixtures):
        """Scenario 10: Both independent modalities succeed, yielding overall 'success' status."""
        res = run_screening(
            image_path=test_fixtures["valid"],
            heart_rate=72.0,
            spo2=98.0,
            temperature=36.8,
            weights_path=test_fixtures["checkpoint"],
            device="cpu",
            include_gradcam=True,
        )

        assert res["status"] == "success"
        assert res["eye_screening"]["status"] == "success"
        assert res["vitals_screening"]["status"] == "success"
        assert res["overall_risk_level"] in ("LOW", "MODERATE", "HIGH")
        assert res["duration_ms"] > 0.0


class TestPart52OutputCorrectnessInvariants:
    """Verifies output invariants and contract guarantees."""

    def test_output_structure_invariants_and_json_serialization(self, test_fixtures):
        """Output dictionary must contain all required top-level keys and serialize cleanly to JSON."""
        res = run_screening(
            image_path=test_fixtures["valid"],
            heart_rate=74.0,
            spo2=99.0,
            temperature=36.7,
            weights_path=test_fixtures["checkpoint"],
            device="cpu",
        )

        required_keys = [
            "status",
            "screening_id",
            "timestamp",
            "duration_ms",
            "timings_ms",
            "eye_screening",
            "vitals_screening",
            "overall_risk_level",
            "disclaimer",
        ]
        for key in required_keys:
            assert key in res, f"Missing required top-level key: {key}"

        # JSON serialization roundtrip
        serialized = json.dumps(res)
        deserialized = json.loads(serialized)
        assert deserialized["status"] == res["status"]
        assert deserialized["overall_risk_level"] == res["overall_risk_level"]

    def test_class_probabilities_sum_to_one_and_match_argmax(self, test_fixtures):
        """Class probabilities must sum to 1.0 +- 1e-4 and argmax must equal prediction."""
        res = run_screening(
            image_path=test_fixtures["valid"],
            heart_rate=70.0,
            spo2=98.0,
            temperature=36.7,
            weights_path=test_fixtures["checkpoint"],
            device="cpu",
            include_gradcam=False,
        )

        eye = res["eye_screening"]
        assert eye["status"] == "success"
        probs = eye["all_probabilities"]
        assert isinstance(probs, dict)
        assert len(probs) == NUM_CLASSES
        assert set(probs.keys()) == set(CLASS_NAMES)

        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {prob_sum}"

        # Argmax check
        max_class = max(probs, key=probs.get)
        assert eye["prediction"] == max_class
        assert abs(eye["confidence"] - probs[max_class]) < 1e-4
        assert eye["class_index"] == CLASS_NAMES.index(max_class)

    def test_quality_gate_failure_halts_eye_inference_and_gradcam(self, test_fixtures):
        """Quality gate failure strictly halts preprocessing, inference, and Grad-CAM."""
        with patch("ml.screening_pipeline.preprocess_image") as mock_prep, \
             patch("ml.screening_pipeline.predict_eye_disease") as mock_inf, \
             patch("ml.screening_pipeline.generate_gradcam") as mock_gcam:
            res = run_screening(
                image_path=test_fixtures["blurry"],
                heart_rate=72.0,
                spo2=98.0,
                temperature=36.8,
                weights_path=test_fixtures["checkpoint"],
                device="cpu",
            )
            mock_prep.assert_not_called()
            mock_inf.assert_not_called()
            mock_gcam.assert_not_called()

        assert res["eye_screening"]["status"] == "rejected"
        assert res["eye_screening"]["prediction"] is None
        assert res["eye_screening"]["confidence"] is None
        assert res["eye_screening"]["all_probabilities"] is None
        assert res["eye_screening"]["gradcam_image"] is None

    def test_vital_signs_do_not_leak_into_eye_predictions(self, test_fixtures):
        """Vital sign variation must not leak into eye disease inference probabilities."""
        img = test_fixtures["valid"]
        ckpt = test_fixtures["checkpoint"]

        vitals_list = [
            (60.0, 99.0, 36.5),   # Baseline normal
            (150.0, 85.0, 39.5),  # Extreme critical
            (45.0, 92.0, 35.0),   # Bradycardia/hypothermia
        ]

        results = []
        for hr, spo2, temp in vitals_list:
            res = run_screening(
                image_path=img,
                heart_rate=hr,
                spo2=spo2,
                temperature=temp,
                weights_path=ckpt,
                device="cpu",
                include_gradcam=False,
            )
            results.append(res["eye_screening"])

        base = results[0]
        for idx, other in enumerate(results[1:], start=1):
            assert other["prediction"] == base["prediction"]
            assert abs(other["confidence"] - base["confidence"]) < 1e-7
            for c in CLASS_NAMES:
                assert abs(other["all_probabilities"][c] - base["all_probabilities"][c]) < 1e-7

    def test_eye_image_does_not_leak_into_vitals_analysis(self, test_fixtures):
        """Eye image status (valid, blurry, corrupt, missing) must not leak into vitals analysis."""
        hr, spo2, temp = 76.0, 98.0, 36.8
        ckpt = test_fixtures["checkpoint"]

        # Run across different eye input conditions
        r_valid = run_screening(image_path=test_fixtures["valid"], heart_rate=hr, spo2=spo2, temperature=temp, weights_path=ckpt, device="cpu", include_gradcam=False)
        r_blurry = run_screening(image_path=test_fixtures["blurry"], heart_rate=hr, spo2=spo2, temperature=temp, weights_path=ckpt, device="cpu", include_gradcam=False)
        r_corrupt = run_screening(image_path=test_fixtures["corrupt"], heart_rate=hr, spo2=spo2, temperature=temp, weights_path=ckpt, device="cpu", include_gradcam=False)
        r_none = run_screening(image_path=None, heart_rate=hr, spo2=spo2, temperature=temp, weights_path=ckpt, device="cpu", include_gradcam=False)

        v_valid = r_valid["vitals_screening"]
        for other in (r_blurry["vitals_screening"], r_corrupt["vitals_screening"], r_none["vitals_screening"]):
            assert other["status"] == v_valid["status"] == "success"
            assert other["overall_status"] == v_valid["overall_status"] == "Normal"
            assert other["risk_level"] == v_valid["risk_level"] == "LOW"
            assert other["is_abnormal"] == v_valid["is_abnormal"] is False
            assert other["summary"] == v_valid["summary"]


class TestPart52RealArtifactsIntegration:
    """End-to-end integration tests using real production checkpoint and test images."""

    def test_real_champion_checkpoint_and_held_out_test_image(self):
        """Runs complete screening pipeline on real trained checkpoint and real held-out image."""
        if not os.path.exists(DEFAULT_CHECKPOINT_PATH):
            pytest.skip(f"Production checkpoint not found at {DEFAULT_CHECKPOINT_PATH}")

        if not os.path.exists(MANIFEST_PATH):
            pytest.skip(f"Split manifest not found at {MANIFEST_PATH}")

        manifest = load_split_manifest(MANIFEST_PATH)
        test_images = manifest["splits"]["test"]

        # Find first eligible quality-passing image from test split
        passing_image = None
        for item in test_images:
            fp = item["filepath"]
            if os.path.exists(fp) and check_image_quality(fp)["overall_pass"]:
                passing_image = fp
                break

        assert passing_image is not None, "No quality-passing test image found in split manifest"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        res = run_screening(
            image_path=passing_image,
            heart_rate=72.0,
            spo2=98.0,
            temperature=36.8,
            weights_path=DEFAULT_CHECKPOINT_PATH,
            device=device,
            include_gradcam=True,
        )

        assert res["status"] == "success"
        eye = res["eye_screening"]
        assert eye["status"] == "success"
        assert eye["prediction"] in CLASS_NAMES
        assert eye["confidence"] >= 0.25
        assert eye["gradcam_image"] is not None
        assert len(eye["gradcam_image"]) > 1000

        vitals = res["vitals_screening"]
        assert vitals["status"] == "success"
        assert vitals["overall_status"] == "Normal"
        assert vitals["risk_level"] == "LOW"

        assert SYSTEM_SCREENING_DISCLAIMER in res["disclaimer"]

    def test_cli_execution_with_real_checkpoint(self, tmp_path):
        """Verifies CLI entrypoint produces valid JSON file and zero exit code with real checkpoint."""
        if not os.path.exists(DEFAULT_CHECKPOINT_PATH) or not os.path.exists(MANIFEST_PATH):
            pytest.skip("Production artifacts not found")

        manifest = load_split_manifest(MANIFEST_PATH)
        passing_image = None
        for item in manifest["splits"]["test"]:
            fp = item["filepath"]
            if os.path.exists(fp) and check_image_quality(fp)["overall_pass"]:
                passing_image = fp
                break

        assert passing_image is not None

        out_file = str(tmp_path / "part52_cli_result.json")
        exit_code = cli_main([
            "--image", passing_image,
            "--hr", "75",
            "--spo2", "99",
            "--temp", "36.7",
            "--weights", DEFAULT_CHECKPOINT_PATH,
            "--output", out_file,
        ])

        assert exit_code == 0
        assert os.path.exists(out_file)
        with open(out_file, "r") as f:
            data = json.load(f)
        assert data["status"] == "success"
        assert data["eye_screening"]["prediction"] in CLASS_NAMES
        assert data["vitals_screening"]["status"] == "success"
