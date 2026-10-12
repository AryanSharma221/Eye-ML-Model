"""
Tests for ML/Backend Integration Hardening and API Contract Verification (Part 17).

Validates:
1. Exact backend import paths and signatures across all ML modules.
2. Complete eye-screening pipeline replicating FastAPI /api/v1/predict_eye.
3. Complete vitals pipeline replicating FastAPI /api/v1/submit_vitals.
4. Model lifecycle, checkpoint validation, and safe in-memory caching.
5. Strict 4-class response schema invariants, probability bounds, and JSON serializability.
6. Grad-CAM Base64 PNG decoding, signature integrity, and visual dimensions.
7. Complete failure contracts and domain exception typing across boundary.
8. Real trained checkpoint inference (CPU & CUDA) vs. controlled synthetic weights.
9. Multi-threaded concurrency safety for forward passes and Grad-CAM backpropagation.
10. Simulated FastAPI route handlers demonstrating zero unhandled 500 crashes.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES, load_split_manifest
from ml.exceptions import (
    CheckpointCorruptError,
    CheckpointNotFoundError,
    GradCAMGenerationError,
    InvalidImageInputError,
    ModelInferenceError,
    PhysiologicalTypeValidationError,
    PhysiologicalValidationError,
    QualityGateRejectionError,
    ResourceExhaustionError,
)
from ml.eye_model import (
    DEFAULT_CHECKPOINT_PATH,
    build_densenet201_model,
    clear_model_cache,
    get_cached_model,
    get_model,
    load_eye_model_checkpoint,
    predict_eye_disease,
    screen_eye_image,
)
from ml.gradcam import generate_gradcam
from ml.physio_model import (
    STATUS_ABNORMAL,
    STATUS_HIGHER,
    STATUS_LOWER,
    STATUS_NORMAL,
    analyze_vitals,
)
from ml.preprocessing import preprocess_image
from ml.quality_gate import check_image_quality

# Physical paths to trained checkpoints and split manifest
REAL_CHECKPOINT_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "weights", "eye_model.pth"
)
BEST_CHECKPOINT_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "weights", "eye_densenet201_best.pth"
)
SPLIT_MANIFEST_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "dataset", "split_manifest.json"
)


@pytest.fixture
def test_fundus_fixtures(tmp_path):
    """Generates synthetic valid and invalid images and a test checkpoint for integration testing."""
    valid_path = str(tmp_path / "valid_retina.png")
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 100, (140, 130, 120), -1)

    np.random.seed(42)
    noise = np.random.randint(-40, 40, (300, 300, 3), dtype=np.int16)
    noisy_canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.circle(noisy_canvas, (190, 150), 22, (230, 220, 200), -1)
    cv2.line(noisy_canvas, (190, 150), (100, 100), (40, 30, 160), 2)
    cv2.line(noisy_canvas, (190, 150), (100, 200), (40, 30, 160), 2)
    cv2.line(noisy_canvas, (190, 150), (130, 150), (40, 30, 160), 2)
    cv2.imwrite(valid_path, noisy_canvas)

    # Blurry image failing quality checks
    blurry_path = str(tmp_path / "blurry_image.png")
    blur_canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(blur_canvas, (150, 150), 100, (140, 130, 120), -1)
    blurred = cv2.GaussianBlur(blur_canvas, (35, 35), sigmaX=15)
    cv2.imwrite(blurry_path, blurred)

    # Controlled small model checkpoint with canonical aliases
    ckpt_path = str(tmp_path / "test_checkpoint.pth")
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    payload = {
        "state_dict": model.state_dict(),
        "classes": list(CLASS_NAMES),
        "epoch": 1,
        "best_acc": 0.88,
        "val_accuracy": 0.88,
        "architecture": "DenseNet201",
    }
    torch.save(payload, ckpt_path)

    return {
        "valid_img": valid_path,
        "blurry_img": blurry_path,
        "checkpoint": ckpt_path,
    }


class TestBackendImportPaths:
    """Verifies that the backend can import all required components directly."""

    def test_direct_submodule_imports(self):
        from ml.quality_gate import check_image_quality as ciq
        from ml.preprocessing import preprocess_image as ppi
        from ml.eye_model import predict_eye_disease as ped, screen_eye_image as sei
        from ml.gradcam import generate_gradcam as gg
        from ml.physio_model import analyze_vitals as av

        assert callable(ciq)
        assert callable(ppi)
        assert callable(ped)
        assert callable(sei)
        assert callable(gg)
        assert callable(av)

    def test_root_package_exports(self):
        import ml
        assert hasattr(ml, "check_image_quality")
        assert hasattr(ml, "preprocess_image")
        assert hasattr(ml, "predict_eye_disease")
        assert hasattr(ml, "screen_eye_image")
        assert hasattr(ml, "generate_gradcam")
        assert hasattr(ml, "analyze_vitals")
        assert hasattr(ml, "CLASS_NAMES")


class TestBackendEyeScreeningEndpointWorkflow:
    """Simulates Member 2's /api/v1/predict_eye route step-by-step and as a composite."""

    def test_step_by_step_backend_pipeline_on_valid_image(self, test_fundus_fixtures):
        valid_img = test_fundus_fixtures["valid_img"]
        ckpt = test_fundus_fixtures["checkpoint"]

        # Step 1: Quality gate check
        q_res = check_image_quality(valid_img)
        assert q_res["overall_pass"] is True
        assert q_res["error_message"] is None

        # Step 2: Preprocessing
        tensor = preprocess_image(valid_img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

        # Step 3: Forward inference
        inf_res = predict_eye_disease(tensor, weights_path=ckpt, device=torch.device("cpu"))
        assert inf_res["prediction"] in CLASS_NAMES
        assert 0 <= inf_res["class_index"] < 4
        assert 0.0 <= inf_res["confidence"] <= 1.0
        assert inf_res["risk_level"] in ["LOW", "MODERATE", "HIGH"]
        assert len(inf_res["all_probabilities"]) == 4

        # Step 4: Grad-CAM explainability
        cam_b64 = generate_gradcam(
            valid_img,
            predicted_class_idx=inf_res["class_index"],
            weights_path=ckpt,
            device=torch.device("cpu"),
        )
        assert isinstance(cam_b64, str)
        assert len(cam_b64) > 100

    def test_composite_screen_eye_image_success_contract(self, test_fundus_fixtures):
        valid_img = test_fundus_fixtures["valid_img"]
        ckpt = test_fundus_fixtures["checkpoint"]

        res = screen_eye_image(
            valid_img,
            weights_path=ckpt,
            device=torch.device("cpu"),
            include_gradcam=True,
        )

        assert res["status"] == "success"
        assert "prediction" in res
        assert "class_index" in res
        assert "confidence" in res
        assert "risk_level" in res
        assert "all_probabilities" in res
        assert "quality_checks" in res
        assert "gradcam_image" in res
        # JSON serializability check
        serialized = json.dumps(res)
        assert serialized is not None

    def test_composite_screen_eye_image_without_gradcam(self, test_fundus_fixtures):
        valid_img = test_fundus_fixtures["valid_img"]
        ckpt = test_fundus_fixtures["checkpoint"]

        res = screen_eye_image(
            valid_img,
            weights_path=ckpt,
            device=torch.device("cpu"),
            include_gradcam=False,
        )

        assert res["status"] == "success"
        assert "prediction" in res
        assert "gradcam_image" not in res

    def test_composite_screen_eye_image_rejection_halts_inference(self, test_fundus_fixtures):
        blurry_img = test_fundus_fixtures["blurry_img"]
        ckpt = test_fundus_fixtures["checkpoint"]

        res = screen_eye_image(blurry_img, weights_path=ckpt, device=torch.device("cpu"))
        assert res["status"] == "rejected"
        assert "reason" in res
        assert "quality_checks" in res
        # Guarantees that no prediction or gradcam is produced for rejected images
        assert "prediction" not in res
        assert "gradcam_image" not in res

    def test_predict_eye_disease_validate_quality_flag(self, test_fundus_fixtures):
        blurry_img = test_fundus_fixtures["blurry_img"]
        ckpt = test_fundus_fixtures["checkpoint"]

        # validate_quality=True routes through quality gate and rejects without crashing
        res = predict_eye_disease(blurry_img, weights_path=ckpt, validate_quality=True)
        assert res["status"] == "rejected"
        assert "reason" in res


class TestBackendVitalsEndpointWorkflow:
    """Simulates Member 2's /api/v1/submit_vitals route."""

    def test_normal_vitals_submission(self):
        res = analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=36.8)
        assert res["overall_status"] == "Normal"
        assert res["risk_level"] == "LOW"
        assert res["is_abnormal"] is False
        assert res["hr_status"] == "Normal"
        assert res["spo2_status"] == "Normal"
        assert res["temp_status"] == "Normal"
        assert json.dumps(res) is not None

    def test_abnormal_vitals_submission(self):
        res = analyze_vitals(heart_rate=140.0, spo2=87.0, temperature=38.9)
        assert res["overall_status"] == "Abnormal"
        assert res["risk_level"] == "HIGH"
        assert res["is_abnormal"] is True
        assert json.dumps(res) is not None

    def test_boundary_and_moderate_vitals_submission(self):
        # Slightly elevated HR (110 bpm), mild hypoxemia (93% SpO2), normal temp (36.6°C)
        res = analyze_vitals(heart_rate=110.0, spo2=93.0, temperature=36.6)
        assert res["hr_status"] == STATUS_HIGHER
        assert res["spo2_status"] == STATUS_LOWER
        assert res["temp_status"] == STATUS_NORMAL
        assert res["risk_level"] in ["MODERATE", "HIGH"]
        assert res["is_abnormal"] is True

    def test_vitals_submission_error_dictionary_mode(self):
        # When raise_errors=False, returns structured error dictionary rather than raising
        res = analyze_vitals(heart_rate=500.0, spo2=98.0, temperature=36.8, raise_errors=False)
        assert res["status"] == "error"
        assert "error_message" in res
        assert res["overall_status"] == "Invalid"
        assert res["risk_level"] == "UNKNOWN"
        assert json.dumps(res) is not None


class TestModelLifecycleAndCachingReliability:
    """Verifies that model loading, reuse, and failures are robust."""

    def test_cached_model_reuse_identity(self, test_fundus_fixtures):
        ckpt = test_fundus_fixtures["checkpoint"]
        cpu = torch.device("cpu")
        clear_model_cache()

        m1 = get_cached_model(weights_path=ckpt, device=cpu)
        m2 = get_cached_model(weights_path=ckpt, device=cpu)
        assert m1 is m2

    def test_cache_clearing(self, test_fundus_fixtures):
        ckpt = test_fundus_fixtures["checkpoint"]
        cpu = torch.device("cpu")

        m1 = get_cached_model(weights_path=ckpt, device=cpu)
        clear_model_cache()
        m2 = get_cached_model(weights_path=ckpt, device=cpu)
        assert m1 is not m2

    def test_missing_checkpoint_raises_typed_filenotfound(self):
        with pytest.raises(CheckpointNotFoundError):
            predict_eye_disease(torch.zeros(1, 3, 224, 224), weights_path="nonexistent_checkpoint.pth")

    def test_corrupt_checkpoint_raises_typed_ioerror(self, tmp_path):
        bad_path = str(tmp_path / "bad.pth")
        with open(bad_path, "wb") as f:
            f.write(b"CORRUPTED_BIN_NOT_A_VALID_CHECKPOINT")

        with pytest.raises(CheckpointCorruptError):
            predict_eye_disease(torch.zeros(1, 3, 224, 224), weights_path=bad_path)


class TestConcurrencyAndThreadSafety:
    """Verifies thread-safe inference and Grad-CAM execution under concurrent requests."""

    def test_concurrent_screening_requests(self, test_fundus_fixtures):
        valid_img = test_fundus_fixtures["valid_img"]
        ckpt = test_fundus_fixtures["checkpoint"]

        def worker_task(thread_id: int):
            res = screen_eye_image(
                valid_img,
                weights_path=ckpt,
                device=torch.device("cpu"),
                include_gradcam=True,
                use_cache=True,
            )
            assert res["status"] == "success"
            assert "gradcam_image" in res
            assert len(res["gradcam_image"]) > 100
            return res["prediction"], res["confidence"]

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker_task, i) for i in range(6)]
            results = [f.result() for f in futures]

        assert len(results) == 6
        first_pred, first_conf = results[0]
        for pred, conf in results[1:]:
            assert pred == first_pred
            assert abs(conf - first_conf) < 1e-4

    def test_concurrent_vitals_requests(self):
        def vitals_task(hr, spo2, temp):
            return analyze_vitals(hr, spo2, temp)

        test_cases = [
            (70.0, 98.0, 36.6),
            (130.0, 98.0, 36.6),
            (50.0, 92.0, 37.5),
            (80.0, 99.0, 36.9),
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(vitals_task, hr, sp, tp) for hr, sp, tp in test_cases]
            results = [f.result() for f in futures]

        assert results[0]["overall_status"] == "Normal"
        assert results[1]["overall_status"] == "Abnormal"
        assert results[2]["overall_status"] == "Abnormal"
        assert results[3]["overall_status"] == "Normal"


class TestResponseSchemaInvariants:
    """Verifies strict adherence to data schemas, probability bounds, and types."""

    def test_canonical_four_class_schema_invariants(self, test_fundus_fixtures):
        valid_img = test_fundus_fixtures["valid_img"]
        ckpt = test_fundus_fixtures["checkpoint"]

        res = screen_eye_image(valid_img, weights_path=ckpt, device=torch.device("cpu"), include_gradcam=False)
        assert res["status"] == "success"

        # 1. Prediction must be in canonical CLASS_NAMES
        assert res["prediction"] in CLASS_NAMES
        # 2. class_index must map exactly to prediction
        assert CLASS_NAMES[res["class_index"]] == res["prediction"]
        # 3. Exactly 4 classes in canonical dictionary order
        assert list(res["all_probabilities"].keys()) == CLASS_NAMES
        # 4. Confidence must equal the probability of the predicted class
        assert abs(res["confidence"] - res["all_probabilities"][res["prediction"]]) < 1e-4

    def test_probability_distribution_invariants(self, test_fundus_fixtures):
        valid_img = test_fundus_fixtures["valid_img"]
        ckpt = test_fundus_fixtures["checkpoint"]

        res = screen_eye_image(valid_img, weights_path=ckpt, device=torch.device("cpu"))
        probs = res["all_probabilities"]

        prob_sum = 0.0
        for name, p in probs.items():
            assert isinstance(p, float), f"Expected float, got {type(p)}"
            assert not np.isnan(p), "Probability must not be NaN"
            assert not np.isinf(p), "Probability must not be infinite"
            assert 0.0 <= p <= 1.0, f"Probability out of range: {p}"
            prob_sum += p

        # Sum of probabilities must equal 1.0 within floating point rounding
        assert abs(prob_sum - 1.0) <= 0.005, f"Probabilities do not sum to 1.0: {prob_sum}"

    def test_gradcam_base64_png_invariants(self, test_fundus_fixtures):
        valid_img = test_fundus_fixtures["valid_img"]
        ckpt = test_fundus_fixtures["checkpoint"]

        res = screen_eye_image(valid_img, weights_path=ckpt, device=torch.device("cpu"), include_gradcam=True)
        b64_str = res["gradcam_image"]
        assert isinstance(b64_str, str)
        assert len(b64_str) > 500

        # Decode base64 to bytes
        png_bytes = base64.b64decode(b64_str)
        # Verify official 8-byte PNG file signature: 0x89 0x50 0x4E 0x47 0x0D 0x0A 0x1A 0x0A
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Grad-CAM payload is not a valid PNG"

        # Decode via OpenCV to verify image decodability and channel dimensions
        img_arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert img_arr is not None
        assert img_arr.ndim == 3
        assert img_arr.shape[2] == 3
        assert img_arr.shape[0] == 300 and img_arr.shape[1] == 300  # native input dimensions preserved

    def test_vitals_schema_invariants(self):
        res = analyze_vitals(heart_rate=75.0, spo2=99.0, temperature=36.7)
        required_keys = {
            "heart_rate", "spo2", "temperature",
            "hr_status", "spo2_status", "temp_status",
            "overall_status", "risk_level", "is_abnormal",
            "summary", "disclaimer",
        }
        assert required_keys.issubset(res.keys())
        assert isinstance(res["heart_rate"]["value"], float)
        assert isinstance(res["spo2"]["value"], float)
        assert isinstance(res["temperature"]["value"], float)
        assert isinstance(res["is_abnormal"], bool)
        assert res["risk_level"] in ["LOW", "MODERATE", "HIGH"]

    def test_all_endpoints_strict_json_serializability(self, test_fundus_fixtures):
        valid_img = test_fundus_fixtures["valid_img"]
        ckpt = test_fundus_fixtures["checkpoint"]

        eye_res = screen_eye_image(valid_img, weights_path=ckpt, include_gradcam=True)
        vitals_res = analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=36.8)

        # Standard json.dumps should succeed without custom encoders
        json_eye = json.dumps(eye_res)
        json_vitals = json.dumps(vitals_res)
        assert json_eye is not None
        assert json_vitals is not None


class TestFailureContractsAndErrorDistinction:
    """Verifies that failures produce typed domain errors or clean structured contracts."""

    def test_quality_gate_sentinel_on_unreadable_file(self, tmp_path):
        bad_file = str(tmp_path / "corrupt.jpg")
        with open(bad_file, "wb") as f:
            f.write(b"NOT_AN_IMAGE_FILE")

        res = check_image_quality(bad_file)
        assert res["overall_pass"] is False
        assert res["blur_score"] == 0.0
        assert res["brightness_mean"] == 0.0
        assert res["contrast_std"] == 0.0
        assert res["error_message"] is not None
        assert "corrupt" in res["error_message"].lower() or "decode" in res["error_message"].lower()

    def test_preprocessing_raises_on_corrupt_or_empty_image(self, tmp_path):
        empty_file = str(tmp_path / "empty.png")
        with open(empty_file, "wb") as f:
            pass

        with pytest.raises(InvalidImageInputError):
            preprocess_image(empty_file)

        corrupt_file = str(tmp_path / "corrupt.png")
        with open(corrupt_file, "wb") as f:
            f.write(b"BROKEN_BYTES")

        with pytest.raises(InvalidImageInputError):
            preprocess_image(corrupt_file)

    def test_screen_eye_image_rejection_payload_contract(self, test_fundus_fixtures):
        blurry = test_fundus_fixtures["blurry_img"]
        ckpt = test_fundus_fixtures["checkpoint"]

        res = screen_eye_image(blurry, weights_path=ckpt, include_gradcam=True)
        assert res["status"] == "rejected"
        assert "reason" in res
        assert "quality_checks" in res
        assert "prediction" not in res
        assert "gradcam_image" not in res

    def test_gradcam_direct_error_contracts(self, tmp_path, test_fundus_fixtures):
        valid_img = test_fundus_fixtures["valid_img"]
        ckpt = test_fundus_fixtures["checkpoint"]

        # Invalid class index < 0 or >= 4
        with pytest.raises(ValueError):
            generate_gradcam(valid_img, predicted_class_idx=-1, weights_path=ckpt)

        with pytest.raises(ValueError):
            generate_gradcam(valid_img, predicted_class_idx=4, weights_path=ckpt)

        # Non-integer class index
        with pytest.raises(TypeError):
            generate_gradcam(valid_img, predicted_class_idx=2.5, weights_path=ckpt)  # type: ignore

        # Corrupt image
        bad_img = str(tmp_path / "bad.png")
        with open(bad_img, "wb") as f:
            f.write(b"BROKEN")
        with pytest.raises(InvalidImageInputError):
            generate_gradcam(bad_img, predicted_class_idx=0, weights_path=ckpt)

    def test_vitals_out_of_range_error_contracts(self):
        # Severe implausible vitals (e.g. HR > 300)
        with pytest.raises(PhysiologicalValidationError):
            analyze_vitals(heart_rate=450.0, spo2=98.0, temperature=36.8, raise_errors=True)

        with pytest.raises(PhysiologicalValidationError):
            analyze_vitals(heart_rate=75.0, spo2=105.0, temperature=36.8, raise_errors=True)

        with pytest.raises(PhysiologicalValidationError):
            analyze_vitals(heart_rate=75.0, spo2=98.0, temperature=15.0, raise_errors=True)

    def test_vitals_invalid_type_error_contracts(self):
        with pytest.raises(PhysiologicalTypeValidationError):
            analyze_vitals(heart_rate=None, spo2=98.0, temperature=36.8, raise_errors=True)  # type: ignore

        with pytest.raises(PhysiologicalTypeValidationError):
            analyze_vitals(heart_rate="75", spo2=98.0, temperature=36.8, raise_errors=True)  # type: ignore

        with pytest.raises(PhysiologicalTypeValidationError):
            analyze_vitals(heart_rate=75.0, spo2=True, temperature=36.8, raise_errors=True)  # type: ignore

    def test_vitals_nan_and_infinite_error_contracts(self):
        with pytest.raises(PhysiologicalValidationError):
            analyze_vitals(heart_rate=float("nan"), spo2=98.0, temperature=36.8, raise_errors=True)

        with pytest.raises(PhysiologicalValidationError):
            analyze_vitals(heart_rate=75.0, spo2=float("inf"), temperature=36.8, raise_errors=True)


class TestRealCheckpointIntegration:
    """Exercises real inference using actual trained model weights and dataset images."""

    @pytest.fixture(autouse=True)
    def clean_cache(self):
        clear_model_cache()
        yield
        clear_model_cache()

    def test_real_weights_exist_and_loadable(self):
        target_path = REAL_CHECKPOINT_PATH if os.path.exists(REAL_CHECKPOINT_PATH) else BEST_CHECKPOINT_PATH
        if not os.path.exists(target_path):
            pytest.skip(f"No trained checkpoint found at {target_path}")

        assert os.path.getsize(target_path) > 10 * 1024 * 1024  # > 10 MB

        model = build_densenet201_model(num_classes=NUM_CLASSES)
        loaded = load_eye_model_checkpoint(model, target_path, device=torch.device("cpu"))
        assert loaded is not None

    def test_real_weights_held_out_sample_inference_cpu(self):
        target_path = REAL_CHECKPOINT_PATH if os.path.exists(REAL_CHECKPOINT_PATH) else BEST_CHECKPOINT_PATH
        if not os.path.exists(target_path):
            pytest.skip("No trained checkpoint found for real inference")

        if not os.path.exists(SPLIT_MANIFEST_PATH):
            pytest.skip("Manifest not found")

        manifest = load_split_manifest(SPLIT_MANIFEST_PATH)
        test_records = manifest["splits"]["test"]

        # Find first sample passing quality gate
        passing_image = None
        for r in test_records:
            if os.path.exists(r["filepath"]):
                q = check_image_quality(r["filepath"])
                if q["overall_pass"]:
                    passing_image = r["filepath"]
                    break

        if not passing_image:
            pytest.skip("No quality-passing sample image found in test split")

        # Execute screen_eye_image on CPU
        res = screen_eye_image(
            passing_image,
            weights_path=target_path,
            device=torch.device("cpu"),
            include_gradcam=True,
        )

        assert res["status"] == "success"
        assert res["prediction"] in CLASS_NAMES
        assert 0 <= res["class_index"] < NUM_CLASSES
        assert res["confidence"] > 0.40  # Trained model has decisive confidence
        assert "gradcam_image" in res
        assert len(res["gradcam_image"]) > 1000

        # JSON serializability
        assert json.dumps(res) is not None

    def test_real_weights_held_out_sample_inference_cuda_if_available(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this host")

        target_path = REAL_CHECKPOINT_PATH if os.path.exists(REAL_CHECKPOINT_PATH) else BEST_CHECKPOINT_PATH
        if not os.path.exists(target_path):
            pytest.skip("No trained checkpoint found")

        if not os.path.exists(SPLIT_MANIFEST_PATH):
            pytest.skip("Manifest not found")

        manifest = load_split_manifest(SPLIT_MANIFEST_PATH)
        test_records = manifest["splits"]["test"]

        passing_image = None
        for r in test_records:
            if os.path.exists(r["filepath"]):
                q = check_image_quality(r["filepath"])
                if q["overall_pass"]:
                    passing_image = r["filepath"]
                    break

        if not passing_image:
            pytest.skip("No quality-passing sample found")

        res_cuda = screen_eye_image(
            passing_image,
            weights_path=target_path,
            device=torch.device("cuda"),
            include_gradcam=True,
        )
        assert res_cuda["status"] == "success"
        assert res_cuda["prediction"] in CLASS_NAMES


class TestBackendFastAPIRouteSimulation:
    """
    Simulates Member 2's FastAPI route handlers to prove zero unhandled crashes.
    """

    def _simulate_fastapi_predict_eye_route(self, image_path: str, weights_path: str):
        """Simulates FastAPI @app.post('/api/v1/predict_eye')."""
        if not os.path.exists(image_path):
            return {"status_code": 400, "detail": f"Bad Request: File not found: {image_path}"}
        try:
            result = screen_eye_image(image_path, weights_path=weights_path, include_gradcam=True)
            if result["status"] == "rejected":
                # Route returns 200 with status=rejected and explanation
                return {"status_code": 200, "json": result}
            return {"status_code": 200, "json": result}
        except (CheckpointNotFoundError, CheckpointCorruptError, ModelInferenceError, ResourceExhaustionError) as err:
            return {"status_code": 503, "detail": f"Service Unavailable: {err}"}
        except (InvalidImageInputError, FileNotFoundError, ValueError) as err:
            return {"status_code": 400, "detail": f"Bad Request: {err}"}
        except Exception as err:
            return {"status_code": 500, "detail": f"Internal Server Error: {err}"}

    def _simulate_fastapi_submit_vitals_route(self, hr: Any, spo2: Any, temp: Any):
        """Simulates FastAPI @app.post('/api/v1/submit_vitals')."""
        try:
            result = analyze_vitals(hr, spo2, temp, raise_errors=True)
            return {"status_code": 200, "json": result}
        except (PhysiologicalValidationError, PhysiologicalTypeValidationError) as err:
            return {"status_code": 422, "detail": f"Unprocessable Entity: {err}"}
        except Exception as err:
            return {"status_code": 500, "detail": f"Internal Server Error: {err}"}

    def test_fastapi_predict_eye_route_simulation(self, test_fundus_fixtures, tmp_path):
        valid_img = test_fundus_fixtures["valid_img"]
        blurry_img = test_fundus_fixtures["blurry_img"]
        ckpt = test_fundus_fixtures["checkpoint"]

        # Valid image -> HTTP 200
        resp_valid = self._simulate_fastapi_predict_eye_route(valid_img, ckpt)
        assert resp_valid["status_code"] == 200
        assert resp_valid["json"]["status"] == "success"

        # Blurry image -> HTTP 200 with status=rejected
        resp_reject = self._simulate_fastapi_predict_eye_route(blurry_img, ckpt)
        assert resp_reject["status_code"] == 200
        assert resp_reject["json"]["status"] == "rejected"

        # Missing file -> HTTP 400 Bad Request
        resp_missing = self._simulate_fastapi_predict_eye_route("nonexistent.png", ckpt)
        assert resp_missing["status_code"] == 400

        # Missing checkpoint -> HTTP 503 Service Unavailable
        resp_no_ckpt = self._simulate_fastapi_predict_eye_route(valid_img, "no_weights.pth")
        assert resp_no_ckpt["status_code"] == 503

    def test_fastapi_submit_vitals_route_simulation(self):
        # Valid vitals -> HTTP 200
        resp_valid = self._simulate_fastapi_submit_vitals_route(72.0, 98.0, 36.8)
        assert resp_valid["status_code"] == 200
        assert resp_valid["json"]["overall_status"] == "Normal"

        # Out-of-range vitals -> HTTP 422 Unprocessable Entity
        resp_invalid_range = self._simulate_fastapi_submit_vitals_route(500.0, 98.0, 36.8)
        assert resp_invalid_range["status_code"] == 422

        # Non-numeric vitals -> HTTP 422 Unprocessable Entity
        resp_invalid_type = self._simulate_fastapi_submit_vitals_route("abc", 98.0, 36.8)
        assert resp_invalid_type["status_code"] == 422
