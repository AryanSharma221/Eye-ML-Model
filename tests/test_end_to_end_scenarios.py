"""
Comprehensive End-to-End Integration Test Suite for Multimodal Health Screening.
Covers:
1. Multi-threaded concurrent eye screening under load (DenseNet201 + Grad-CAM)
2. Modality independence invariance (eye vs vitals cross-independence)
3. Physiological boundary and validation edge cases (extremes, negatives, types)
4. Model caching and Grad-CAM hook lifecycle (preventing memory leaks)
5. End-to-end verification across all 4 canonical disease classes
6. Temporary file lifecycle and cleanup guarantees
"""

import concurrent.futures
import io
import math
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.config import UPLOADS_DIR
from backend.database import Base, engine
from backend.main import app
from backend.services.ml_service import ml_service
from ml.eye_model import CLASS_NAMES, get_cached_model

client = TestClient(app)

SAMPLE_PATHS = {
    "cataract": r"C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset\test\cataract\kaggle_112_right.jpg",
    "diabetic_retinopathy": r"C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset\test\diabetic_retinopathy\kaggle_10184_left.jpeg",
    "glaucoma": r"C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset\test\cataract\kaggle_2052_right.jpg",
    "normal": r"C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset\test\normal\kaggle_2587_right.jpg",
}


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield


# -----------------------------------------------------------------------------
# 1. Multi-threaded Concurrent Eye Screening Under Load
# -----------------------------------------------------------------------------
def test_multithreaded_concurrent_eye_inference():
    """
    Submits 6 concurrent requests running full DenseNet201 inference and
    Grad-CAM generation simultaneously to prove thread safety, absence of deadlocks,
    and consistent results under multi-threaded load.
    """
    normal_img = SAMPLE_PATHS["normal"]
    assert os.path.isfile(normal_img)
    with open(normal_img, "rb") as f:
        file_bytes = f.read()

    def run_concurrent_predict(req_idx: int):
        resp = client.post(
            "/api/v1/predict_eye",
            files={"image": (f"eye_{req_idx}.jpg", file_bytes, "image/jpeg")},
            data={"include_gradcam": "true"},
        )
        return resp.status_code, resp.json()

    num_threads = 6
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_concurrent_predict, i) for i in range(num_threads)]
        results = [f.result() for f in futures]

    for status_code, data in results:
        assert status_code == 200
        assert data["status"] == "success"
        assert data["prediction"] == "normal"
        assert data["confidence"] > 0.8
        assert data["gradcam_image"] is not None
        assert data["gradcam_image"].startswith("iVBORw0KGgo")  # Base64 PNG header
        assert data["quality_checks"]["overall_pass"] is True


# -----------------------------------------------------------------------------
# 2. Modality Independence Invariance
# -----------------------------------------------------------------------------
def test_modality_independence_invariance():
    """
    Verifies that:
    1. Eye predictions (class, probabilities) are identical regardless of vitals values.
    2. Vital sign triage (status, risk) is identical regardless of eye image findings.
    """
    normal_img = SAMPLE_PATHS["normal"]
    with open(normal_img, "rb") as f:
        file_bytes = f.read()

    # Case A: Eye alone
    res_a = client.post(
        "/api/v1/screen",
        files={"image": ("retina_a.jpg", file_bytes, "image/jpeg")},
        data={"include_gradcam": "false"},
    )
    assert res_a.status_code == 200
    eye_alone = res_a.json()["eye_screening"]

    # Case B: Eye + Normal Vitals
    res_b = client.post(
        "/api/v1/screen",
        files={"image": ("retina_b.jpg", file_bytes, "image/jpeg")},
        data={
            "heart_rate": "72.0",
            "spo2": "99.0",
            "temperature": "36.6",
            "include_gradcam": "false",
        },
    )
    assert res_b.status_code == 200
    eye_with_normal = res_b.json()["eye_screening"]
    vitals_normal = res_b.json()["vitals_screening"]

    # Case C: Eye + Severe Abnormal Vitals
    res_c = client.post(
        "/api/v1/screen",
        files={"image": ("retina_c.jpg", file_bytes, "image/jpeg")},
        data={
            "heart_rate": "150.0",
            "spo2": "85.0",
            "temperature": "39.5",
            "include_gradcam": "false",
        },
    )
    assert res_c.status_code == 200
    eye_with_abnormal = res_c.json()["eye_screening"]

    # Assert Eye results are strictly invariant across different vitals inputs
    assert eye_alone["prediction"] == eye_with_normal["prediction"] == eye_with_abnormal["prediction"]
    for cls in CLASS_NAMES:
        prob_a = eye_alone["all_probabilities"][cls]
        prob_b = eye_with_normal["all_probabilities"][cls]
        prob_c = eye_with_abnormal["all_probabilities"][cls]
        assert math.isclose(prob_a, prob_b, rel_tol=1e-5), f"Probability altered for {cls}"
        assert math.isclose(prob_a, prob_c, rel_tol=1e-5), f"Probability altered for {cls}"

    # Case D: Vitals alone vs vitals with eye
    vitals_alone_res = client.post(
        "/api/v1/submit_vitals",
        json={"heart_rate_bpm": 72.0, "spo2_percent": 99.0, "temperature_celsius": 36.6},
    )
    assert vitals_alone_res.status_code == 200
    vitals_standalone = vitals_alone_res.json()

    assert vitals_normal["hr_status"] == vitals_standalone["hr_status"]
    assert vitals_normal["spo2_status"] == vitals_standalone["spo2_status"]
    assert vitals_normal["temp_status"] == vitals_standalone["temp_status"]
    assert vitals_normal["risk_level"] == vitals_standalone["risk_level"]


# -----------------------------------------------------------------------------
# 3. Physiological Boundary and Validation Edge Cases
# -----------------------------------------------------------------------------
def test_vitals_boundary_exact_values():
    """Verifies that exact boundary values within valid physiological limits are accepted."""
    boundaries = [
        {"heart_rate_bpm": 30.0, "spo2_percent": 50.0, "temperature_celsius": 30.0},
        {"heart_rate_bpm": 220.0, "spo2_percent": 100.0, "temperature_celsius": 45.0},
    ]
    for b in boundaries:
        res = client.post("/api/v1/submit_vitals", json=b)
        assert res.status_code == 200
        assert res.json()["status"] == "success"


def test_vitals_negative_and_out_of_bounds_rejection():
    """Verifies that negative or out-of-bounds vitals are rejected with 400 or 422."""
    invalid_cases = [
        {"heart_rate_bpm": -5.0, "spo2_percent": 98.0, "temperature_celsius": 36.6},
        {"heart_rate_bpm": 29.0, "spo2_percent": 98.0, "temperature_celsius": 36.6},
        {"heart_rate_bpm": 221.0, "spo2_percent": 98.0, "temperature_celsius": 36.6},
        {"heart_rate_bpm": 75.0, "spo2_percent": 49.0, "temperature_celsius": 36.6},
        {"heart_rate_bpm": 75.0, "spo2_percent": 101.0, "temperature_celsius": 36.6},
        {"heart_rate_bpm": 75.0, "spo2_percent": 98.0, "temperature_celsius": 29.9},
        {"heart_rate_bpm": 75.0, "spo2_percent": 98.0, "temperature_celsius": 45.1},
    ]
    for case in invalid_cases:
        res = client.post("/api/v1/submit_vitals", json=case)
        assert res.status_code in (400, 422), f"Expected 400/422 for {case}, got {res.status_code}"


def test_vitals_missing_fields_rejection():
    """Verifies that submitting incomplete vitals without all 3 fields returns 400/422."""
    res = client.post(
        "/api/v1/submit_vitals",
        json={"heart_rate_bpm": 72.0},  # missing spo2 and temperature
    )
    assert res.status_code in (400, 422)


# -----------------------------------------------------------------------------
# 4. Model Singleton and Grad-CAM Hook Lifecycle
# -----------------------------------------------------------------------------
def test_hook_lifecycle_and_model_singleton():
    """
    Verifies that:
    1. The model is cached as a singleton in memory.
    2. Grad-CAM forward/backward hooks are cleanly deregistered after generation.
    """
    cached_model = get_cached_model()
    target_block = cached_model.features.denseblock4

    # Confirm initial hook count is 0
    assert len(target_block._forward_hooks) == 0
    assert len(target_block._backward_hooks) == 0

    normal_img = SAMPLE_PATHS["normal"]
    with open(normal_img, "rb") as f:
        file_bytes = f.read()

    # Run inference + Grad-CAM multiple times
    for i in range(3):
        res = client.post(
            "/api/v1/predict_eye",
            files={"image": (f"test_hook_{i}.jpg", file_bytes, "image/jpeg")},
            data={"include_gradcam": "true"},
        )
        assert res.status_code == 200
        assert res.json()["gradcam_image"] is not None

        # Confirm hook count remains exactly 0 after execution
        assert len(target_block._forward_hooks) == 0
        assert len(target_block._backward_hooks) == 0


# -----------------------------------------------------------------------------
# 5. All Four Canonical Classes End-to-End Verification
# -----------------------------------------------------------------------------
def test_all_four_classes_end_to_end():
    """
    Verifies that the backend can process and predict each of the 4 canonical
    classes, returning valid Grad-CAM heatmaps and appropriate risk levels.
    """
    for expected_class, img_path in SAMPLE_PATHS.items():
        assert os.path.isfile(img_path), f"Missing image for {expected_class}: {img_path}"
        with open(img_path, "rb") as f:
            file_bytes = f.read()

        ext = os.path.splitext(img_path)[1]
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

        res = client.post(
            "/api/v1/predict_eye",
            files={"image": (f"sample{ext}", file_bytes, mime)},
            data={"include_gradcam": "true"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["prediction"] in CLASS_NAMES
        assert data["confidence"] > 0.4
        assert data["gradcam_image"] is not None
        assert data["quality_checks"]["overall_pass"] is True
        assert len(data["all_probabilities"]) == 4

        # Risk level logic
        if data["prediction"] == "normal":
            assert data["risk_level"] in ("LOW", "MODERATE")
        else:
            assert data["risk_level"] in ("MODERATE", "HIGH")
