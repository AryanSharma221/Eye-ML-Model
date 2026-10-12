"""
End-to-End Integration Tests for Multimodal Health Screening Backend and Web API.
Tests cover Scenarios A through F, health readiness, validation gates, and session reporting.
"""

import io
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import Base, engine
from ml.eye_model import CLASS_NAMES

# Initialize test client
client = TestClient(app)

# Real held-out test sample paths passing quality gate
TEST_SAMPLES = {
    "cataract": r"C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset\test\cataract\kaggle_2052_right.jpg",
    "diabetic_retinopathy": r"C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset\test\diabetic_retinopathy\kaggle_10184_left.jpeg",
    "glaucoma": r"C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset\test\glaucoma\kaggle_1491_left.jpg",
    "normal": r"C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset\test\normal\kaggle_2587_right.jpg",
}


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Ensure database schema is created."""
    Base.metadata.create_all(bind=engine)
    yield


def create_synthetic_blurry_image() -> bytes:
    """Creates a blurry synthetic image guaranteed to fail the Laplacian blur gate (< 100)."""
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    # Apply severe Gaussian blur
    img = img.filter(ImageFilter.GaussianBlur(radius=15))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Test Suite 1: Health & Readiness Endpoints
# -----------------------------------------------------------------------------
def test_health_endpoints():
    """Verifies that /health and /api/v1/health report model readiness and device."""
    for path in ["/health", "/api/v1/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["checkpoint_available"] is True
        assert data["supported_classes"] == CLASS_NAMES
        assert "timestamp" in data
        assert isinstance(data["device"], str)


def test_liveness_and_readiness_probes():
    """Verifies dedicated liveness (/health/live) and readiness (/health/ready) probes."""
    # 1. Liveness Probe
    for path in ["/health/live", "/api/v1/health/live"]:
        live_res = client.get(path)
        assert live_res.status_code == 200
        live_data = live_res.json()
        assert live_data["status"] == "alive"
        assert "timestamp" in live_data

    # 2. Readiness Probe
    for path in ["/health/ready", "/api/v1/health/ready"]:
        ready_res = client.get(path)
        assert ready_res.status_code == 200
        ready_data = ready_res.json()
        assert ready_data["status"] == "ready"
        assert ready_data["checkpoint_available"] is True
        assert ready_data["model_loaded"] is True
        assert ready_data["database_connected"] is True


# -----------------------------------------------------------------------------
# Test Suite 2: Scenario A - Normal Fundus Image + Normal Vitals
# -----------------------------------------------------------------------------
def test_scenario_a_normal_image_and_normal_vitals():
    """Scenario A: Valid normal retinal image + normal vitals -> success, LOW risk."""
    normal_path = TEST_SAMPLES["normal"]
    assert os.path.isfile(normal_path), f"Test image missing: {normal_path}"

    with open(normal_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/screen",
        files={"image": ("normal_eye.png", file_bytes, "image/png")},
        data={
            "heart_rate": 72.0,
            "spo2": 98.5,
            "temperature": 36.6,
            "include_gradcam": "true",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["overall_risk_level"] in ("LOW", "MODERATE")

    # Eye screening assertions
    eye = data["eye_screening"]
    assert eye["status"] == "success"
    assert eye["prediction"] in CLASS_NAMES
    assert 0.0 <= eye["confidence"] <= 1.0
    assert eye["quality_checks"]["overall_pass"] is True
    assert eye["gradcam_image"] is not None
    assert len(eye["all_probabilities"]) == 4

    # Vital screening assertions
    vitals = data["vitals_screening"]
    assert vitals["status"] == "success"
    assert vitals["hr_status"] == "Normal"
    assert vitals["spo2_status"] == "Normal"
    assert vitals["temp_status"] == "Normal"
    assert vitals["is_abnormal"] is False
    assert vitals["risk_level"] == "LOW"

    # Clinical disclaimer present
    assert "disclaimer" in data
    assert "preliminary screening" in data["disclaimer"].lower() or "clinical triage" in data["disclaimer"].lower()


# -----------------------------------------------------------------------------
# Test Suite 3: Scenario B - Pathological Fundus Image + Abnormal Vitals
# -----------------------------------------------------------------------------
def test_scenario_b_pathological_image_and_abnormal_vitals():
    """Scenario B: Valid pathological retinal image + abnormal vitals -> success, HIGH risk."""
    cataract_path = TEST_SAMPLES["cataract"]
    assert os.path.isfile(cataract_path), f"Test image missing: {cataract_path}"

    with open(cataract_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/screen",
        files={"image": ("cataract_eye.png", file_bytes, "image/png")},
        data={
            "heart_rate": 128.0,   # Tachycardia (>100)
            "spo2": 88.0,          # Hypoxia (<90)
            "temperature": 38.8,   # Fever (>38.0)
            "include_gradcam": "true",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["overall_risk_level"] == "HIGH"

    # Eye screening assertions
    eye = data["eye_screening"]
    assert eye["status"] == "success"
    assert eye["prediction"] in CLASS_NAMES
    assert eye["gradcam_image"] is not None

    # Vital screening assertions
    vitals = data["vitals_screening"]
    assert vitals["status"] == "success"
    assert vitals["is_abnormal"] is True
    assert vitals["risk_level"] == "HIGH"
    assert "Abnormal" in vitals["spo2_status"] or "Higher" in vitals["hr_status"]


# -----------------------------------------------------------------------------
# Test Suite 4: Scenario C - Blurry/Non-Retinal Image + Valid Vitals
# -----------------------------------------------------------------------------
def test_scenario_c_blurry_image_rejection_with_valid_vitals():
    """
    Scenario C: Blurry image fails quality gate.
    Unified endpoint evaluates vitals independently while rejecting retinal image.
    """
    blurry_bytes = create_synthetic_blurry_image()

    response = client.post(
        "/api/v1/screen",
        files={"image": ("blurry.jpg", blurry_bytes, "image/jpeg")},
        data={
            "heart_rate": 75.0,
            "spo2": 98.0,
            "temperature": 36.8,
            "include_gradcam": "true",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Eye screening must be strictly rejected
    eye = data["eye_screening"]
    assert eye["status"] == "rejected"
    assert "quality" in eye["reason"].lower() or "blur" in eye["reason"].lower()
    assert eye["quality_checks"]["overall_pass"] is False
    assert eye["gradcam_image"] is None  # Never generated for rejected image

    # Vitals screening evaluated independently
    vitals = data["vitals_screening"]
    assert vitals["status"] == "success"
    assert vitals["risk_level"] == "LOW"


def test_scenario_c_blurry_image_predict_eye_alone():
    """Scenario C (standalone): Blurry image sent to /predict_eye returns 400 Bad Request."""
    blurry_bytes = create_synthetic_blurry_image()

    response = client.post(
        "/api/v1/predict_eye",
        files={"image": ("blurry.jpg", blurry_bytes, "image/jpeg")},
    )

    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "rejected"
    assert "reason" in data
    assert data["quality_checks"]["overall_pass"] is False


# -----------------------------------------------------------------------------
# Test Suite 5: Scenario D - Valid Retinal Image + Missing/Partial Vitals
# -----------------------------------------------------------------------------
def test_scenario_d_valid_image_and_missing_vitals():
    """Scenario D: Valid retinal image with no vitals provided -> eye screening succeeds."""
    glaucoma_path = TEST_SAMPLES["glaucoma"]
    assert os.path.isfile(glaucoma_path)

    with open(glaucoma_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/screen",
        files={"image": ("glaucoma_eye.png", file_bytes, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("success", "partial")

    eye = data["eye_screening"]
    assert eye["status"] == "success"
    assert eye["prediction"] in CLASS_NAMES
    assert eye["gradcam_image"] is not None

    vitals = data["vitals_screening"]
    assert vitals["status"] == "skipped"


def test_scenario_d_valid_image_and_partial_vitals():
    """Scenario D: Valid retinal image with partial vitals (missing SpO2 and Temp)."""
    glaucoma_path = TEST_SAMPLES["glaucoma"]

    with open(glaucoma_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/screen",
        files={"image": ("glaucoma_eye.png", file_bytes, "image/png")},
        data={"heart_rate": 80.0},  # spo2 and temp omitted
    )

    assert response.status_code == 200
    data = response.json()

    # Eye screening succeeds
    eye = data["eye_screening"]
    assert eye["status"] == "success"

    # Vitals reports error due to incomplete fields without aborting eye screening
    vitals = data["vitals_screening"]
    assert vitals["status"] == "error"
    assert "missing" in vitals["reason"].lower()


# -----------------------------------------------------------------------------
# Test Suite 6: Scenario E - Corrupted / Invalid Image File
# -----------------------------------------------------------------------------
def test_scenario_e_corrupted_image_file():
    """Scenario E: Corrupted non-image bytes disguised as .jpg -> 400 Bad Request."""
    corrupted_bytes = b"NOT_A_VALID_JPEG_HEADER_RANDOM_CORRUPT_BYTES_DATA_STREAM"

    response = client.post(
        "/api/v1/predict_eye",
        files={"image": ("corrupted.jpg", corrupted_bytes, "image/jpeg")},
    )

    assert response.status_code == 400
    data = response.json()
    err_str = str(data).lower()
    assert "failed to decode" in err_str or "corrupt" in err_str or "invalid" in err_str


def test_scenario_e_unsupported_file_extension():
    """Scenario E: Unsupported file extension (e.g. .pdf or .txt) -> 400 Bad Request."""
    response = client.post(
        "/api/v1/predict_eye",
        files={"image": ("report.pdf", b"%PDF-1.4...", "application/pdf")},
    )
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


# -----------------------------------------------------------------------------
# Test Suite 7: Scenario F - Missing File / Empty Request
# -----------------------------------------------------------------------------
def test_scenario_f_missing_file_upload_predict_eye():
    """Scenario F: POST /predict_eye without image parameter -> 422 Unprocessable."""
    response = client.post("/api/v1/predict_eye")
    assert response.status_code == 422


def test_scenario_f_empty_screen_request():
    """Scenario F: POST /api/v1/screen without image and without vitals -> 400 Bad Request."""
    response = client.post("/api/v1/screen")
    assert response.status_code == 400
    assert "at least one modality" in response.json()["detail"].lower()


# -----------------------------------------------------------------------------
# Test Suite 8: Vitals Range Boundary Validation
# -----------------------------------------------------------------------------
def test_vitals_out_of_range_rejection():
    """Tests physiological limit enforcement (HR [30, 220], SpO2 [50, 100], Temp [30, 45])."""
    # Heart rate > 220
    res = client.post(
        "/api/v1/submit_vitals",
        json={"heart_rate_bpm": 250.0, "spo2_percent": 98.0, "temperature_celsius": 37.0},
    )
    assert res.status_code == 422 or res.status_code == 400

    # SpO2 < 50
    res2 = client.post(
        "/api/v1/submit_vitals",
        json={"heart_rate_bpm": 80.0, "spo2_percent": 40.0, "temperature_celsius": 37.0},
    )
    assert res2.status_code == 422 or res2.status_code == 400

    # Temperature > 45
    res3 = client.post(
        "/api/v1/submit_vitals",
        json={"heart_rate_bpm": 80.0, "spo2_percent": 98.0, "temperature_celsius": 48.0},
    )
    assert res3.status_code == 422 or res3.status_code == 400


# -----------------------------------------------------------------------------
# Test Suite 9: Patient Management & Session Reporting
# -----------------------------------------------------------------------------
def test_patient_registration_and_session_report():
    """Tests creating patient, running session, and generating printable HTML report."""
    # 1. Create Patient
    p_res = client.post(
        "/api/v1/patients",
        json={"name": "Alice Cooper", "age": 62, "gender": "Female", "contact": "alice@med.org"},
    )
    assert p_res.status_code == 201
    patient = p_res.json()
    p_id = patient["id"]

    # 2. Run Multimodal Screening with patient_id
    normal_path = TEST_SAMPLES["normal"]
    with open(normal_path, "rb") as f:
        file_bytes = f.read()

    screen_res = client.post(
        "/api/v1/screen",
        files={"image": ("alice_fundus.png", file_bytes, "image/png")},
        data={
            "patient_id": p_id,
            "heart_rate": 74.0,
            "spo2": 99.0,
            "temperature": 36.7,
        },
    )
    assert screen_res.status_code == 200
    s_data = screen_res.json()
    sess_id = s_data["session_id"]
    assert sess_id is not None

    # 3. Retrieve Session
    sess_res = client.get(f"/api/v1/sessions/{sess_id}")
    assert sess_res.status_code == 200
    sess = sess_res.json()
    assert len(sess["eye_records"]) >= 1
    assert len(sess["physio_records"]) >= 1

    # 4. Fetch Printable HTML Report
    report_res = client.get(f"/api/v1/sessions/{sess_id}/report")
    assert report_res.status_code == 200
    assert "text/html" in report_res.headers["content-type"]
    html_text = report_res.text
    assert "Alice Cooper" in html_text
    assert "Clinical Screening Report" in html_text
    assert "DenseNet201" in html_text
    assert "Clinical Disclaimer" in html_text


# -----------------------------------------------------------------------------
# Test Suite 10: Static Frontend Mount
# -----------------------------------------------------------------------------
def test_static_frontend_mount():
    """Verifies that visiting / serves the built React web application."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Multimodal Health Screening" in response.text
