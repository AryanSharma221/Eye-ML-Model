"""
End-to-End Frontend + Backend Integration Test Suite.
Verifies the complete integration:
1. Static frontend build serving and asset links
2. Real CORS origin headers for Vite dev server (http://localhost:5173)
3. Eye-screening image upload, quality gate checks, and Grad-CAM delivery
4. Vital-sign analysis with structured physiological triage
5. Combined multimodal screening with independent stream error preservation
6. Patient identity registration, session attachment, and ReportLab PDF download
"""

import io
import os
from pathlib import Path
from PIL import Image, ImageFilter
import pytest
from fastapi.testclient import TestClient

from backend.database import Base, engine
from backend.main import app

client = TestClient(app)

TEST_SAMPLES = {
    "cataract": r"C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset\test\cataract\kaggle_2052_right.jpg",
    "normal": r"C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset\test\normal\kaggle_2587_right.jpg",
}


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield


def test_frontend_serving_and_html_entry():
    """Verifies that the root path serves the compiled React frontend with valid HTML."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert '<div id="root">' in res.text or "<div id='root'>" in res.text
    assert "Multimodal Health Screening" in res.text


def test_cors_headers_for_vite_client():
    """Verifies that CORS headers permit Vite dev server at http://localhost:5173."""
    res = client.options(
        "/api/v1/screen",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_frontend_health_probe_readiness():
    """Verifies that the /health endpoint consumed by Navbar.jsx indicates AI model readiness."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["checkpoint_available"] is True
    assert "cataract" in data["supported_classes"]
    assert "normal" in data["supported_classes"]


def test_full_multimodal_screening_flow():
    """Verifies complete combined screening flow (Image + Vitals)."""
    normal_img = TEST_SAMPLES["normal"]
    assert os.path.isfile(normal_img)

    with open(normal_img, "rb") as f:
        file_bytes = f.read()

    res = client.post(
        "/api/v1/screen",
        files={"image": ("test_eye.jpg", file_bytes, "image/jpeg")},
        data={
            "heart_rate": "76.0",
            "spo2": "98.5",
            "temperature": "36.8",
            "include_gradcam": "true",
        },
    )
    assert res.status_code == 200
    body = res.json()

    # 1. Structure
    assert body["status"] == "success"
    assert "overall_risk_level" in body
    assert "duration_ms" in body
    assert body["disclaimer"] is not None

    # 2. Eye screening
    eye = body["eye_screening"]
    assert eye["status"] == "success"
    assert eye["prediction"] in ("normal", "cataract", "glaucoma", "diabetic_retinopathy")
    assert 0.0 <= eye["confidence"] <= 1.0
    assert eye["gradcam_image"] is not None
    assert eye["quality_checks"]["overall_pass"] is True

    # 3. Vitals screening
    vitals = body["vitals_screening"]
    assert vitals["status"] == "success"
    assert vitals["overall_status"] == "Normal"
    assert vitals["risk_level"] == "LOW"


def test_quality_gate_rejection_in_combined_flow():
    """Verifies that a blurry image does not prevent independent vital sign analysis."""
    # Create blurry image
    img = Image.new("RGB", (300, 300), color=(120, 120, 120))
    img = img.filter(ImageFilter.GaussianBlur(radius=20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    blurry_bytes = buf.getvalue()

    res = client.post(
        "/api/v1/screen",
        files={"image": ("blurry.jpg", blurry_bytes, "image/jpeg")},
        data={
            "heart_rate": "80.0",
            "spo2": "99.0",
            "temperature": "36.7",
        },
    )
    assert res.status_code == 200
    body = res.json()

    # Eye screening should be rejected
    eye = body["eye_screening"]
    assert eye["status"] == "rejected"
    assert eye["quality_checks"]["overall_pass"] is False
    assert "failed" in eye["reason"].lower() or "blur" in eye["reason"].lower()

    # Vitals screening should still succeed
    vitals = body["vitals_screening"]
    assert vitals["status"] == "success"
    assert vitals["overall_status"] == "Normal"


def test_independent_eye_endpoint_contract():
    """Verifies /api/v1/predict_eye endpoint used by independent image analysis."""
    normal_img = TEST_SAMPLES["normal"]
    with open(normal_img, "rb") as f:
        file_bytes = f.read()

    res = client.post(
        "/api/v1/predict_eye",
        files={"image": ("retina.jpg", file_bytes, "image/jpeg")},
        data={"include_gradcam": "true"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["prediction"] == "normal"
    assert data["confidence"] > 0.5
    assert data["gradcam_image"] is not None
    assert "quality_checks" in data
    assert data["quality_checks"]["overall_pass"] is True


def test_independent_vitals_endpoint_contract():
    """Verifies /api/v1/submit_vitals endpoint used by ESP32/Manual inputs."""
    res = client.post(
        "/api/v1/submit_vitals",
        json={
            "heart_rate_bpm": 112.0,
            "spo2_percent": 97.0,
            "temperature_celsius": 38.4,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["hr_status"] == "Higher than usual"
    assert data["temp_status"] == "Abnormal"
    assert data["overall_status"] == "Abnormal"
    assert data["risk_level"] == "HIGH"


def test_patient_session_reportlab_pdf_flow():
    """Verifies patient registration, session attachment, and PDF download."""
    # 1. Register patient
    p_res = client.post(
        "/api/v1/patients",
        json={"name": "Integration Patient", "age": 58, "gender": "Female", "contact": "patient@integration.test"},
    )
    assert p_res.status_code == 201
    patient_id = p_res.json()["id"]

    # 2. Run screening attached to patient
    normal_img = TEST_SAMPLES["normal"]
    with open(normal_img, "rb") as f:
        file_bytes = f.read()

    screen_res = client.post(
        "/api/v1/screen",
        files={"image": ("retina.jpg", file_bytes, "image/jpeg")},
        data={
            "patient_id": patient_id,
            "heart_rate": "72.0",
            "spo2": "98.0",
            "temperature": "36.6",
        },
    )
    assert screen_res.status_code == 200
    sess_id = screen_res.json()["session_id"]
    assert sess_id is not None

    # 3. Retrieve sessions for patient
    hist_res = client.get(f"/api/v1/patients/{patient_id}/sessions")
    assert hist_res.status_code == 200
    sessions = hist_res.json()
    assert len(sessions) >= 1
    assert sessions[0]["id"] == sess_id

    # 4. Download ReportLab PDF
    pdf_res = client.get(f"/api/v1/sessions/{sess_id}/report?format=pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers.get("content-type") == "application/pdf"
    assert pdf_res.content.startswith(b"%PDF")
    assert len(pdf_res.content) > 1000
