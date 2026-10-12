"""
Advanced Automated Backend Test Suite for Multimodal Health Screening.
Covers:
- CORS configuration headers
- Oversized upload rejection (HTTP 413)
- Missing or incompatible checkpoint handling (HTTP 503)
- Inference error resilience (HTTP 500)
- Isolated Grad-CAM failure tolerance
- Temporary file cleanup on rejection
- Patient CRUD operations (PUT, DELETE, session history)
- PDF report generation via ReportLab (HTTP 200, application/pdf)
- Concurrency stress testing
"""

import concurrent.futures
import io
import os
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from backend.config import MAX_FILE_SIZE_BYTES, UPLOADS_DIR
from backend.database import Base, SessionLocal, engine
from backend.main import app
from backend.models_db import EyeData, Patient, PhysiologicalData, Session as ScreeningSession
from ml.exceptions import (
    CheckpointCorruptError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    ModelInferenceError,
)

client = TestClient(app)

TEST_SAMPLES = {
    "normal": r"C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset\test\normal\kaggle_2587_right.jpg",
}


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield


# -----------------------------------------------------------------------------
# 1. CORS Configuration Tests
# -----------------------------------------------------------------------------
def test_cors_preflight_and_headers():
    """Verifies that CORS preflight OPTIONS returns allowed origins and methods."""
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "GET" in response.headers.get("access-control-allow-methods", "")


# -----------------------------------------------------------------------------
# 2. Oversized Upload Rejection (HTTP 413)
# -----------------------------------------------------------------------------
def test_oversized_file_upload_rejected():
    """Verifies that files exceeding maximum size (30MB) are rejected with 413 Payload Too Large."""
    # Simulate a file exceeding MAX_FILE_SIZE_BYTES
    oversized_data = b"0" * (MAX_FILE_SIZE_BYTES + 1024)
    response = client.post(
        "/api/v1/predict_eye",
        files={"image": ("oversized.png", oversized_data, "image/png")},
    )
    assert response.status_code == 413
    assert "exceeds maximum limit" in response.json()["detail"].lower()


# -----------------------------------------------------------------------------
# 3. Missing & Incompatible Checkpoint Handling (HTTP 503)
# -----------------------------------------------------------------------------
def test_missing_checkpoint_returns_503():
    """Verifies that CheckpointNotFoundError returns HTTP 503 Service Unavailable."""
    with patch("backend.services.ml_service.ml_service.predict_eye") as mock_predict:
        mock_predict.side_effect = CheckpointNotFoundError("eye_model.pth missing")
        with open(TEST_SAMPLES["normal"], "rb") as f:
            file_bytes = f.read()

        response = client.post(
            "/api/v1/predict_eye",
            files={"image": ("normal.jpg", file_bytes, "image/jpeg")},
        )
        assert response.status_code == 503
        assert "checkpoint" in response.json()["detail"].lower()


def test_corrupt_checkpoint_returns_503():
    """Verifies that CheckpointCorruptError returns HTTP 503 Service Unavailable."""
    with patch("backend.services.ml_service.ml_service.predict_eye") as mock_predict:
        mock_predict.side_effect = CheckpointCorruptError("Invalid magic bytes")
        with open(TEST_SAMPLES["normal"], "rb") as f:
            file_bytes = f.read()

        response = client.post(
            "/api/v1/predict_eye",
            files={"image": ("normal.jpg", file_bytes, "image/jpeg")},
        )
        assert response.status_code == 503
        assert "corrupted" in response.json()["detail"].lower()


def test_readiness_probe_fails_503_when_checkpoint_missing():
    """Verifies that /health/ready returns HTTP 503 when the model checkpoint is missing."""
    with patch("os.path.isfile", return_value=False):
        res = client.get("/health/ready")
        assert res.status_code == 503
        data = res.json()
        assert data["status"] == "unready"
        assert data["checkpoint_available"] is False
        assert "not found" in data["reason"].lower()


# -----------------------------------------------------------------------------
# 4. Inference Error Handling (HTTP 500)
# -----------------------------------------------------------------------------
def test_model_inference_error_returns_500():
    """Verifies that unexpected inference failures return HTTP 500 with sanitized message."""
    with patch("backend.services.ml_service.ml_service.predict_eye") as mock_predict:
        mock_predict.side_effect = RuntimeError("CUDA out of memory in DenseNet201")
        with open(TEST_SAMPLES["normal"], "rb") as f:
            file_bytes = f.read()

        response = client.post(
            "/api/v1/predict_eye",
            files={"image": ("normal.jpg", file_bytes, "image/jpeg")},
        )
        assert response.status_code == 500
        # Verify stack trace or internal details not leaked
        assert "internal error" in response.json()["detail"].lower() or "error occurred" in response.json()["detail"].lower()


# -----------------------------------------------------------------------------
# 5. Isolated Grad-CAM Failure Tolerance
# -----------------------------------------------------------------------------
def test_gradcam_isolated_failure_does_not_break_eye_screening():
    """Verifies that if Grad-CAM raises an exception, the eye prediction still succeeds."""
    with patch("backend.services.ml_service.generate_gradcam") as mock_gcam:
        mock_gcam.side_effect = RuntimeError("Isolated Grad-CAM hook failure")
        with open(TEST_SAMPLES["normal"], "rb") as f:
            file_bytes = f.read()

        response = client.post(
            "/api/v1/predict_eye",
            files={"image": ("normal.jpg", file_bytes, "image/jpeg")},
            data={"include_gradcam": "true"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["gradcam_image"] is None
        assert data["prediction"] == "normal"


# -----------------------------------------------------------------------------
# 6. Temporary File Cleanup on Rejection
# -----------------------------------------------------------------------------
def test_temporary_file_cleanup_on_quality_rejection():
    """Verifies that images failing the quality gate are safely cleaned up from disk."""
    from PIL import Image, ImageFilter
    img = Image.new("RGB", (300, 300), color=(100, 100, 100))
    img = img.filter(ImageFilter.GaussianBlur(radius=15))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    blurry_bytes = buf.getvalue()

    initial_files = set(Path(UPLOADS_DIR / "fundus").glob("*")) if (UPLOADS_DIR / "fundus").exists() else set()

    response = client.post(
        "/api/v1/predict_eye",
        files={"image": ("blurry.jpg", blurry_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json()["status"] == "rejected"

    post_files = set(Path(UPLOADS_DIR / "fundus").glob("*")) if (UPLOADS_DIR / "fundus").exists() else set()
    # Confirm no new leftover files remained in the uploads directory
    new_files = post_files - initial_files
    assert len(new_files) == 0, f"Leaked temporary files on rejection: {new_files}"


# -----------------------------------------------------------------------------
# 7. Patient CRUD (PUT, DELETE, Sessions History)
# -----------------------------------------------------------------------------
def test_patient_crud_operations():
    """Tests PUT update, GET single, DELETE cascade, and patient sessions history."""
    # 1. Create Patient
    create_res = client.post(
        "/api/v1/patients",
        json={"name": "Bob Dylan", "age": 75, "gender": "Male", "contact": "bob@music.org"},
    )
    assert create_res.status_code == 201
    p_id = create_res.json()["id"]

    # 2. Get Single Patient
    get_res = client.get(f"/api/v1/patients/{p_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Bob Dylan"

    # 3. Update Patient
    put_res = client.put(
        f"/api/v1/patients/{p_id}",
        json={"name": "Robert Zimmerman", "age": 82, "gender": "Male", "contact": "bob@rollingstone.com"},
    )
    assert put_res.status_code == 200
    assert put_res.json()["name"] == "Robert Zimmerman"
    assert put_res.json()["age"] == 82

    # 4. Attach a session
    sess_res = client.post(
        "/api/v1/sessions",
        json={"patient_id": p_id, "notes": "Initial consultation"},
    )
    assert sess_res.status_code == 201

    # 5. Get Patient Sessions
    p_sess_res = client.get(f"/api/v1/patients/{p_id}/sessions")
    assert p_sess_res.status_code == 200
    assert len(p_sess_res.json()) >= 1

    # 6. Delete Patient (204 No Content)
    del_res = client.delete(f"/api/v1/patients/{p_id}")
    assert del_res.status_code == 204

    # 7. Confirm Patient Deleted (404)
    confirm_res = client.get(f"/api/v1/patients/{p_id}")
    assert confirm_res.status_code == 404


# -----------------------------------------------------------------------------
# 8. ReportLab PDF Report Generation
# -----------------------------------------------------------------------------
def test_reportlab_pdf_generation():
    """Verifies that GET /api/v1/sessions/{id}/report?format=pdf generates valid PDF bytes."""
    # 1. Create Patient & Session
    p_res = client.post(
        "/api/v1/patients",
        json={"name": "Clara Schumann", "age": 45, "gender": "Female", "contact": "clara@concerto.org"},
    )
    assert p_res.status_code == 201
    p_id = p_res.json()["id"]

    s_res = client.post("/api/v1/sessions", json={"patient_id": p_id, "notes": "PDF check"})
    assert s_res.status_code == 201
    s_id = s_res.json()["id"]

    # 2. Add vitals to session
    v_res = client.post(
        "/api/v1/submit_vitals",
        json={
            "session_id": s_id,
            "heart_rate_bpm": 68.0,
            "spo2_percent": 99.0,
            "temperature_celsius": 36.6,
        },
    )
    assert v_res.status_code == 200

    # 3. Request PDF report via format=pdf
    pdf_res = client.get(f"/api/v1/sessions/{s_id}/report?format=pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers.get("content-type") == "application/pdf"
    # Verify PDF magic bytes '%PDF'
    assert pdf_res.content.startswith(b"%PDF")
    assert len(pdf_res.content) > 500


# -----------------------------------------------------------------------------
# 9. Concurrency Stress Test
# -----------------------------------------------------------------------------
def test_concurrent_api_requests():
    """Executes 10 concurrent requests to verify thread safety and absence of race conditions."""
    def call_health():
        res = client.get("/api/v1/health")
        return res.status_code, res.json()

    def call_vitals():
        res = client.post(
            "/api/v1/submit_vitals",
            json={"heart_rate_bpm": 72.0, "spo2_percent": 98.0, "temperature_celsius": 36.8},
        )
        return res.status_code, res.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(call_health) for _ in range(5)] + [executor.submit(call_vitals) for _ in range(5)]
        results = [f.result() for f in futures]

    for status_code, body in results:
        assert status_code == 200
        assert body["status"] in ("healthy", "success")
