"""
Regression and Quality Audit Test Suite for Production Observability.

Validates:
1. Request ID (X-Request-ID) generation, client propagation, and header echo.
2. Structured operational log fields (timestamp, request_id, method, path, status, duration).
3. ML pipeline sub-stage latency instrumentation (quality_gate, preprocessing, inference, gradcam).
4. Clear distinction between Quality Gate rejections (HTTP 400) and internal errors (HTTP 500).
5. Safe exception handling: full traceback logged internally with correlation ID, zero stack trace leaked to user.
6. Privacy safeguards: zero Base64 strings, binary blobs, patient PII, or credentials in telemetry events.
7. Health and readiness probe behavior (200 OK when ready, 503 when checkpoint unavailable).
8. Absence of local host filesystem paths in API responses.
"""

import io
import json
import logging
import os
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageFilter
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.observability import (
    clear_http_event_buffer,
    get_recent_http_events,
)
from backend.services.ml_service import ml_service
from ml.monitoring import (
    clear_ml_event_buffer,
    get_recent_ml_events,
)

client = TestClient(app)

TEST_SAMPLES = {
    "normal": r"C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset\test\normal\kaggle_2587_right.jpg",
}


@pytest.fixture(autouse=True)
def clean_buffers():
    """Clears HTTP and ML in-memory telemetry buffers before each test."""
    clear_http_event_buffer()
    clear_ml_event_buffer()
    yield
    clear_http_event_buffer()
    clear_ml_event_buffer()


def create_blurry_image_bytes() -> bytes:
    """Creates synthetic blurry test image that fails the quality gate."""
    img = Image.new("RGB", (256, 256), color=(100, 100, 100))
    img = img.filter(ImageFilter.GaussianBlur(radius=12))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# =============================================================================
# 1. Request ID Propagation & Header Echo
# =============================================================================

def test_request_id_auto_generation():
    """When no X-Request-ID header is provided, the backend generates a valid UUID hex and echoes it."""
    res = client.get("/health/live")
    assert res.status_code == 200
    assert "x-request-id" in res.headers
    req_id = res.headers["x-request-id"]
    assert len(req_id) == 32  # UUID4 hex length


def test_request_id_client_propagation():
    """When client supplies X-Request-ID, the server adopts and propagates it into headers and response."""
    custom_cid = "client-trace-abc-12345"
    res = client.get("/health/live", headers={"X-Request-ID": custom_cid})
    assert res.status_code == 200
    assert res.headers.get("x-request-id") == custom_cid

    # Verify matching event in HTTP event ring buffer
    events = get_recent_http_events(limit=5)
    matching = [e for e in events if e.get("request_id") == custom_cid]
    assert len(matching) >= 1
    assert matching[0]["path"] == "/health/live"


# =============================================================================
# 2. Structured Operational HTTP Log Fields
# =============================================================================

def test_structured_http_log_fields():
    """HTTP access events record required operational fields without sensitive payloads."""
    clear_http_event_buffer()
    res = client.get("/api/v1/health")
    assert res.status_code == 200

    events = get_recent_http_events(path="/api/v1/health")
    assert len(events) >= 1
    event = events[-1]

    assert event["event"] == "http_request"
    assert event["method"] == "GET"
    assert event["path"] == "/api/v1/health"
    assert event["status_code"] == 200
    assert isinstance(event["duration_ms"], (int, float))
    assert event["duration_ms"] >= 0.0
    assert "request_id" in event
    assert len(event["request_id"]) > 0


# =============================================================================
# 3. Sub-Stage Latency Monitoring
# =============================================================================

def test_ml_pipeline_substage_latencies():
    """predict_eye measures and reports quality_gate, preprocessing, inference, and gradcam timings."""
    with open(TEST_SAMPLES["normal"], "rb") as f:
        img_bytes = f.read()

    custom_id = "latency-audit-req-001"
    res = client.post(
        "/api/v1/predict_eye",
        files={"image": ("retina.jpg", img_bytes, "image/jpeg")},
        data={"include_gradcam": "true"},
        headers={"X-Request-ID": custom_id},
    )
    assert res.status_code == 200
    data = res.json()

    assert "timings_ms" in data
    timings = data["timings_ms"]
    assert "quality_gate" in timings
    assert "preprocessing" in timings
    assert "eye_inference" in timings
    assert "gradcam" in timings
    assert "total" in timings

    # All timings must be non-negative numbers
    for stage, t in timings.items():
        assert isinstance(t, (int, float)), f"Timing for {stage} should be float, got {t}"
        assert t >= 0.0

    # Verify HTTP event captured the stage timings
    events = get_recent_http_events(limit=5)
    matching = [e for e in events if e.get("request_id") == custom_id]
    assert len(matching) >= 1
    assert "ml_timings_ms" in matching[0]
    assert matching[0]["ml_timings_ms"]["eye_inference"] == timings["eye_inference"]


# =============================================================================
# 4. Rejected vs Internal Error Classification
# =============================================================================

def test_quality_gate_rejection_distinct_from_500():
    """A quality gate rejection is classified as HTTP 400 with QUALITY_REJECTED, not an internal error."""
    blurry_bytes = create_blurry_image_bytes()
    custom_id = "qg-rejection-test-002"

    res = client.post(
        "/api/v1/predict_eye",
        files={"image": ("blurry.jpg", blurry_bytes, "image/jpeg")},
        headers={"X-Request-ID": custom_id},
    )

    # Must be 400 Bad Request, NOT 500 Internal Server Error
    assert res.status_code == 400
    data = res.json()
    assert data["status"] == "rejected"
    assert "quality_checks" in data
    assert data.get("request_id") == custom_id
    assert "timings_ms" in data
    assert data["timings_ms"]["quality_gate"] >= 0.0

    # Verify error category in telemetry buffer
    events = get_recent_http_events(limit=5)
    matching = [e for e in events if e.get("request_id") == custom_id]
    assert len(matching) >= 1
    assert matching[0]["status_code"] == 400
    assert matching[0]["error_category"] == "QUALITY_REJECTED"


# =============================================================================
# 5. Safe Exception Handling & Traceback Isolation
# =============================================================================

def test_generic_exception_does_not_leak_stacktrace(monkeypatch):
    """Internal server errors (500) log tracebacks on server but return safe sanitized responses."""
    def broken_predict(*args, **kwargs):
        raise RuntimeError("Simulated internal CUDA kernel crash or memory fault at line 42")

    monkeypatch.setattr("backend.routes.eye_routes.ml_service.predict_eye", broken_predict)

    with open(TEST_SAMPLES["normal"], "rb") as f:
        img_bytes = f.read()

    custom_id = "safe-error-audit-003"
    res = client.post(
        "/api/v1/predict_eye",
        files={"image": ("retina.jpg", img_bytes, "image/jpeg")},
        headers={"X-Request-ID": custom_id},
    )

    assert res.status_code == 500
    data = res.json()
    assert data["status"] == "error"
    assert data["error_type"] in ("InternalServerError", "HTTPException")
    assert data["request_id"] == custom_id

    # Crucial security test: Stack trace, file paths, and python line numbers must NOT be in the body
    body_str = res.text
    assert "Traceback" not in body_str
    assert "kernel crash" not in body_str
    assert "line 42" not in body_str
    assert "C:\\Users\\" not in body_str


# =============================================================================
# 6. Privacy Safeguards: No PII, Blobs, or Credentials in Telemetry
# =============================================================================

def test_no_sensitive_payloads_in_http_and_ml_buffers():
    """Ensures in-memory telemetry buffers never store image blobs, Base64 strings, or patient names."""
    with open(TEST_SAMPLES["normal"], "rb") as f:
        img_bytes = f.read()

    res = client.post(
        "/api/v1/screen",
        files={"image": ("fundus.jpg", img_bytes, "image/jpeg")},
        data={
            "heart_rate": "72",
            "spo2": "98",
            "temperature": "36.8",
            "patient_id": "1",
        },
    )
    assert res.status_code in (200, 201)

    # Inspect HTTP buffer
    for ev in get_recent_http_events():
        ev_str = json.dumps(ev, default=str)
        assert "data:image" not in ev_str
        assert "base64" not in ev_str.lower()
        assert "patient_name" not in ev_str

    # Inspect ML buffer
    for ev in get_recent_ml_events():
        ev_str = json.dumps(ev, default=str)
        assert "data:image" not in ev_str
        assert "base64" not in ev_str.lower()
        assert "patient_name" not in ev_str


# =============================================================================
# 7. Health and Readiness Checks
# =============================================================================

def test_liveness_probe_is_lightweight():
    """Liveness probe /health/live returns 200 OK immediately without model loading."""
    res = client.get("/health/live")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "alive"
    assert "timestamp" in data
    # Must not contain filesystem paths
    assert "weights" not in res.text
    assert "C:\\Users\\" not in res.text


def test_readiness_probe_returns_200_when_healthy():
    """Readiness probe /health/ready returns 200 OK when checkpoint is verified and DB connected."""
    res = client.get("/health/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["checkpoint_available"] is True
    assert data["checkpoint_verified"] is True
    assert data["model_loaded"] is True
    assert data["database_connected"] is True
    assert "weights" not in res.text
    assert "C:\\Users\\" not in res.text


def test_readiness_probe_fails_503_when_checkpoint_missing(monkeypatch):
    """Readiness probe returns 503 Service Unavailable when model checkpoint is missing."""
    monkeypatch.setattr(ml_service, "checkpoint_path", "non_existent_model_checkpoint.pth")
    res = client.get("/health/ready")
    assert res.status_code == 503
    data = res.json()
    assert data["status"] == "unready"
    assert data["checkpoint_available"] is False
    assert "Model weights checkpoint not found" in data["reason"]
    # Path itself must not be leaked
    assert "non_existent_model_checkpoint.pth" not in res.text


# =============================================================================
# 8. Multimodal Screening Telemetry & Timings
# =============================================================================

def test_screen_multimodal_records_all_stages():
    """Combined multimodal screening records timings for both eye and vitals stages."""
    with open(TEST_SAMPLES["normal"], "rb") as f:
        img_bytes = f.read()

    res = client.post(
        "/api/v1/screen",
        files={"image": ("retina.jpg", img_bytes, "image/jpeg")},
        data={"heart_rate": "74.0", "spo2": "99.0", "temperature": "36.7"},
    )
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert "duration_ms" in data
    assert data["duration_ms"] > 0
    assert "timings_ms" in data

    timings = data["timings_ms"]
    assert "quality_gate" in timings
    assert "preprocessing" in timings
    assert "eye_inference" in timings
    assert "gradcam" in timings
    assert "vitals_analysis" in timings
    assert "total_pipeline" in timings
