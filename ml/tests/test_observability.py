"""
Tests for ML Observability, Structured Logging, and Telemetry Infrastructure.

Validates:
1. configure_logging idempotency, log level setting, and handler deduplication.
2. Log file creation, parent directory auto-creation, and persistent file emission.
3. Structured text formatter (MLStructuredFormatter) and JSON formatter (MLJsonFormatter).
4. Strict privacy sanitization: path basenames, Base64 stripping, credentials/PII masking,
   and physiological raw value redaction.
5. End-to-end pipeline events: 'screening.started', 'screening.completed', request_id propagation,
   and sub-stage monotonic latency timings (timings_ms).
6. Error code enumeration (ErrorCode) on failure paths: quality gate rejection, vitals error,
   and isolated Grad-CAM failure.
7. Non-crashing logging guarantee: simulated logging failures do not crash inference.
"""

import json
import logging
import os
import shutil
import tempfile
import cv2
import numpy as np
import pytest

from ml.monitoring import (
    ErrorCode,
    MLJsonFormatter,
    MLStructuredFormatter,
    clear_ml_event_buffer,
    configure_logging,
    get_recent_ml_events,
    log_ml_event,
    ml_logger,
    sanitize_details,
    sanitize_path,
)
from ml.screening_pipeline import run_screening


@pytest.fixture(autouse=True)
def reset_logging_and_buffer():
    """Ensure clean logger state and event buffer before and after each test."""
    clear_ml_event_buffer()
    # Reset logger handlers
    for h in list(ml_logger.handlers):
        ml_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    ml_logger.setLevel(logging.INFO)
    yield
    clear_ml_event_buffer()
    for h in list(ml_logger.handlers):
        ml_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for test log files and clean up after."""
    d = tempfile.mkdtemp(prefix="ml_test_logs_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_valid_fundus_image_path(temp_log_dir):
    """Create and save a realistic synthetic valid retinal image to disk."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 230, (30, 80, 180), -1)  # Orange-red retinal disk
    cv2.circle(img, (200, 256), 35, (100, 200, 240), -1)  # Optic disc
    cv2.circle(img, (320, 256), 25, (15, 30, 80), -1)    # Macula
    # Retinal vessel lines
    cv2.line(img, (200, 256), (100, 120), (10, 20, 100), 2)
    cv2.line(img, (200, 256), (120, 390), (10, 20, 100), 2)
    cv2.line(img, (200, 256), (300, 150), (10, 20, 100), 2)
    file_path = os.path.join(temp_log_dir, "valid_fundus.png")
    cv2.imwrite(file_path, img)
    return file_path


# =============================================================================
# 1. Logging Configuration & Idempotency Tests
# =============================================================================

def test_configure_logging_idempotent_no_duplicate_handlers():
    """Configuring logging multiple times must not accumulate duplicate handlers."""
    configure_logging(level="INFO", log_to_console=True)
    initial_handler_count = len(ml_logger.handlers)
    assert initial_handler_count >= 1

    # Reconfigure multiple times
    configure_logging(level="DEBUG", log_to_console=True)
    configure_logging(level="WARNING", log_to_console=True)

    assert len(ml_logger.handlers) == initial_handler_count
    assert ml_logger.level == logging.WARNING


def test_configure_logging_creates_file_and_directories(temp_log_dir):
    """File logging automatically creates missing nested directories and writes logs."""
    nested_log_path = os.path.join(temp_log_dir, "nested", "sub", "test_ml.log")
    assert not os.path.exists(nested_log_path)

    configure_logging(level="INFO", log_to_console=False, log_file=nested_log_path)
    assert os.path.exists(os.path.dirname(nested_log_path))

    log_ml_event(
        ml_logger,
        event_type="TEST",
        operation="file_check",
        status="SUCCESS",
        stage="TESTING",
        event_name="test.event",
    )

    # Flush handlers
    for h in ml_logger.handlers:
        h.flush()

    assert os.path.exists(nested_log_path)
    with open(nested_log_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "event=test.event" in content
    assert "status=SUCCESS" in content


# =============================================================================
# 2. Structured & JSON Formatters
# =============================================================================

def test_structured_text_formatter_fields():
    """MLStructuredFormatter formats standard key operational fields cleanly."""
    formatter = MLStructuredFormatter()
    record = logging.LogRecord(
        name="ml",
        level=logging.INFO,
        pathname="ml/test.py",
        lineno=10,
        msg="test message",
        args=(),
        exc_info=None,
    )
    setattr(
        record,
        "ml_event",
        {
            "stage": "inference",
            "event_name": "eye.inference.completed",
            "status": "SUCCESS",
            "request_id": "req-xyz-123",
            "duration_ms": 14.52,
            "error_code": None,
            "device": "cuda:0",
            "details": {"predicted_class": "normal", "confidence": 0.98},
        },
    )

    formatted = formatter.format(record)
    assert "[INFO]" in formatted
    assert "[ML:inference]" in formatted
    assert "event=eye.inference.completed" in formatted
    assert "status=SUCCESS" in formatted
    assert "req_id=req-xyz-123" in formatted
    assert "duration=14.52ms" in formatted
    assert "dev=cuda:0" in formatted
    assert "predicted_class=normal" in formatted


def test_json_formatter_valid_json_lines(temp_log_dir):
    """MLJsonFormatter produces valid JSON objects parseable with json.loads."""
    json_log_path = os.path.join(temp_log_dir, "events.jsonl")
    configure_logging(
        level="INFO", log_to_console=False, log_file=json_log_path, json_format=True
    )

    log_ml_event(
        ml_logger,
        event_type="SCREENING",
        operation="run_screening",
        status="SUCCESS",
        stage="pipeline",
        event_name="screening.completed",
        duration_ms=45.2,
        request_id="req-999",
        details={"status": "success", "predicted_class": "glaucoma"},
    )

    for h in ml_logger.handlers:
        h.flush()

    with open(json_log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) >= 1
    record_data = json.loads(lines[-1].strip())
    assert record_data["event_name"] == "screening.completed"
    assert record_data["status"] == "SUCCESS"
    assert record_data["request_id"] == "req-999"
    assert record_data["duration_ms"] == 45.2
    assert record_data["stage"] == "pipeline"


# =============================================================================
# 3. Privacy Safeguards & Sanitization Tests
# =============================================================================

def test_sanitize_path_strips_directory_and_preserves_basename():
    """Host filesystem paths must be reduced to base filename to prevent path leaks."""
    assert sanitize_path(r"C:\Users\Aryan\Documents\patient_fundus.png") == "patient_fundus.png"
    assert sanitize_path("/var/medical/records/patient123/scan.jpg") == "scan.jpg"
    assert sanitize_path(None) is None
    assert sanitize_path("") is None


def test_sanitize_details_masks_base64_and_pii():
    """Sanitizer masks Base64 images, passwords, API tokens, and long raw payloads."""
    raw_details = {
        "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
        "long_blob": "A" * 200,
        "api_key": "secret_key_12345",
        "patient_name": "John Doe",
        "safe_metric": 0.95,
    }
    cleaned = sanitize_details(raw_details)

    assert cleaned["image_data"] == "[BLOB_DATA_MASKED]"
    assert cleaned["long_blob"] == "[BLOB_DATA_MASKED]"
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["patient_name"] == "[REDACTED_PII]"
    assert cleaned["safe_metric"] == 0.95


def test_sanitize_details_masks_raw_physiological_values():
    """Raw patient vital signs are masked while triage status strings are preserved."""
    raw_details = {
        "heart_rate": 78,
        "spo2": 98.5,
        "temperature": 37.1,
        "heart_rate_bpm": 80,
        "vitals_status": "NORMAL",
    }
    cleaned = sanitize_details(raw_details)

    assert cleaned["heart_rate"] == "[PHYSIO_DATA_MASKED]"
    assert cleaned["spo2"] == "[PHYSIO_DATA_MASKED]"
    assert cleaned["temperature"] == "[PHYSIO_DATA_MASKED]"
    assert cleaned["heart_rate_bpm"] == "[PHYSIO_DATA_MASKED]"
    # Qualitative category status is preserved
    assert cleaned["vitals_status"] == "NORMAL"


# =============================================================================
# 4. End-to-End Pipeline Observability & Latency Timings
# =============================================================================

def test_screening_pipeline_emits_telemetry_and_populates_timings(sample_valid_fundus_image_path):
    """Pipeline emits start/complete events, propagates request_id, and records stage latencies."""
    clear_ml_event_buffer()
    custom_req_id = "req-obs-integration-test-01"

    response = run_screening(
        image_path=sample_valid_fundus_image_path,
        heart_rate=72.0,
        spo2=98.0,
        temperature=36.8,
        request_id=custom_req_id,
        include_gradcam=True,
    )

    # 1. Timings must be present in response
    assert "timings_ms" in response
    timings = response["timings_ms"]
    assert "total_pipeline" in timings
    assert "quality_gate" in timings
    assert "preprocessing" in timings
    assert "eye_inference" in timings
    assert "vitals_analysis" in timings

    # All measured timings must be non-negative floating point numbers
    for k, v in timings.items():
        assert isinstance(v, (int, float))
        assert v >= 0.0

    # 2. Check emitted event logs
    events = get_recent_ml_events()
    event_names = [e.get("event_name") for e in events]

    assert "screening.started" in event_names
    assert "screening.completed" in event_names

    # Check request correlation ID propagation
    matching_req_events = [e for e in events if e.get("request_id") == custom_req_id]
    assert len(matching_req_events) >= 2


def test_quality_gate_rejection_logs_error_code(temp_log_dir):
    """When an image fails the quality gate, ErrorCode.QUALITY_REJECTED is logged."""
    clear_ml_event_buffer()
    # Completely black underexposed image
    black_img = np.zeros((200, 200, 3), dtype=np.uint8)
    black_path = os.path.join(temp_log_dir, "black_img.png")
    cv2.imwrite(black_path, black_img)

    response = run_screening(
        image_path=black_path,
        request_id="req-qg-fail-02",
    )

    assert response["status"] == "rejected"
    assert response["eye_screening"]["status"] == "rejected"

    events = get_recent_ml_events()
    qg_rejected_events = [
        e for e in events if e.get("error_code") == ErrorCode.QUALITY_REJECTED
    ]
    assert len(qg_rejected_events) >= 1
    assert qg_rejected_events[0]["status"] == "QUALITY_REJECTED"


def test_invalid_vitals_logs_error_code(sample_valid_fundus_image_path):
    """When vital signs are out of clinical range, ErrorCode.VITALS_VALIDATION is logged."""
    clear_ml_event_buffer()

    response = run_screening(
        image_path=sample_valid_fundus_image_path,
        heart_rate=350.0,  # Impossible heart rate
        spo2=98.0,
        temperature=37.0,
        request_id="req-vitals-err-03",
    )

    assert response["status"] == "partial"  # Eye succeeded, vitals errored
    assert response["vitals_screening"]["status"] == "error"

    events = get_recent_ml_events()
    vitals_err_events = [
        e for e in events if e.get("error_code") == ErrorCode.VITALS_VALIDATION
    ]
    assert len(vitals_err_events) >= 1


# =============================================================================
# 5. Non-Crashing Logger Guarantee
# =============================================================================

def test_logger_exception_does_not_crash_pipeline(monkeypatch, sample_valid_fundus_image_path):
    """If the underlying logger raises an exception, the ML pipeline must continue smoothly."""
    def broken_log(*args, **kwargs):
        raise OSError("Disk full / pipe closed simulation")

    monkeypatch.setattr(ml_logger, "log", broken_log)

    # Should not raise exception
    response = run_screening(
        image_path=sample_valid_fundus_image_path,
        heart_rate=70.0,
        spo2=98.0,
        temperature=36.6,
    )

    assert response["status"] == "success"
    assert response["eye_screening"]["prediction"] is not None


def test_gradcam_failure_logs_error_code_and_preserves_prediction(monkeypatch, sample_valid_fundus_image_path):
    """When Grad-CAM fails, ErrorCode.GRADCAM_FAILED is logged and inference result is preserved."""
    clear_ml_event_buffer()

    def broken_gradcam(*args, **kwargs):
        raise RuntimeError("Simulated Grad-CAM OOM or target layer hook failure")

    monkeypatch.setattr("ml.screening_pipeline.generate_gradcam", broken_gradcam)

    response = run_screening(
        image_path=sample_valid_fundus_image_path,
        heart_rate=72.0,
        spo2=98.0,
        temperature=36.8,
        include_gradcam=True,
    )

    # Screening must still succeed
    assert response["status"] == "success"
    assert response["eye_screening"]["prediction"] is not None
    assert response["eye_screening"]["gradcam_image"] is None
    assert "Simulated Grad-CAM" in response["eye_screening"]["gradcam_error"]

    events = get_recent_ml_events()
    gradcam_err_events = [
        e for e in events if e.get("error_code") == ErrorCode.GRADCAM_FAILED
    ]
    assert len(gradcam_err_events) >= 1

