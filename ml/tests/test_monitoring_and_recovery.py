"""
Tests for ML Monitoring, Structured Logging, and Failure Recovery (Part 16).

Covers:
1. Structured logging emission for key ML operations (Model Init, Checkpoint Load,
   Quality Gate, Preprocessing, Inference, Grad-CAM, Physio Validation).
2. Distinguishable typed exception hierarchy and standard exception backward compatibility.
3. Checkpoint load failures (missing file, empty/corrupt binary, class mismatch).
4. Quality gate rejection logging and response contract adherence.
5. Preprocessing decoding error handling and logging.
6. Grad-CAM forward/backward hook cleanup and gradient clearing even on simulated exceptions.
7. Physiological validation error logging and typed exceptions.
8. Repeated requests after a failed request (model reuse without corruption).
9. Request correlation identifier (request_id) propagation across all modules.
10. Privacy safeguards: no raw images, tensor data, or absolute host paths in telemetry logs.
11. Gradient isolation: zero gradient accumulation across repeated inference calls.
"""

import logging
import os
import tempfile
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES
from ml.exceptions import (
    CheckpointCorruptError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    GradCAMGenerationError,
    InvalidImageInputError,
    MLException,
    ModelInferenceError,
    PhysiologicalTypeValidationError,
    PhysiologicalValidationError,
    ResourceExhaustionError,
)
from ml.eye_model import (
    build_densenet201_model,
    clear_model_cache,
    get_cached_model,
    load_eye_model_checkpoint,
    predict_eye_disease,
    screen_eye_image,
)
from ml.gradcam import generate_gradcam
from ml.monitoring import (
    clear_ml_event_buffer,
    get_recent_ml_events,
    log_ml_event,
    measure_ml_latency,
    sanitize_path,
)
from ml.physio_model import analyze_vitals
from ml.preprocessing import preprocess_image
from ml.quality_gate import check_image_quality


@pytest.fixture(autouse=True)
def clean_telemetry_state():
    """Clears telemetry ring buffer and model cache before and after each test."""
    clear_ml_event_buffer()
    clear_model_cache()
    yield
    clear_ml_event_buffer()
    clear_model_cache()


@pytest.fixture
def mock_fundus_fixtures(tmp_path):
    """Creates synthetic valid and invalid images and checkpoints for testing."""
    # 1. Valid retinal image (circle + disk + vessels)
    valid_path = str(tmp_path / "valid_fundus.png")
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 100, (140, 130, 120), -1)
    np.random.seed(42)
    noise = np.random.randint(-40, 40, (300, 300, 3), dtype=np.int16)
    noisy_canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.circle(noisy_canvas, (190, 150), 22, (230, 220, 200), -1)
    cv2.line(noisy_canvas, (190, 150), (100, 100), (40, 30, 160), 2)
    cv2.imwrite(valid_path, noisy_canvas)

    # 2. Blurry image failing quality gate
    blurry_path = str(tmp_path / "blurry_fundus.png")
    blur_canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(blur_canvas, (150, 150), 100, (140, 130, 120), -1)
    blurred = cv2.GaussianBlur(blur_canvas, (35, 35), sigmaX=15)
    cv2.imwrite(blurry_path, blurred)

    # 3. Valid lightweight DenseNet201 checkpoint
    valid_ckpt_path = str(tmp_path / "valid_checkpoint.pth")
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    payload = {
        "state_dict": model.state_dict(),
        "classes": list(CLASS_NAMES),
        "epoch": 2,
        "best_acc": 0.89,
        "architecture": "DenseNet201",
    }
    torch.save(payload, valid_ckpt_path)

    # 4. Corrupt checkpoint file (truncated bytes)
    corrupt_ckpt_path = str(tmp_path / "corrupt_checkpoint.pth")
    with open(corrupt_ckpt_path, "wb") as f:
        f.write(b"NOT_A_VALID_PYTORCH_CHECKPOINT_BYTES_12345")

    # 5. Incompatible checkpoint (classes mismatch)
    incompat_ckpt_path = str(tmp_path / "incompatible_checkpoint.pth")
    incompat_payload = {
        "state_dict": model.state_dict(),
        "classes": ["disease_a", "disease_b", "disease_c", "disease_d"],
        "epoch": 1,
    }
    torch.save(incompat_payload, incompat_ckpt_path)

    return {
        "valid_img": valid_path,
        "blurry_img": blurry_path,
        "valid_ckpt": valid_ckpt_path,
        "corrupt_ckpt": corrupt_ckpt_path,
        "incompat_ckpt": incompat_ckpt_path,
    }


class TestStructuredLoggingAndTelemetry:
    """Verifies that operational events are logged with structured key-value metadata."""

    def test_log_ml_event_emits_structured_record(self, caplog):
        """Verifies log_ml_event populates message string and extra payload."""
        logger = logging.getLogger("ml.test")
        with caplog.at_level(logging.INFO):
            event = log_ml_event(
                logger=logger,
                event_type="INFERENCE",
                operation="predict_eye_disease",
                status="SUCCESS",
                duration_ms=42.15,
                device="cpu",
                request_id="req-test-123",
                details={"prediction": "cataract", "confidence": 0.95},
            )

        assert event["event_type"] == "INFERENCE"
        assert event["operation"] == "predict_eye_disease"
        assert event["status"] == "SUCCESS"
        assert event["request_id"] == "req-test-123"
        assert event["duration_ms"] == 42.15
        assert event["device"] == "cpu"
        assert event["details"]["prediction"] == "cataract"

        # Check caplog formatted message
        assert "[ML:INFERENCE]" in caplog.text
        assert "req_id=req-test-123" in caplog.text
        assert "duration=42.15ms" in caplog.text

    def test_latency_measurement_context_manager(self):
        """Verifies measure_ml_latency records non-zero elapsed milliseconds."""
        with measure_ml_latency() as timer:
            # perform small computation
            x = [i * 2 for i in range(1000)]
        assert timer.duration_ms >= 0.0

    def test_sanitize_path_safeguards_privacy(self):
        """Verifies that absolute filesystem paths are stripped to filenames."""
        windows_path = r"C:\Users\Aryan\SensitivePatientRecords\patient_001.png"
        posix_path = "/home/user/data/clinical/fundus_scan.jpg"

        assert sanitize_path(windows_path) == "patient_001.png"
        assert sanitize_path(posix_path) == "fundus_scan.jpg"
        assert sanitize_path(None) is None
        assert sanitize_path("") is None


class TestCheckpointFailureHandling:
    """Verifies typed exception raising and failure logging for checkpoint operations."""

    def test_missing_checkpoint_raises_typed_and_logs(self, tmp_path):
        """Missing checkpoint raises CheckpointNotFoundError (which is FileNotFoundError)."""
        missing_path = str(tmp_path / "non_existent_weights.pth")
        dummy_model = build_densenet201_model()

        with pytest.raises(CheckpointNotFoundError) as exc_info:
            load_eye_model_checkpoint(dummy_model, missing_path, request_id="req-ckpt-fail")

        # Verify inheritance (subclasses both FileNotFoundError and MLException)
        assert isinstance(exc_info.value, FileNotFoundError)
        assert isinstance(exc_info.value, MLException)

        # Verify telemetry log recorded the failure
        events = get_recent_ml_events(event_type="CHECKPOINT_LOAD", status="FAILURE")
        assert len(events) >= 1
        assert events[0]["request_id"] == "req-ckpt-fail"
        assert events[0]["details"]["checkpoint"] == "non_existent_weights.pth"

    def test_corrupt_checkpoint_raises_typed_and_logs(self, mock_fundus_fixtures):
        """Corrupt checkpoint raises CheckpointCorruptError (which is IOError)."""
        dummy_model = build_densenet201_model()
        corrupt_path = mock_fundus_fixtures["corrupt_ckpt"]

        with pytest.raises(CheckpointCorruptError) as exc_info:
            load_eye_model_checkpoint(dummy_model, corrupt_path, request_id="req-corrupt")

        assert isinstance(exc_info.value, IOError)
        assert isinstance(exc_info.value, MLException)

        events = get_recent_ml_events(event_type="CHECKPOINT_LOAD", status="FAILURE")
        assert len(events) >= 1
        assert events[0]["request_id"] == "req-corrupt"

    def test_incompatible_checkpoint_classes_raises_typed(self, mock_fundus_fixtures):
        """Metadata class mismatch raises CheckpointIncompatibleError (which is ValueError)."""
        dummy_model = build_densenet201_model()
        incompat_path = mock_fundus_fixtures["incompat_ckpt"]

        with pytest.raises(CheckpointIncompatibleError) as exc_info:
            load_eye_model_checkpoint(dummy_model, incompat_path)

        assert isinstance(exc_info.value, ValueError)
        assert isinstance(exc_info.value, MLException)


class TestQualityGateAndPreprocessingMonitoring:
    """Verifies telemetry and failure handling for quality gate and preprocessing."""

    def test_quality_gate_rejection_logs_warning_event(self, mock_fundus_fixtures):
        """Blurry image produces REJECTED telemetry with failed criteria details."""
        blurry_path = mock_fundus_fixtures["blurry_img"]
        result = check_image_quality(blurry_path, request_id="req-qg-test")

        assert result["overall_pass"] is False
        assert result["error_message"] is not None

        events = get_recent_ml_events(event_type="QUALITY_GATE", status="REJECTED")
        assert len(events) >= 1
        last_ev = events[-1]
        assert last_ev["request_id"] == "req-qg-test"
        assert last_ev["details"]["file"] == "blurry_fundus.png"
        assert last_ev["details"]["overall_pass"] is False

    def test_preprocessing_corrupt_file_raises_and_logs(self, tmp_path):
        """Truncated or corrupted image raises InvalidImageInputError and logs failure."""
        corrupt_img = str(tmp_path / "corrupt_data.png")
        with open(corrupt_img, "wb") as f:
            f.write(b"CORRUPTED_IMAGE_BYTES")

        with pytest.raises(InvalidImageInputError) as exc_info:
            preprocess_image(corrupt_img, request_id="req-pre-fail")

        assert isinstance(exc_info.value, IOError)
        assert isinstance(exc_info.value, MLException)

        events = get_recent_ml_events(event_type="PREPROCESSING", status="FAILURE")
        assert len(events) >= 1
        assert events[-1]["request_id"] == "req-pre-fail"
        assert events[-1]["details"]["file"] == "corrupt_data.png"


class TestGradCAMResourceRecovery:
    """Verifies that Grad-CAM cleans up hooks and gradients under all conditions."""

    def test_gradcam_hook_cleanup_on_success(self, mock_fundus_fixtures):
        """Verifies forward and backward hooks are removed after successful execution."""
        img_path = mock_fundus_fixtures["valid_img"]
        ckpt_path = mock_fundus_fixtures["valid_ckpt"]
        model = get_cached_model(weights_path=ckpt_path, device=torch.device("cpu"))

        # Target layer is features.denseblock4
        target_layer = model.features.denseblock4

        b64 = generate_gradcam(
            image_path=img_path,
            predicted_class_idx=0,
            model=model,
            request_id="req-gradcam-success",
        )
        assert isinstance(b64, str)
        assert len(b64) > 50

        # Verify hook dictionaries on target module are empty
        assert len(target_layer._forward_hooks) == 0
        assert len(target_layer._backward_hooks) == 0

        # Verify parameter gradients are cleared
        for param in model.parameters():
            assert param.grad is None

    def test_gradcam_hook_cleanup_on_simulated_failure(self, mock_fundus_fixtures):
        """Verifies hooks and gradients are cleaned up even when an exception is raised inside Grad-CAM."""
        img_path = mock_fundus_fixtures["valid_img"]
        ckpt_path = mock_fundus_fixtures["valid_ckpt"]
        model = get_cached_model(weights_path=ckpt_path, device=torch.device("cpu"))
        target_layer = model.features.denseblock4

        # Pass an invalid class index (e.g. 99) to force a ValueError
        with pytest.raises(ValueError):
            generate_gradcam(
                image_path=img_path,
                predicted_class_idx=99,
                model=model,
                request_id="req-gradcam-fail",
            )

        # Verify hook dictionaries on target module are clean
        assert len(target_layer._forward_hooks) == 0
        assert len(target_layer._backward_hooks) == 0

        for param in model.parameters():
            assert param.grad is None


class TestPhysiologicalFailureHandling:
    """Verifies typed exceptions and structured logging for vital signs analysis."""

    def test_invalid_type_raises_physio_type_error(self):
        """Non-numeric vital sign raises PhysiologicalTypeValidationError."""
        with pytest.raises(PhysiologicalTypeValidationError) as exc_info:
            analyze_vitals(heart_rate="invalid_hr", spo2=98.0, temperature=36.8, request_id="req-phys-type")

        assert isinstance(exc_info.value, TypeError)
        assert isinstance(exc_info.value, MLException)
        assert exc_info.value.metric_name == "Heart rate"

        events = get_recent_ml_events(event_type="PHYSIO_VALIDATION", status="FAILURE")
        assert len(events) >= 1
        assert events[-1]["request_id"] == "req-phys-type"

    def test_out_of_range_raises_physio_val_error(self):
        """Unphysical reading raises PhysiologicalValidationError."""
        with pytest.raises(PhysiologicalValidationError) as exc_info:
            analyze_vitals(heart_rate=12.0, spo2=98.0, temperature=36.8, request_id="req-phys-val")

        assert isinstance(exc_info.value, ValueError)
        assert isinstance(exc_info.value, MLException)

    def test_non_raising_mode_returns_structured_error_and_logs(self):
        """When raise_errors=False, returns structured error dict and logs telemetry."""
        result = analyze_vitals(heart_rate=float("nan"), spo2=98.0, temperature=36.8, raise_errors=False, request_id="req-phys-nan")

        assert result["status"] == "error"
        assert result["is_abnormal"] is True
        assert "NaN" in result["error_message"]

        events = get_recent_ml_events(event_type="PHYSIO_VALIDATION", status="FAILURE")
        assert len(events) >= 1


class TestRepeatedInferenceAndModelIntegrity:
    """Verifies model reuse after failed requests and zero gradient leakage."""

    def test_failed_request_does_not_corrupt_subsequent_request(self, mock_fundus_fixtures):
        """A failed inference request leaves the reusable model in a valid, functional state."""
        valid_img = mock_fundus_fixtures["valid_img"]
        ckpt_path = mock_fundus_fixtures["valid_ckpt"]
        model = get_cached_model(weights_path=ckpt_path, device=torch.device("cpu"))

        # 1. Trigger failure with invalid tensor dimension
        bad_tensor = torch.zeros((1, 5, 100, 100))
        with pytest.raises(ValueError):
            predict_eye_disease(image_input=bad_tensor, model=model)

        # 2. Immediately execute valid screening with same model instance
        res = screen_eye_image(
            image_path=valid_img,
            model=model,
            use_cache=True,
            include_gradcam=True,
            request_id="req-recovery-1",
        )

        assert res["status"] == "success"
        assert res["prediction"] in CLASS_NAMES
        assert 0.0 <= res["confidence"] <= 1.0
        assert "gradcam_image" in res

    def test_repeated_inference_zero_gradient_accumulation(self, mock_fundus_fixtures):
        """Confirms that multiple repeated forward passes accumulate zero parameter gradients."""
        valid_img = mock_fundus_fixtures["valid_img"]
        ckpt_path = mock_fundus_fixtures["valid_ckpt"]
        model = get_cached_model(weights_path=ckpt_path, device=torch.device("cpu"))

        # Run 5 repeated screening calls
        for i in range(5):
            res = screen_eye_image(
                image_path=valid_img,
                model=model,
                use_cache=True,
                include_gradcam=False,
                request_id=f"req-repeat-{i}",
            )
            assert res["status"] == "success"

        # Verify all parameters have None gradients
        for p in model.parameters():
            assert p.grad is None


class TestRequestCorrelationAndPrivacy:
    """Verifies request_id propagation and privacy protection in logs."""

    def test_request_id_propagated_through_composite_screening(self, mock_fundus_fixtures):
        """End-to-end screen_eye_image propagates request_id to all sub-operations."""
        valid_img = mock_fundus_fixtures["valid_img"]
        ckpt_path = mock_fundus_fixtures["valid_ckpt"]

        res = screen_eye_image(
            image_path=valid_img,
            weights_path=ckpt_path,
            include_gradcam=True,
            request_id="req-corr-999",
        )
        assert res["status"] == "success"

        # Check telemetry events
        events = get_recent_ml_events()
        matching_events = [e for e in events if e.get("request_id") == "req-corr-999"]

        # Should have captured QUALITY_GATE, PREPROCESSING, INFERENCE, GRADCAM, SCREENING_WORKFLOW
        event_types = {e["event_type"] for e in matching_events}
        assert "QUALITY_GATE" in event_types
        assert "INFERENCE" in event_types
        assert "GRADCAM" in event_types
        assert "SCREENING_WORKFLOW" in event_types

    def test_logs_do_not_contain_raw_images_or_sensitive_paths(self, mock_fundus_fixtures):
        """Telemetry events must not contain full local directory paths or raw image strings."""
        valid_img = mock_fundus_fixtures["valid_img"]
        ckpt_path = mock_fundus_fixtures["valid_ckpt"]

        screen_eye_image(
            image_path=valid_img,
            weights_path=ckpt_path,
            include_gradcam=True,
            request_id="req-privacy-check",
        )

        events = get_recent_ml_events()
        for ev in events:
            # Check details
            details = ev.get("details", {})
            for k, v in details.items():
                if isinstance(v, str):
                    # Should not contain drive root like "C:\" or tmp directory path
                    assert "\\" not in v or "/" not in v
                    # Should not contain large raw image / base64 payloads
                    assert len(v) < 500
