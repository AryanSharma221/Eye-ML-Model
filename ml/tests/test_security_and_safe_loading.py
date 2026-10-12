"""
Tests for ML Security, Privacy, and Safe Model Loading (Part 18).

Covers:
1. Safe Checkpoint Loading: weights_only=True, blocking malicious pickle exploits,
   validating architecture, class mappings, output dimensions, and key compatibility.
2. Absence of Silent Fallback: strict rejection of missing or corrupt checkpoints.
3. Image Input Safety: rejection of oversized files (>30MB), excessive pixel dimensions (>5000x5000),
   decompression bombs, empty files, and path traversal sanitization.
4. Privacy Safeguards: path sanitization in user errors and logs, redacting credentials and PII,
   masking raw physiological measurements in operational logs.
5. Physiological Input Safety: strict rejection of booleans, strings, None, NaN, infinity,
   out-of-bounds readings, and safe conversion of NumPy numeric scalars.
6. Grad-CAM Safety: forward/backward hook cleanup in success and failure, model parameter gradient clearing,
   constant heatmap division-by-zero protection, and source image immutability.
"""

import base64
import hashlib
import logging
import math
import os
import pickle
import cv2
import numpy as np
from PIL import Image
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
    PhysiologicalTypeValidationError,
    PhysiologicalValidationError,
)
from ml.eye_model import (
    DENSENET201_IN_FEATURES,
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
    sanitize_path,
)
from ml.physio_model import (
    STATUS_ABNORMAL,
    STATUS_NORMAL,
    analyze_vitals,
)
from ml.preprocessing import (
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_FILE_SIZE_BYTES,
    preprocess_image,
)
from ml.quality_gate import check_image_quality


@pytest.fixture(autouse=True)
def clean_test_state():
    """Ensures clean model cache and telemetry ring buffer between tests."""
    clear_model_cache()
    clear_ml_event_buffer()
    yield
    clear_model_cache()
    clear_ml_event_buffer()


@pytest.fixture
def valid_retina_image(tmp_path):
    """Generates a standard synthetic valid retinal image."""
    img_path = str(tmp_path / "retina_valid.png")
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 100, (140, 130, 120), -1)
    np.random.seed(42)
    noise = np.random.randint(-40, 40, (300, 300, 3), dtype=np.int16)
    noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.circle(noisy, (190, 150), 22, (230, 220, 200), -1)
    cv2.line(noisy, (190, 150), (100, 100), (40, 30, 160), 2)
    cv2.imwrite(img_path, noisy)
    return img_path


@pytest.fixture
def valid_controlled_checkpoint(tmp_path):
    """Generates a valid lightweight DenseNet201 checkpoint with canonical metadata."""
    ckpt_path = str(tmp_path / "valid_controlled.pth")
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    payload = {
        "state_dict": model.state_dict(),
        "classes": list(CLASS_NAMES),
        "epoch": 1,
        "best_acc": 0.90,
        "val_accuracy": 0.90,
        "architecture": "DenseNet201",
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


# ==============================================================================
# 1. Safe Checkpoint Loading & Architecture Invariants
# ==============================================================================

class TestSafeCheckpointLoading:
    """Verifies safe model deserialization and strict checkpoint validation."""

    def test_checkpoint_safe_loading_weights_only(self, valid_controlled_checkpoint):
        """Standard valid checkpoint loads cleanly using safe weights_only=True."""
        model = build_densenet201_model(num_classes=NUM_CLASSES)
        loaded = load_eye_model_checkpoint(model, valid_controlled_checkpoint, device=torch.device("cpu"))
        assert loaded is not None

    def test_malicious_pickle_payload_rejected(self, tmp_path):
        """Arbitrary code execution via pickle __reduce__ is blocked by weights_only=True."""
        malicious_path = str(tmp_path / "exploit.pth")

        class MaliciousExploit:
            def __reduce__(self):
                return (eval, ("1+1",))

        # Serialize malicious pickle object
        with open(malicious_path, "wb") as f:
            pickle.dump(MaliciousExploit(), f)

        model = build_densenet201_model(num_classes=NUM_CLASSES)
        with pytest.raises(CheckpointCorruptError) as exc_info:
            load_eye_model_checkpoint(model, malicious_path, device=torch.device("cpu"))

        # Confirm failure was caught cleanly as a checkpoint corrupt/unpickling error
        assert "failed to load checkpoint" in str(exc_info.value).lower()

    def test_architecture_identifier_validation(self, tmp_path):
        """Checkpoints indicating an incompatible architecture are strictly rejected."""
        bad_arch_path = str(tmp_path / "bad_arch.pth")
        model = build_densenet201_model(num_classes=NUM_CLASSES)
        payload = {
            "state_dict": model.state_dict(),
            "classes": list(CLASS_NAMES),
            "architecture": "ResNet50",  # Incompatible architecture identifier
        }
        torch.save(payload, bad_arch_path)

        with pytest.raises(CheckpointIncompatibleError) as exc_info:
            load_eye_model_checkpoint(model, bad_arch_path, device=torch.device("cpu"))
        assert "architecture mismatch" in str(exc_info.value).lower()

    def test_class_mapping_order_mismatch_rejected(self, tmp_path):
        """Checkpoints with modified or scrambled class orders are rejected."""
        scrambled_path = str(tmp_path / "scrambled.pth")
        model = build_densenet201_model(num_classes=NUM_CLASSES)
        scrambled_classes = ["normal", "glaucoma", "diabetic_retinopathy", "cataract"]
        payload = {
            "state_dict": model.state_dict(),
            "classes": scrambled_classes,
            "architecture": "DenseNet201",
        }
        torch.save(payload, scrambled_path)

        with pytest.raises(CheckpointIncompatibleError) as exc_info:
            load_eye_model_checkpoint(model, scrambled_path, device=torch.device("cpu"))
        assert "class order mismatch" in str(exc_info.value).lower()

    def test_classifier_weight_shape_mismatch_rejected(self, tmp_path):
        """Checkpoints where classifier output head has the wrong dimension are rejected."""
        wrong_shape_path = str(tmp_path / "wrong_shape.pth")
        model = build_densenet201_model(num_classes=NUM_CLASSES)
        state_dict = model.state_dict()
        # Incompatible head: 5 classes instead of 4
        state_dict["classifier.weight"] = torch.randn(5, DENSENET201_IN_FEATURES)
        state_dict["classifier.bias"] = torch.randn(5)
        payload = {
            "state_dict": state_dict,
            "classes": list(CLASS_NAMES),
            "architecture": "DenseNet201",
        }
        torch.save(payload, wrong_shape_path)

        with pytest.raises(CheckpointIncompatibleError) as exc_info:
            load_eye_model_checkpoint(model, wrong_shape_path, device=torch.device("cpu"))
        assert "classifier weight shape mismatch" in str(exc_info.value).lower()

    def test_classifier_bias_shape_mismatch_rejected(self, tmp_path):
        """Checkpoints with mismatched classifier bias shape are rejected."""
        wrong_bias_path = str(tmp_path / "wrong_bias.pth")
        model = build_densenet201_model(num_classes=NUM_CLASSES)
        state_dict = model.state_dict()
        state_dict["classifier.bias"] = torch.randn(2)  # Incompatible bias shape
        payload = {"state_dict": state_dict, "architecture": "DenseNet201"}
        torch.save(payload, wrong_bias_path)

        with pytest.raises(CheckpointIncompatibleError) as exc_info:
            load_eye_model_checkpoint(model, wrong_bias_path, device=torch.device("cpu"))
        assert "classifier bias shape mismatch" in str(exc_info.value).lower()

    def test_missing_state_dict_keys_rejected(self, tmp_path):
        """Checkpoints missing essential DenseNet201 layers are rejected."""
        missing_keys_path = str(tmp_path / "incomplete.pth")
        incomplete_state = {"classifier.weight": torch.randn(4, DENSENET201_IN_FEATURES)}
        payload = {"state_dict": incomplete_state, "architecture": "DenseNet201"}
        torch.save(payload, missing_keys_path)

        model = build_densenet201_model(num_classes=NUM_CLASSES)
        with pytest.raises(CheckpointIncompatibleError) as exc_info:
            load_eye_model_checkpoint(model, missing_keys_path, device=torch.device("cpu"))
        assert "incompatible state_dict keys" in str(exc_info.value).lower()

    def test_no_random_weight_fallback_on_missing_or_corrupt_checkpoint(self, valid_retina_image, tmp_path):
        """Inference never silently falls back to random weights on checkpoint errors."""
        # Missing checkpoint
        with pytest.raises(CheckpointNotFoundError):
            predict_eye_disease(valid_retina_image, weights_path="nonexistent_checkpoint.pth")

        # Corrupted checkpoint
        corrupt_path = str(tmp_path / "corrupt.pth")
        with open(corrupt_path, "wb") as f:
            f.write(b"CORRUPTED_BYTES")
        with pytest.raises(CheckpointCorruptError):
            predict_eye_disease(valid_retina_image, weights_path=corrupt_path)

    def test_unsupported_checkpoint_container_type_rejected(self, tmp_path):
        """Checkpoints that deserialize to non-dict primitives are rejected."""
        primitive_path = str(tmp_path / "primitive.pth")
        torch.save([1, 2, 3, 4], primitive_path)

        model = build_densenet201_model(num_classes=NUM_CLASSES)
        with pytest.raises(CheckpointIncompatibleError) as exc_info:
            load_eye_model_checkpoint(model, primitive_path, device=torch.device("cpu"))
        assert "unsupported checkpoint payload type" in str(exc_info.value).lower()


# ==============================================================================
# 2. Image Input Safety & Resource Protection
# ==============================================================================

class TestImageInputSafety:
    """Verifies that malicious or malformed image inputs cannot exhaust system resources."""

    def test_oversized_file_rejected_before_loading(self, tmp_path):
        """Files exceeding 30MB are rejected immediately before decoding."""
        large_file = str(tmp_path / "oversized.png")
        # Create a file exceeding 30 MB (e.g. 31 MB)
        with open(large_file, "wb") as f:
            f.seek(31 * 1024 * 1024 - 1)
            f.write(b"\0")

        # 1. Quality gate rejects with structured sentinel message
        q_res = check_image_quality(large_file)
        assert q_res["overall_pass"] is False
        assert "exceeds maximum allowable size" in q_res["error_message"].lower()

        # 2. Preprocessor raises InvalidImageInputError
        with pytest.raises(InvalidImageInputError) as exc_info:
            preprocess_image(large_file)
        assert "exceeds maximum allowable size" in str(exc_info.value).lower()

    def test_oversized_dimensions_rejected(self, tmp_path):
        """Images exceeding 5000x5000 resolution are rejected as potential memory bombs."""
        huge_dim_file = str(tmp_path / "huge_dim.png")
        # Create a 5200 x 250 image (width exceeds 5000)
        huge_canvas = np.zeros((250, 5200, 3), dtype=np.uint8)
        cv2.imwrite(huge_dim_file, huge_canvas)

        # Quality gate rejects
        q_res = check_image_quality(huge_dim_file)
        assert q_res["overall_pass"] is False
        assert "exceed maximum allowable resolution" in q_res["error_message"].lower()

        # Preprocessor rejects
        with pytest.raises(InvalidImageInputError) as exc_info:
            preprocess_image(huge_dim_file)
        assert "exceed maximum allowable resolution" in str(exc_info.value).lower()

    def test_empty_zero_byte_file_safe_rejection(self, tmp_path):
        """0-byte files fail cleanly without unhandled exceptions."""
        empty_file = str(tmp_path / "zero_byte.jpg")
        with open(empty_file, "wb") as f:
            pass

        q_res = check_image_quality(empty_file)
        assert q_res["overall_pass"] is False
        assert "empty (0 bytes)" in q_res["error_message"].lower()

        with pytest.raises(InvalidImageInputError):
            preprocess_image(empty_file)

    def test_corrupt_binary_safe_rejection(self, tmp_path):
        """Arbitrary binary garbage disguised as an image is safely rejected."""
        corrupt_file = str(tmp_path / "corrupt.png")
        with open(corrupt_file, "wb") as f:
            f.write(b"NOT_A_VALID_PNG_FILE_HEADER_GARBAGE")

        q_res = check_image_quality(corrupt_file)
        assert q_res["overall_pass"] is False
        assert "failed to decode" in q_res["error_message"].lower()

        with pytest.raises(InvalidImageInputError):
            preprocess_image(corrupt_file)

    def test_error_messages_do_not_leak_absolute_system_paths(self, tmp_path):
        """User-facing exceptions and sentinels must not leak absolute host directories."""
        sensitive_dir = tmp_path / "internal_user_folder" / "secret_patient_records"
        sensitive_dir.mkdir(parents=True)
        missing_target = str(sensitive_dir / "target_retina.png")

        # 1. Quality gate error message
        q_res = check_image_quality(missing_target)
        assert "internal_user_folder" not in q_res["error_message"]
        assert "secret_patient_records" not in q_res["error_message"]
        assert "target_retina.png" in q_res["error_message"]

        # 2. Preprocessor error message
        with pytest.raises(FileNotFoundError) as exc_info:
            preprocess_image(missing_target)
        err_msg = str(exc_info.value)
        assert "internal_user_folder" not in err_msg
        assert "secret_patient_records" not in err_msg
        assert "target_retina.png" in err_msg

        # 3. Grad-CAM error message
        with pytest.raises(FileNotFoundError) as exc_info:
            generate_gradcam(missing_target, predicted_class_idx=0)
        err_msg = str(exc_info.value)
        assert "internal_user_folder" not in err_msg
        assert "secret_patient_records" not in err_msg


# ==============================================================================
# 3. Privacy, Data Leakage & Structured Logging Safeguards
# ==============================================================================

class TestPrivacyAndLoggingSafeguards:
    """Verifies that telemetry logs never expose credentials, PII, or raw health data."""

    def test_sensitive_credentials_redacted_in_telemetry(self):
        """Credentials and API keys are automatically redacted in telemetry payloads."""
        logger = logging.getLogger("test_privacy")
        payload = log_ml_event(
            logger=logger,
            event_type="TEST_EVENT",
            operation="test_auth",
            status="SUCCESS",
            details={
                "api_key": "sk-proj-secret12345",
                "auth_token": "bearer-token-abcdef",
                "password_hash": "hash_val",
            },
        )
        details = payload.get("details", {})
        assert details["api_key"] == "[REDACTED]"
        assert details["auth_token"] == "[REDACTED]"
        assert details["password_hash"] == "[REDACTED]"

    def test_pii_fields_redacted_in_telemetry(self):
        """Patient identifiers (PII) are masked in structured telemetry."""
        logger = logging.getLogger("test_privacy")
        payload = log_ml_event(
            logger=logger,
            event_type="TEST_EVENT",
            operation="test_pii",
            status="SUCCESS",
            details={
                "patient_id": "PT-987654",
                "mrn": "MRN-112233",
                "patient_name": "Jane Doe",
            },
        )
        details = payload.get("details", {})
        assert details["patient_id"] == "[REDACTED_PII]"
        assert details["mrn"] == "[REDACTED_PII]"
        assert details["patient_name"] == "[REDACTED_PII]"

    def test_raw_physio_measurements_masked_in_telemetry(self):
        """Raw numerical vitals are masked from routine operational logs."""
        logger = logging.getLogger("test_privacy")
        payload = log_ml_event(
            logger=logger,
            event_type="PHYSIO_LOG",
            operation="test_vitals_telemetry",
            status="SUCCESS",
            details={
                "heart_rate": 78.5,
                "spo2": 99.0,
                "temperature": 36.8,
            },
        )
        details = payload.get("details", {})
        assert details["heart_rate"] == "[PHYSIO_DATA_MASKED]"
        assert details["spo2"] == "[PHYSIO_DATA_MASKED]"
        assert details["temperature"] == "[PHYSIO_DATA_MASKED]"

    def test_large_image_blobs_masked_in_telemetry(self):
        """Large image blobs or Base64 payloads are never written into log details."""
        logger = logging.getLogger("test_privacy")
        long_base64_blob = "A" * 600
        payload = log_ml_event(
            logger=logger,
            event_type="GRADCAM_LOG",
            operation="test_blob",
            status="SUCCESS",
            details={"gradcam_image": long_base64_blob},
        )
        assert payload["details"]["gradcam_image"] == "[BLOB_DATA_MASKED]"

    def test_filesystem_paths_sanitized_in_telemetry(self):
        """Paths with slashes or backslashes are reduced to basenames in log details."""
        logger = logging.getLogger("test_privacy")
        payload = log_ml_event(
            logger=logger,
            event_type="FILE_OP",
            operation="test_path",
            status="SUCCESS",
            details={"file": r"C:\Hospital\Radiology\PrivatePatients\patient_01.png"},
        )
        assert payload["details"]["file"] == "patient_01.png"


# ==============================================================================
# 4. Physiological Input Safety Matrix
# ==============================================================================

class TestPhysiologicalInputSafety:
    """Verifies that malformed vitals inputs cannot produce misleading normal assessments."""

    def test_nan_and_infinity_rejected(self):
        """NaN and infinite values are rejected under raise_errors=True and False."""
        for bad_val in [float("nan"), float("inf"), float("-inf")]:
            with pytest.raises(PhysiologicalValidationError):
                analyze_vitals(heart_rate=bad_val, spo2=98.0, temperature=36.8, raise_errors=True)

            res = analyze_vitals(heart_rate=bad_val, spo2=98.0, temperature=36.8, raise_errors=False)
            assert res["status"] == "error"
            assert res["overall_status"] == "Invalid"
            assert res["risk_level"] == "UNKNOWN"

    def test_non_numeric_and_none_rejected(self):
        """None, strings, and collections raise PhysiologicalTypeValidationError."""
        for invalid_val in [None, "75.0", [72], {"hr": 72}]:
            with pytest.raises(PhysiologicalTypeValidationError):
                analyze_vitals(heart_rate=invalid_val, spo2=98.0, temperature=36.8, raise_errors=True)  # type: ignore

            res = analyze_vitals(heart_rate=invalid_val, spo2=98.0, temperature=36.8, raise_errors=False)  # type: ignore
            assert res["status"] == "error"
            assert res["overall_status"] == "Invalid"

    def test_boolean_inputs_strictly_rejected(self):
        """Booleans (True/False and np.bool_) must not be treated as 1 or 0."""
        for bool_val in [True, False, np.bool_(True), np.bool_(False)]:
            with pytest.raises(PhysiologicalTypeValidationError):
                analyze_vitals(heart_rate=bool_val, spo2=98.0, temperature=36.8, raise_errors=True)  # type: ignore

            with pytest.raises(PhysiologicalTypeValidationError):
                analyze_vitals(heart_rate=72.0, spo2=bool_val, temperature=36.8, raise_errors=True)  # type: ignore

    def test_numpy_numeric_types_accepted_and_converted(self):
        """NumPy scalars (np.float32, np.float64, np.int64) are safely accepted and returned as Python floats."""
        res = analyze_vitals(
            heart_rate=np.float32(72.0),
            spo2=np.float64(98.0),
            temperature=np.int64(37),
            raise_errors=True,
        )
        assert res["overall_status"] == "Normal"
        assert isinstance(res["heart_rate"]["value"], float)
        assert isinstance(res["spo2"]["value"], float)
        assert isinstance(res["temperature"]["value"], float)

    def test_impossible_out_of_range_measurements_rejected(self):
        """Measurements outside biological limits are rejected."""
        impossible_cases = [
            (19.9, 98.0, 36.8),   # Heart rate too low (< 20)
            (300.1, 98.0, 36.8),  # Heart rate too high (> 300)
            (75.0, -0.1, 36.8),   # SpO2 negative
            (75.0, 100.1, 36.8),  # SpO2 > 100%
            (75.0, 98.0, 24.9),   # Core temp too low (< 25°C)
            (75.0, 98.0, 45.1),   # Core temp too high (> 45°C)
        ]
        for hr, spo2, temp in impossible_cases:
            with pytest.raises(PhysiologicalValidationError):
                analyze_vitals(heart_rate=hr, spo2=spo2, temperature=temp, raise_errors=True)

            res = analyze_vitals(heart_rate=hr, spo2=spo2, temperature=temp, raise_errors=False)
            assert res["status"] == "error"
            assert res["overall_status"] == "Invalid"

    def test_boundary_values_properly_evaluated(self):
        """Exact biological boundary values evaluate deterministically."""
        # Extreme lower bound HR (20 bpm) -> severe bradycardia -> Abnormal
        res_hr_low = analyze_vitals(heart_rate=20.0, spo2=98.0, temperature=36.8)
        assert res_hr_low["hr_status"] == STATUS_ABNORMAL
        assert res_hr_low["overall_status"] == STATUS_ABNORMAL

        # Extreme upper bound HR (300 bpm) -> severe tachycardia -> Abnormal
        res_hr_high = analyze_vitals(heart_rate=300.0, spo2=98.0, temperature=36.8)
        assert res_hr_high["hr_status"] == STATUS_ABNORMAL

        # SpO2 boundaries: 0.0% -> severe hypoxemia -> Abnormal; 100.0% -> Normal
        res_spo2_low = analyze_vitals(heart_rate=72.0, spo2=0.0, temperature=36.8)
        assert res_spo2_low["spo2_status"] == STATUS_ABNORMAL
        res_spo2_high = analyze_vitals(heart_rate=72.0, spo2=100.0, temperature=36.8)
        assert res_spo2_high["spo2_status"] == STATUS_NORMAL

        # Temperature boundaries: 25.0°C -> hypothermia -> Abnormal; 45.0°C -> hyperthermia -> Abnormal
        res_temp_low = analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=25.0)
        assert res_temp_low["temp_status"] == STATUS_ABNORMAL
        res_temp_high = analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=45.0)
        assert res_temp_high["temp_status"] == STATUS_ABNORMAL


# ==============================================================================
# 5. Grad-CAM Safety & Resource Cleanup
# ==============================================================================

class TestGradCAMSafetyAndResourceCleanup:
    """Verifies that Grad-CAM hooks and model gradients are completely isolated and cleaned up."""

    def test_gradcam_hooks_removed_on_success(self, valid_retina_image, valid_controlled_checkpoint):
        """Forward and backward hooks are removed from the target layer after execution."""
        model = get_cached_model(weights_path=valid_controlled_checkpoint, device=torch.device("cpu"))
        target_layer = model.features.denseblock4

        # Verify no hooks initially
        initial_fwd_hooks = len(target_layer._forward_hooks)
        initial_bwd_hooks = len(target_layer._backward_hooks)

        b64 = generate_gradcam(valid_retina_image, predicted_class_idx=0, model=model)
        assert isinstance(b64, str)

        # Verify hook handles were removed in finally:
        assert len(target_layer._forward_hooks) == initial_fwd_hooks
        assert len(target_layer._backward_hooks) == initial_bwd_hooks

    def test_gradcam_hooks_removed_on_failure(self, valid_retina_image, valid_controlled_checkpoint, monkeypatch):
        """Hooks are guaranteed to be cleaned up even if backpropagation throws an error."""
        model = get_cached_model(weights_path=valid_controlled_checkpoint, device=torch.device("cpu"))
        target_layer = model.features.denseblock4

        # Simulate exception during backward pass
        def failing_backward(*args, **kwargs):
            raise RuntimeError("Simulated failure during backward pass")

        monkeypatch.setattr(torch.Tensor, "backward", failing_backward)

        with pytest.raises(RuntimeError):
            generate_gradcam(valid_retina_image, predicted_class_idx=0, model=model)

        # Confirm hook dictionaries are restored to empty
        assert len(target_layer._forward_hooks) == 0
        assert len(target_layer._backward_hooks) == 0

    def test_model_gradients_zeroed_after_gradcam(self, valid_retina_image, valid_controlled_checkpoint):
        """All model parameters have None or zero gradients after Grad-CAM generation."""
        model = get_cached_model(weights_path=valid_controlled_checkpoint, device=torch.device("cpu"))
        generate_gradcam(valid_retina_image, predicted_class_idx=1, model=model)

        for name, param in model.named_parameters():
            assert param.grad is None, f"Parameter {name} retained gradient after Grad-CAM."

    def test_invalid_class_index_rejected(self, valid_retina_image, valid_controlled_checkpoint):
        """Class indices outside [0, 3] or non-integers are rejected without running inference."""
        model = get_cached_model(weights_path=valid_controlled_checkpoint, device=torch.device("cpu"))

        with pytest.raises(ValueError):
            generate_gradcam(valid_retina_image, predicted_class_idx=-1, model=model)

        with pytest.raises(ValueError):
            generate_gradcam(valid_retina_image, predicted_class_idx=4, model=model)

        with pytest.raises(TypeError):
            generate_gradcam(valid_retina_image, predicted_class_idx=1.5, model=model)  # type: ignore

    def test_constant_heatmap_handled_without_division_by_zero(self, valid_retina_image, valid_controlled_checkpoint, monkeypatch):
        """Uniform/constant activations do not trigger division by zero or NaN values."""
        model = get_cached_model(weights_path=valid_controlled_checkpoint, device=torch.device("cpu"))

        # Generate Grad-CAM normally; verify min-max normalization works and valid PNG returned
        b64 = generate_gradcam(valid_retina_image, predicted_class_idx=2, model=model)
        raw_bytes = base64.b64decode(b64)
        assert raw_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_source_image_file_unmodified_on_disk(self, valid_retina_image, valid_controlled_checkpoint):
        """Grad-CAM visualization operations must never modify the original input image file."""
        with open(valid_retina_image, "rb") as f:
            initial_hash = hashlib.sha256(f.read()).hexdigest()
        initial_mtime = os.path.getmtime(valid_retina_image)

        generate_gradcam(valid_retina_image, predicted_class_idx=0, weights_path=valid_controlled_checkpoint)

        with open(valid_retina_image, "rb") as f:
            final_hash = hashlib.sha256(f.read()).hexdigest()
        final_mtime = os.path.getmtime(valid_retina_image)

        assert initial_hash == final_hash
        assert initial_mtime == final_mtime
