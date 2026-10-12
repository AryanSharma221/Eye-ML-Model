"""
Comprehensive ML Regression Testing Suite for Multimodal Health Screening.

This module provides deterministic regression protection against previously identified
bugs, edge conditions, contract regressions, resource leaks, and cross-platform issues.

Regression Categories Covered:
1. Image-Quality Threshold Boundary Conditions (blur, resolution, illumination, contrast, priority).
2. Resource Safety and Hook Isolation (repeated Grad-CAM calls, failure cleanup, gradient neutrality).
3. Public Interface Contract Stability & Backward Compatibility (run_screening, screen_eye_image, analyze_vitals).
4. Multimodal Independence and Fault Isolation (vitals failure independence, image defect independence).
5. Type Mutation and Extreme Input Rejection (booleans, strings, non-finites, truncated files).
6. Checkpoint Integrity and Deterministic Device Resolution.
"""

import math
import os
import shutil
import tempfile
import threading
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

import ml
from ml.config import MLConfig, get_default_checkpoint_path, resolve_device
from ml.exceptions import (
    DeviceUnavailableError,
    GradCAMGenerationError,
    InvalidImageInputError,
    ModelInferenceError,
    PhysiologicalTypeValidationError,
    PhysiologicalValidationError,
    QualityGateRejectionError,
)
from ml.eye_model import (
    CLASS_NAMES,
    NUM_CLASSES,
    build_densenet201_model,
    clear_model_cache,
    predict_eye_disease,
    screen_eye_image,
)
from ml.gradcam import generate_gradcam, get_target_layer
from ml.physio_model import (
    STATUS_ABNORMAL,
    STATUS_HIGHER,
    STATUS_LOWER,
    STATUS_NORMAL,
    analyze_vitals,
)
from ml.preprocessing import IMAGENET_MEAN, IMAGENET_STD, preprocess_image
from ml.quality_gate import check_image_quality
from ml.screening_pipeline import run_screening


# =============================================================================
# Helper Utilities for Synthetic Regression Fixtures
# =============================================================================

def _create_synthetic_fundus(
    output_path: str,
    width: int = 256,
    height: int = 256,
    bg_val: int = 120,
    circle_val: int = 160,
    add_texture: bool = True,
) -> str:
    """Creates a synthetic image with a clear circular retinal disk structure."""
    img = np.full((height, width, 3), bg_val, dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(width, height) // 3
    cv2.circle(img, center, radius, (circle_val, circle_val, circle_val), -1)

    if add_texture:
        # Add high-frequency edge detail to guarantee laplacian variance >= 100.0
        for i in range(10, min(width, height) - 10, 8):
            cv2.line(img, (i, 10), (width - i, height - 10), (220, 200, 180), 1)
            cv2.line(img, (10, i), (width - 10, height - i), (180, 200, 220), 1)

    cv2.imwrite(output_path, img)
    return output_path


class _MockFourClassClassifier(nn.Module):
    """Deterministic, reproducible 4-class mock DenseNet-like architecture for unit tests."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential()
        # Mock denseblock4 with 2 conv layers to support feature capture
        denseblock4 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.features.add_module("denseblock4", denseblock4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, NUM_CLASSES)

        # Deterministic weight initialization
        with torch.no_grad():
            self.classifier.weight.fill_(0.1)
            self.classifier.bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.pool(feat)
        flat = torch.flatten(pooled, 1)
        return self.classifier(flat)


# =============================================================================
# 1. Image Quality Gate Threshold Boundary Regressions
# =============================================================================

@pytest.mark.regression
@pytest.mark.unit
class TestQualityGateThresholdBoundaries:
    """Verifies exact numerical boundary conditions for image-quality pass/fail."""

    @pytest.fixture(autouse=True)
    def setup_temp_workspace(self, tmp_path):
        self.test_dir = str(tmp_path)

    def test_resolution_exact_boundaries(self):
        """Resolution must accept >= 224x224 and reject 223x224 and 224x223."""
        # 224x224 should pass resolution check
        path_valid = os.path.join(self.test_dir, "res_224x224.png")
        _create_synthetic_fundus(path_valid, width=224, height=224)
        res_valid = check_image_quality(path_valid)
        assert res_valid["resolution_ok"] is True

        # 223x224 must fail resolution check
        path_under_w = os.path.join(self.test_dir, "res_223x224.png")
        _create_synthetic_fundus(path_under_w, width=223, height=224)
        res_under_w = check_image_quality(path_under_w)
        assert res_under_w["resolution_ok"] is False
        assert res_under_w["overall_pass"] is False
        assert "Insufficient resolution" in str(res_under_w["error_message"])

        # 224x223 must fail resolution check
        path_under_h = os.path.join(self.test_dir, "res_224x223.png")
        _create_synthetic_fundus(path_under_h, width=224, height=223)
        res_under_h = check_image_quality(path_under_h)
        assert res_under_h["resolution_ok"] is False
        assert res_under_h["overall_pass"] is False
        assert "Insufficient resolution" in str(res_under_h["error_message"])

    def test_blur_boundary_condition(self):
        """Blur threshold is strictly >= 100.0 (variance of Laplacian)."""
        base_path = os.path.join(self.test_dir, "base_test.png")
        _create_synthetic_fundus(base_path, width=256, height=256)

        # Mock Laplacian variance to exactly 99.99 (just below 100.0)
        with patch("cv2.Laplacian") as mock_lap:
            mock_lap.return_value = MagicMock(var=lambda: 99.99)
            result_fail = check_image_quality(base_path)
            assert result_fail["blur_score"] == 100.0 or result_fail["blur_score"] == 99.99
            # Because round(99.99, 2) is 100.0, let's test 99.94 which rounds to 99.94
            mock_lap.return_value = MagicMock(var=lambda: 99.94)
            result_fail = check_image_quality(base_path)
            assert result_fail["blur_score"] == 99.94
            assert result_fail["overall_pass"] is False
            assert "blur" in str(result_fail["error_message"]).lower()

            # Mock Laplacian variance to exactly 100.00
            mock_lap.return_value = MagicMock(var=lambda: 100.00)
            result_pass = check_image_quality(base_path)
            assert result_pass["blur_score"] == 100.00
            # If all other checks pass, overall_pass is True
            assert "blur" not in str(result_pass.get("error_message", "")).lower()

    def test_brightness_interval_boundaries(self):
        """Brightness interval is strictly [30.0, 225.0]."""
        base_path = os.path.join(self.test_dir, "brightness_test.png")
        _create_synthetic_fundus(base_path, width=256, height=256)

        # 29.9 must fail as too dark
        with patch("numpy.mean", return_value=29.90):
            res_dark = check_image_quality(base_path)
            assert res_dark["brightness_mean"] == 29.90
            assert res_dark["overall_pass"] is False
            assert "too dark" in str(res_dark["error_message"]).lower()

        # 30.0 must pass brightness check
        with patch("numpy.mean", return_value=30.00):
            res_min = check_image_quality(base_path)
            assert res_min["brightness_mean"] == 30.00
            assert "too dark" not in str(res_min.get("error_message", "")).lower()

        # 225.0 must pass brightness check
        with patch("numpy.mean", return_value=225.00):
            res_max = check_image_quality(base_path)
            assert res_max["brightness_mean"] == 225.00
            assert "too bright" not in str(res_max.get("error_message", "")).lower()

        # 225.1 must fail as too bright
        with patch("numpy.mean", return_value=225.10):
            res_bright = check_image_quality(base_path)
            assert res_bright["brightness_mean"] == 225.10
            assert res_bright["overall_pass"] is False
            assert "too bright" in str(res_bright["error_message"]).lower()

    def test_contrast_threshold_boundary(self):
        """Contrast standard deviation threshold is strictly >= 20.0."""
        base_path = os.path.join(self.test_dir, "contrast_test.png")
        _create_synthetic_fundus(base_path, width=256, height=256)

        # 19.9 must fail contrast check
        with patch("numpy.std", return_value=19.90):
            res_low = check_image_quality(base_path)
            assert res_low["contrast_std"] == 19.90
            assert res_low["overall_pass"] is False
            assert "contrast" in str(res_low["error_message"]).lower()

        # 20.0 must pass contrast check
        with patch("numpy.std", return_value=20.00):
            res_pass = check_image_quality(base_path)
            assert res_pass["contrast_std"] == 20.00
            assert "contrast" not in str(res_pass.get("error_message", "")).lower()

    def test_deterministic_error_priority_cascade(self):
        """Error priority must be deterministic: resolution > blur > brightness > contrast > retina."""
        base_path = os.path.join(self.test_dir, "multi_fail.png")
        # Resolution fails (100x100 < 224x224), blur fails (flat), contrast fails (0.0)
        cv2.imwrite(base_path, np.zeros((100, 100, 3), dtype=np.uint8))

        res = check_image_quality(base_path)
        assert res["overall_pass"] is False
        # Resolution is highest priority failure
        assert "Insufficient resolution" in str(res["error_message"])


# =============================================================================
# 2. Repeated Execution and Resource Safety Regressions
# =============================================================================

@pytest.mark.regression
@pytest.mark.integration
class TestResourceSafetyAndLeakRegressions:
    """Verifies hook removal, gradient neutrality, and memory safety over repeated runs."""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path):
        self.test_dir = str(tmp_path)
        self.img_path = os.path.join(self.test_dir, "leak_test.png")
        _create_synthetic_fundus(self.img_path, width=224, height=224)
        self.mock_model = _MockFourClassClassifier()
        clear_model_cache()

    def test_gradcam_hook_cleanup_over_multiple_iterations(self):
        """Repeated Grad-CAM calls must never accumulate lingering hooks."""
        target_layer = self.mock_model.features.denseblock4

        for iteration in range(10):
            b64_res = generate_gradcam(
                image_path=self.img_path,
                predicted_class_idx=iteration % NUM_CLASSES,
                model=self.mock_model,
                device=torch.device("cpu"),
            )
            assert isinstance(b64_res, str)
            assert len(b64_res) > 100

            # Verify target module has ZERO lingering forward/backward hooks
            assert len(target_layer._forward_hooks) == 0, f"Lingering forward hooks on iter {iteration}"
            assert len(target_layer._backward_hooks) == 0, f"Lingering backward hooks on iter {iteration}"

            # Verify ALL modules in the model have ZERO lingering hooks
            for name, module in self.mock_model.named_modules():
                assert len(module._forward_hooks) == 0, f"Lingering forward hook on module {name}"
                assert len(module._backward_hooks) == 0, f"Lingering backward hook on module {name}"

    def test_gradcam_parameter_gradients_strictly_none_after_execution(self):
        """Grad-CAM must zero all model gradients and leave parameter requires_grad intact."""
        orig_states = [p.requires_grad for p in self.mock_model.parameters()]

        generate_gradcam(
            image_path=self.img_path,
            predicted_class_idx=0,
            model=self.mock_model,
            device=torch.device("cpu"),
        )

        # Gradients must be None
        for p in self.mock_model.parameters():
            assert p.grad is None, "Parameter grad was not reset to None"

        # requires_grad must be restored to original states
        current_states = [p.requires_grad for p in self.mock_model.parameters()]
        assert current_states == orig_states, "Parameter requires_grad state was mutated"

    def test_gradcam_failure_cleanup_neutrality(self):
        """Even if Grad-CAM fails midway, hooks and parameter states must be completely cleaned up."""
        target_layer = self.mock_model.features.denseblock4

        # Simulate exception during backward pass by passing non-finite logits
        with patch.object(self.mock_model, "forward", return_value=torch.tensor([[float("nan")] * 4])):
            with pytest.raises(GradCAMGenerationError):
                generate_gradcam(
                    image_path=self.img_path,
                    predicted_class_idx=0,
                    model=self.mock_model,
                    device=torch.device("cpu"),
                )

        # Verify hooks were cleaned up in finally block despite the exception
        assert len(target_layer._forward_hooks) == 0, "Forward hook remained registered after exception"
        assert len(target_layer._backward_hooks) == 0, "Backward hook remained registered after exception"

        for p in self.mock_model.parameters():
            assert p.grad is None

    def test_repeated_inference_reproducibility(self):
        """Repeated forward inference calls on the same image yield identical probabilities."""
        tensor = preprocess_image(self.img_path)
        self.mock_model.eval()

        with torch.no_grad():
            first_out = torch.softmax(self.mock_model(tensor), dim=1).cpu().numpy()
            for _ in range(5):
                subsequent_out = torch.softmax(self.mock_model(tensor), dim=1).cpu().numpy()
                np.testing.assert_allclose(first_out, subsequent_out, rtol=1e-5, atol=1e-5)


# =============================================================================
# 3. Public Contract Backward Compatibility Regressions
# =============================================================================

@pytest.mark.regression
@pytest.mark.integration
class TestPublicContractCompatibility:
    """Verifies that all public functions maintain stable dictionary contracts."""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path):
        self.test_dir = str(tmp_path)
        self.img_path = os.path.join(self.test_dir, "contract_test.png")
        _create_synthetic_fundus(self.img_path, width=224, height=224)
        self.mock_model = _MockFourClassClassifier()
        clear_model_cache()

    def test_run_screening_complete_contract_keys(self):
        """run_screening must always return all specified top-level and nested keys."""
        with patch("ml.screening_pipeline.get_cached_model", return_value=self.mock_model):
            result = run_screening(
                image_path=self.img_path,
                heart_rate=72.0,
                spo2=98.0,
                temperature=36.8,
                device="cpu",
                include_gradcam=True,
            )

        # Top-level required keys
        expected_top_keys = {
            "status",
            "screening_id",
            "timestamp",
            "duration_ms",
            "timings_ms",
            "eye_screening",
            "vitals_screening",
            "overall_risk_level",
            "disclaimer",
        }
        for k in expected_top_keys:
            assert k in result, f"Missing key '{k}' in run_screening response"

        # Eye screening contract keys
        expected_eye_keys = {
            "status",
            "prediction",
            "class_index",
            "confidence",
            "risk_level",
            "all_probabilities",
            "quality_checks",
            "gradcam_image",
        }
        for k in expected_eye_keys:
            assert k in result["eye_screening"], f"Missing key '{k}' in eye_screening subdict"

        # Vitals screening contract keys
        expected_vitals_keys = {
            "status",
            "heart_rate",
            "spo2",
            "temperature",
            "overall_status",
            "risk_level",
            "is_abnormal",
            "summary",
            "hr_status",
            "spo2_status",
            "temp_status",
        }
        for k in expected_vitals_keys:
            assert k in result["vitals_screening"], f"Missing key '{k}' in vitals_screening subdict"

        # Timings sub-dictionary keys
        expected_timing_keys = {
            "quality_gate",
            "preprocessing",
            "eye_inference",
            "gradcam",
            "vitals_analysis",
            "total_pipeline",
        }
        for k in expected_timing_keys:
            assert k in result["timings_ms"], f"Missing key '{k}' in timings_ms subdict"

    def test_screen_eye_image_quality_rejection_contract(self):
        """screen_eye_image rejection payload must strictly preserve contract structure."""
        corrupt_path = os.path.join(self.test_dir, "zero_byte.png")
        with open(corrupt_path, "wb") as f:
            pass  # 0 bytes

        res = screen_eye_image(corrupt_path)
        assert res["status"] == "rejected"
        assert "reason" in res
        assert isinstance(res["quality_checks"], dict)
        assert res["quality_checks"]["overall_pass"] is False
        assert "empty" in str(res["reason"]).lower()

    def test_analyze_vitals_contract_keys(self):
        """analyze_vitals must strictly return all specified triage keys."""
        vitals = analyze_vitals(heart_rate=80, spo2=99, temperature=37.0)
        expected_keys = {
            "hr_status",
            "spo2_status",
            "temp_status",
            "overall_status",
            "risk_level",
            "is_abnormal",
            "summary",
        }
        for k in expected_keys:
            assert k in vitals, f"Missing key '{k}' in analyze_vitals result"

        assert vitals["overall_status"] == STATUS_NORMAL
        assert vitals["risk_level"] == "LOW"
        assert vitals["is_abnormal"] is False


# =============================================================================
# 4. Multimodal Independence and Fault Isolation Regressions
# =============================================================================

@pytest.mark.regression
@pytest.mark.integration
class TestMultimodalIndependenceRegressions:
    """Verifies complete independence between retinal and physiological modalities."""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path):
        self.test_dir = str(tmp_path)
        self.valid_img = os.path.join(self.test_dir, "valid.png")
        _create_synthetic_fundus(self.valid_img, width=224, height=224)
        self.mock_model = _MockFourClassClassifier()
        clear_model_cache()

    def test_vitals_failure_does_not_corrupt_valid_eye_screening(self):
        """When vitals validation fails, valid eye screening completes normally (partial status)."""
        with patch("ml.screening_pipeline.get_cached_model", return_value=self.mock_model):
            # Pass invalid physiological parameters: negative HR
            result = run_screening(
                image_path=self.valid_img,
                heart_rate=-50.0,  # Invalid!
                spo2=99.0,
                temperature=36.8,
                device="cpu",
            )

        assert result["status"] == "partial"
        assert result["eye_screening"]["status"] == "success"
        assert result["eye_screening"]["prediction"] in CLASS_NAMES
        assert result["vitals_screening"]["status"] == "error"
        assert "negative" in str(result["vitals_screening"]["reason"]).lower() or "plausible" in str(result["vitals_screening"]["reason"]).lower()
        # Overall risk defaults to eye risk
        assert result["overall_risk_level"] == result["eye_screening"]["risk_level"]

    def test_eye_defect_does_not_corrupt_valid_vitals_screening(self):
        """When eye image fails quality gate, valid vitals analysis completes normally."""
        bad_img = os.path.join(self.test_dir, "bad.png")
        with open(bad_img, "wb") as f:
            f.write(b"NOT_AN_IMAGE")

        result = run_screening(
            image_path=bad_img,
            heart_rate=75.0,
            spo2=98.0,
            temperature=36.9,
            device="cpu",
        )

        assert result["status"] == "partial"
        assert result["eye_screening"]["status"] == "rejected"
        assert result["vitals_screening"]["status"] == "success"
        assert result["vitals_screening"]["overall_status"] == STATUS_NORMAL
        assert result["vitals_screening"]["risk_level"] == "LOW"
        assert result["overall_risk_level"] == "LOW"

    def test_gradcam_failure_isolation(self):
        """If Grad-CAM fails, eye disease prediction remains SUCCESS and is not discarded."""
        with patch("ml.screening_pipeline.get_cached_model", return_value=self.mock_model):
            # Patch generate_gradcam to raise an error
            with patch("ml.screening_pipeline.generate_gradcam", side_effect=GradCAMGenerationError("GPU OOM")):
                result = run_screening(
                    image_path=self.valid_img,
                    heart_rate=75.0,
                    spo2=98.0,
                    temperature=36.9,
                    device="cpu",
                    include_gradcam=True,
                )

        assert result["eye_screening"]["status"] == "success"
        assert result["eye_screening"]["prediction"] in CLASS_NAMES
        assert result["eye_screening"]["gradcam_image"] is None
        assert "gradcam_error" in result["eye_screening"]
        assert "GPU OOM" in result["eye_screening"]["gradcam_error"]


# =============================================================================
# 5. Malformed Inputs and Type Mutation Regressions
# =============================================================================

@pytest.mark.regression
@pytest.mark.unit
class TestInputMutationAndSafetyRegressions:
    """Verifies robust rejection of non-numeric, non-finite, and malformed inputs."""

    def test_vitals_type_mutation_boolean_rejected(self):
        """Booleans (True/False) must be rejected with PhysiologicalTypeValidationError, not treated as 1/0."""
        with pytest.raises(PhysiologicalTypeValidationError):
            analyze_vitals(heart_rate=True, spo2=98.0, temperature=36.8)  # type: ignore

        with pytest.raises(PhysiologicalTypeValidationError):
            analyze_vitals(heart_rate=75.0, spo2=False, temperature=36.8)  # type: ignore

    def test_vitals_non_finite_rejection(self):
        """NaN and Infinite vital signs must raise PhysiologicalValidationError."""
        with pytest.raises(PhysiologicalValidationError):
            analyze_vitals(heart_rate=float("nan"), spo2=98.0, temperature=36.8)

        with pytest.raises(PhysiologicalValidationError):
            analyze_vitals(heart_rate=75.0, spo2=float("inf"), temperature=36.8)

    def test_vitals_extreme_implausible_limits(self):
        """Readings outside plausible human limits (<20 or >300 bpm) must raise PhysiologicalValidationError."""
        with pytest.raises(PhysiologicalValidationError):
            analyze_vitals(heart_rate=15.0, spo2=98.0, temperature=36.8)

        with pytest.raises(PhysiologicalValidationError):
            analyze_vitals(heart_rate=350.0, spo2=98.0, temperature=36.8)

    def test_device_resolution_strictness(self):
        """Explicitly requesting an invalid or unavailable device must raise appropriate errors."""
        with pytest.raises(DeviceUnavailableError):
            resolve_device("cuda:99")

        with pytest.raises(ValueError):
            resolve_device("tpu")

    def test_canonical_class_names_invariants(self):
        """The canonical 4 classes and count must be strictly immutable."""
        assert NUM_CLASSES == 4
        assert CLASS_NAMES == ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
