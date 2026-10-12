"""
Unit and Integration Tests for Explainable AI (XAI) & Confidence Validation.
Tests Grad-CAM visual explanations, confidence invariants, temperature scaling,
and safety failure handling.
"""

import base64
import os
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES
from ml.exceptions import (
    GradCAMGenerationError,
    InvalidImageInputError,
    ModelInferenceError,
)
from ml.eye_model import (
    build_densenet201_model,
    predict_eye_disease,
)
from ml.gradcam import generate_gradcam, get_target_layer
from ml.xai_validation import (
    compute_ece_and_mce,
    compute_nll_and_brier,
    fit_temperature_scaling,
)


@pytest.fixture
def test_retina_image(tmp_path) -> str:
    img_path = str(tmp_path / "xai_test_image.png")
    arr = np.full((320, 320, 3), 100, dtype=np.uint8)
    cv2.circle(arr, (160, 160), 100, (180, 160, 140), -1)
    cv2.imwrite(img_path, arr)
    return img_path


class TestConfidenceSafetyAndNonFiniteHandling:
    """Verifies that non-finite model outputs are safely caught without silent failures."""

    def test_nan_logits_in_predict_eye_disease_raises_model_inference_error(
        self, test_retina_image
    ):
        class NaNModel(nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0], 4), float("nan"))

        with pytest.raises(ModelInferenceError) as exc:
            predict_eye_disease(test_retina_image, model=NaNModel().eval())
        assert "non-finite" in str(exc.value).lower()

    def test_inf_logits_in_predict_eye_disease_raises_model_inference_error(
        self, test_retina_image
    ):
        class InfModel(nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0], 4), float("inf"))

        with pytest.raises(ModelInferenceError) as exc:
            predict_eye_disease(test_retina_image, model=InfModel().eval())
        assert "non-finite" in str(exc.value).lower()

    def test_nan_logits_in_gradcam_raises_gradcam_error(
        self, test_retina_image
    ):
        class NaNGradModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential()
                self.features.denseblock4 = nn.Conv2d(3, 4, 3, padding=1)
                self.classifier = nn.Linear(4, 4)

            def forward(self, x):
                _ = self.features.denseblock4(x)
                return torch.full((x.shape[0], 4), float("nan"), requires_grad=True)

        with pytest.raises(GradCAMGenerationError) as exc:
            generate_gradcam(test_retina_image, predicted_class_idx=0, model=NaNGradModel().eval())
        assert "non-finite" in str(exc.value).lower()


class TestCalibrationCalculations:
    """Verifies ECE, MCE, NLL, and temperature scaling mathematics."""

    def test_perfect_calibration_ece_is_zero(self):
        # 100% confidence with 100% accuracy
        confs = np.ones(50, dtype=np.float32)
        accs = np.ones(50, dtype=np.int64)
        ece, mce, _ = compute_ece_and_mce(confs, accs, num_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-5)
        assert mce == pytest.approx(0.0, abs=1e-5)

    def test_completely_uncalibrated_ece(self):
        # 100% confidence with 0% accuracy
        confs = np.ones(50, dtype=np.float32)
        accs = np.zeros(50, dtype=np.int64)
        ece, mce, _ = compute_ece_and_mce(confs, accs, num_bins=10)
        assert ece == pytest.approx(1.0, abs=1e-3)
        assert mce == pytest.approx(1.0, abs=1e-3)

    def test_temperature_scaling_optimization(self):
        torch.manual_seed(42)
        # Overconfident logits
        logits = torch.randn(100, 4) * 5.0
        labels = torch.randint(0, 4, (100,))
        t_opt = fit_temperature_scaling(logits, labels)
        assert 0.05 < t_opt < 10.0


class TestGradCamCounterfactualDiscrimination:
    """Verifies that changing target class changes the gradient computation."""

    def test_class_specific_heatmaps_are_distinct(self, test_retina_image):
        class DistinctFeatureCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential()
                self.features.denseblock4 = nn.Conv2d(3, 8, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(8, 4)
                # Set weights distinct per class
                with torch.no_grad():
                    self.classifier.weight.copy_(torch.randn(4, 8) * 2.0)

            def forward(self, x):
                feats = self.features.denseblock4(x)
                pooled = self.pool(feats)
                return self.classifier(torch.flatten(pooled, 1))

        model = DistinctFeatureCNN().eval()
        b64_c0 = generate_gradcam(test_retina_image, 0, model=model)
        b64_c1 = generate_gradcam(test_retina_image, 1, model=model)

        img0 = cv2.imdecode(np.frombuffer(base64.b64decode(b64_c0), np.uint8), cv2.IMREAD_COLOR)
        img1 = cv2.imdecode(np.frombuffer(base64.b64decode(b64_c1), np.uint8), cv2.IMREAD_COLOR)

        mean_diff = float(np.mean(np.abs(img0.astype(float) - img1.astype(float))))
        # Heatmaps for distinct target classes should have different visual activations
        assert mean_diff > 0.1
