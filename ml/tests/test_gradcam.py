"""
Unit and Integration Tests for Grad-CAM Explainability Module (ml/gradcam.py).

Verifies:
1. Base64 PNG output format, valid PNG binary signature, and decoded image properties.
2. Exact matching of native original image dimensions and aspect ratios.
3. Correct handling for all 4 valid class indices (0 to 3) and rejection of invalid indices.
4. Input validation (missing images, empty paths, directories, empty files, corrupt binaries).
5. Controlled local models (no pretrained weight downloads, CPU compatibility).
6. Missing checkpoint behavior when model is not supplied.
7. Proper handling of constant/zero activation maps without division-by-zero errors.
8. Reliable hook removal on both success and failure.
9. Parameter gradient cleanup (no unintended gradient accumulation).
10. Preservation of caller-supplied model training/evaluation mode.
"""

import base64
import os
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.eye_model import (
    CLASS_NAMES,
    NUM_CLASSES,
    build_densenet201_model,
)
from ml.gradcam import generate_gradcam, get_target_layer


@pytest.fixture
def sample_retinal_image(tmp_path) -> str:
    """Creates a synthetic retinal image with realistic dimensions (300x400)."""
    img_path = str(tmp_path / "test_retina.png")
    canvas = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(canvas, (200, 150), 120, (130, 120, 110), -1)
    cv2.circle(canvas, (240, 150), 25, (220, 210, 190), -1)
    cv2.line(canvas, (240, 150), (120, 90), (40, 30, 160), 2)
    cv2.imwrite(img_path, canvas)
    return img_path


@pytest.fixture
def controlled_densenet_model() -> nn.Module:
    """Returns an uninitialized DenseNet201 on CPU (zero weight downloads)."""
    torch.manual_seed(42)
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    model.eval()
    return model


class TestGradCamInputValidation:
    """Verifies strict input parameter validation."""

    def test_invalid_class_indices_raise_value_error(
        self, sample_retinal_image, controlled_densenet_model
    ):
        for invalid_idx in [-1, -5, 4, 10]:
            with pytest.raises(ValueError) as exc:
                generate_gradcam(
                    image_path=sample_retinal_image,
                    predicted_class_idx=invalid_idx,
                    model=controlled_densenet_model,
                )
            assert "predicted_class_idx" in str(exc.value)

    def test_non_integer_class_indices_raise_type_error(
        self, sample_retinal_image, controlled_densenet_model
    ):
        for invalid_idx in [None, "0", 1.5, True, False, [0]]:
            with pytest.raises(TypeError) as exc:
                generate_gradcam(
                    image_path=sample_retinal_image,
                    predicted_class_idx=invalid_idx,
                    model=controlled_densenet_model,
                )
            assert "integer" in str(exc.value).lower()

    def test_nonexistent_image_path_raises_filenotfound(
        self, tmp_path, controlled_densenet_model
    ):
        missing = str(tmp_path / "nonexistent.png")
        with pytest.raises(FileNotFoundError):
            generate_gradcam(
                image_path=missing,
                predicted_class_idx=0,
                model=controlled_densenet_model,
            )

    def test_empty_or_whitespace_path_raises_value_or_type_error(
        self, controlled_densenet_model
    ):
        with pytest.raises(ValueError):
            generate_gradcam(
                image_path="",
                predicted_class_idx=0,
                model=controlled_densenet_model,
            )
        with pytest.raises(ValueError):
            generate_gradcam(
                image_path="   ",
                predicted_class_idx=0,
                model=controlled_densenet_model,
            )
        with pytest.raises(TypeError):
            generate_gradcam(
                image_path=None,
                predicted_class_idx=0,
                model=controlled_densenet_model,
            )

    def test_directory_path_raises_isadirectoryerror(
        self, tmp_path, controlled_densenet_model
    ):
        with pytest.raises(IsADirectoryError):
            generate_gradcam(
                image_path=str(tmp_path),
                predicted_class_idx=0,
                model=controlled_densenet_model,
            )

    def test_empty_0_byte_file_raises_ioerror(
        self, tmp_path, controlled_densenet_model
    ):
        empty_file = str(tmp_path / "empty.png")
        with open(empty_file, "wb"):
            pass
        with pytest.raises(IOError):
            generate_gradcam(
                image_path=empty_file,
                predicted_class_idx=0,
                model=controlled_densenet_model,
            )

    def test_corrupted_image_file_raises_ioerror(
        self, tmp_path, controlled_densenet_model
    ):
        corrupt_file = str(tmp_path / "corrupt.png")
        with open(corrupt_file, "wb") as f:
            f.write(b"not an image file")
        with pytest.raises(IOError):
            generate_gradcam(
                image_path=corrupt_file,
                predicted_class_idx=0,
                model=controlled_densenet_model,
            )


class TestGradCamExecutionAndOutputFormat:
    """Verifies output format, PNG decoding, dimensions, and class index handling."""

    def test_successful_base64_png_generation(
        self, sample_retinal_image, controlled_densenet_model
    ):
        b64_result = generate_gradcam(
            image_path=sample_retinal_image,
            predicted_class_idx=0,
            model=controlled_densenet_model,
        )

        assert isinstance(b64_result, str)
        assert len(b64_result) > 100
        # Must be raw Base64 without data URL scheme
        assert not b64_result.startswith("data:")
        # Base64 string decode test
        decoded_bytes = base64.b64decode(b64_result)
        # Verify valid PNG binary magic header: 89 50 4E 47 0D 0A 1A 0A
        assert decoded_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    def test_decoding_output_produces_valid_image(
        self, sample_retinal_image, controlled_densenet_model
    ):
        b64_result = generate_gradcam(
            image_path=sample_retinal_image,
            predicted_class_idx=1,
            model=controlled_densenet_model,
        )

        raw_bytes = base64.b64decode(b64_result)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        decoded_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        assert decoded_img is not None
        assert decoded_img.ndim == 3
        assert decoded_img.shape[2] == 3

    def test_output_dimensions_preserve_native_original_image(
        self, tmp_path, controlled_densenet_model
    ):
        # Create an image with distinct dimensions: height 360, width 520
        orig_h, orig_w = 360, 520
        custom_img = str(tmp_path / "native_dims.png")
        arr = np.full((orig_h, orig_w, 3), 120, dtype=np.uint8)
        cv2.circle(arr, (260, 180), 100, (200, 180, 160), -1)
        cv2.imwrite(custom_img, arr)

        b64_result = generate_gradcam(
            image_path=custom_img,
            predicted_class_idx=2,
            model=controlled_densenet_model,
        )

        raw_bytes = base64.b64decode(b64_result)
        decoded_img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)

        assert decoded_img.shape == (orig_h, orig_w, 3)

    def test_all_four_valid_class_indices(
        self, sample_retinal_image, controlled_densenet_model
    ):
        for idx in range(NUM_CLASSES):
            b64_result = generate_gradcam(
                image_path=sample_retinal_image,
                predicted_class_idx=idx,
                model=controlled_densenet_model,
            )
            assert isinstance(b64_result, str)
            raw = base64.b64decode(b64_result)
            assert raw.startswith(b"\x89PNG\r\n\x1a\n")


class TestGradCamModelAndLayerHandling:
    """Verifies target layer resolution, checkpoint loading, and model compatibility."""

    def test_incompatible_model_raises_meaningful_error(
        self, sample_retinal_image
    ):
        # Model without features.denseblock4
        incompatible_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.Flatten(),
            nn.Linear(16 * 222 * 222, 4),
        )

        with pytest.raises(AttributeError) as exc:
            generate_gradcam(
                image_path=sample_retinal_image,
                predicted_class_idx=0,
                model=incompatible_model,
            )
        assert "features.denseblock4" in str(exc.value)

    def test_custom_target_layer_specification(
        self, sample_retinal_image
    ):
        # Mini model with explicitly targeted layer
        class MiniCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.custom_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(8, 4)

            def forward(self, x):
                x = self.custom_conv(x)
                x = self.pool(x)
                x = torch.flatten(x, 1)
                return self.classifier(x)

        mini_model = MiniCNN().eval()

        # Specify target_layer by string
        b64_1 = generate_gradcam(
            image_path=sample_retinal_image,
            predicted_class_idx=0,
            model=mini_model,
            target_layer="custom_conv",
        )
        assert b64_1.startswith("iVBORw0KGgo")

        # Specify target_layer by module
        b64_2 = generate_gradcam(
            image_path=sample_retinal_image,
            predicted_class_idx=1,
            model=mini_model,
            target_layer=mini_model.custom_conv,
        )
        assert b64_2.startswith("iVBORw0KGgo")

    def test_missing_checkpoint_when_no_model_supplied(
        self, sample_retinal_image, tmp_path
    ):
        missing_ckpt = str(tmp_path / "missing_model.pth")
        with pytest.raises(FileNotFoundError) as exc:
            generate_gradcam(
                image_path=sample_retinal_image,
                predicted_class_idx=0,
                model=None,
                weights_path=missing_ckpt,
            )
        assert "checkpoint not found" in str(exc.value).lower()

    def test_checkpoint_loaded_when_valid_path_supplied(
        self, sample_retinal_image, tmp_path
    ):
        valid_ckpt = str(tmp_path / "dummy_ckpt.pth")
        model = build_densenet201_model(num_classes=NUM_CLASSES)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "classes": CLASS_NAMES,
            },
            valid_ckpt,
        )

        b64 = generate_gradcam(
            image_path=sample_retinal_image,
            predicted_class_idx=0,
            model=None,
            weights_path=valid_ckpt,
        )
        assert isinstance(b64, str)
        assert b64.startswith("iVBORw0KGgo")


class TestGradCamResourceSafetyAndEdgeCases:
    """Verifies hook removal, gradient safety, division-by-zero prevention, and mode preservation."""

    def test_constant_zero_activation_map_avoids_division_by_zero(
        self, sample_retinal_image
    ):
        # A model that produces zero gradients w.r.t the target layer
        class ZeroGradModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential()
                self.features.denseblock4 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
                self.classifier = nn.Linear(8, 4)

            def forward(self, x):
                _ = self.features.denseblock4(x)
                # Output does not depend on denseblock4
                return torch.zeros(x.shape[0], 4, requires_grad=True)

        zero_model = ZeroGradModel().eval()
        # Should not raise ZeroDivisionError
        b64 = generate_gradcam(
            image_path=sample_retinal_image,
            predicted_class_idx=0,
            model=zero_model,
        )
        assert isinstance(b64, str)

    def test_hook_cleanup_on_success(
        self, sample_retinal_image, controlled_densenet_model
    ):
        target = controlled_densenet_model.features.denseblock4
        assert len(target._forward_hooks) == 0
        assert len(target._backward_hooks) == 0

        _ = generate_gradcam(
            image_path=sample_retinal_image,
            predicted_class_idx=0,
            model=controlled_densenet_model,
        )

        # After execution, hooks MUST be cleaned up
        assert len(target._forward_hooks) == 0
        assert len(target._backward_hooks) == 0

    def test_hook_cleanup_on_failure(
        self, sample_retinal_image, controlled_densenet_model
    ):
        target = controlled_densenet_model.features.denseblock4

        # Simulate exception during forward pass
        with patch.object(
            controlled_densenet_model, "forward", side_effect=RuntimeError("Forward crash")
        ):
            with pytest.raises(RuntimeError):
                generate_gradcam(
                    image_path=sample_retinal_image,
                    predicted_class_idx=0,
                    model=controlled_densenet_model,
                )

        # Hooks MUST be removed despite exception
        assert len(target._forward_hooks) == 0
        assert len(target._backward_hooks) == 0

    def test_no_unintended_parameter_gradient_accumulation(
        self, sample_retinal_image, controlled_densenet_model
    ):
        # Ensure zero grad before
        controlled_densenet_model.zero_grad(set_to_none=True)

        _ = generate_gradcam(
            image_path=sample_retinal_image,
            predicted_class_idx=0,
            model=controlled_densenet_model,
        )

        # Verify that all parameters have grad is None
        for name, param in controlled_densenet_model.named_parameters():
            assert param.grad is None, f"Parameter {name} retained gradient after Grad-CAM."

    def test_caller_training_mode_preservation(
        self, sample_retinal_image, controlled_densenet_model
    ):
        # Case A: Caller supplied model in training mode
        controlled_densenet_model.train(True)
        assert controlled_densenet_model.training is True
        _ = generate_gradcam(
            image_path=sample_retinal_image,
            predicted_class_idx=0,
            model=controlled_densenet_model,
        )
        assert controlled_densenet_model.training is True

        # Case B: Caller supplied model in eval mode
        controlled_densenet_model.eval()
        assert controlled_densenet_model.training is False
        _ = generate_gradcam(
            image_path=sample_retinal_image,
            predicted_class_idx=0,
            model=controlled_densenet_model,
        )
        assert controlled_densenet_model.training is False

    def test_cpu_device_compatibility(
        self, sample_retinal_image, controlled_densenet_model
    ):
        cpu_device = torch.device("cpu")
        b64 = generate_gradcam(
            image_path=sample_retinal_image,
            predicted_class_idx=3,
            model=controlled_densenet_model,
            device=cpu_device,
        )
        assert isinstance(b64, str)
        assert len(b64) > 0
