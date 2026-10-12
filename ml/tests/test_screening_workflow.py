"""
Tests for the Integrated Eye-Screening Workflow (Quality Gate + Preprocessing + Inference).

Verifies:
1. Images failing quality gate (blur, brightness, contrast, resolution, retinal contour,
   missing/corrupt files) are IMMEDIATELY rejected and NEVER proceed to preprocessing or inference.
2. Images passing quality gate proceed to preprocessing, model forward pass, and return
   the complete 4-class classification contract.
3. Quality rejection payload strictly matches the backend contract {"status": "rejected", ...}.
4. Model-loading and checkpoint failures raise explicit exceptions, distinct from quality rejections.
5. Canonical 4-class ordering (cataract, diabetic_retinopathy, glaucoma, normal) is strictly preserved.
"""

import base64
import hashlib
import os
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

import ml.eye_model
from ml.eye_model import (
    CLASS_NAMES,
    NUM_CLASSES,
    build_densenet201_model,
    predict_eye_disease,
    screen_eye_image,
)
from ml.preprocessing import preprocess_image
from ml.quality_gate import check_image_quality


@pytest.fixture
def valid_retinal_image(tmp_path) -> str:
    """Creates a synthetic retinal fundus image passing all 5 quality gate checks."""
    img_path = str(tmp_path / "valid_retina.png")
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)

    # Fundus disk (radius 100 inside 300x300, mean brightness ~130)
    cv2.circle(canvas, (150, 150), 100, (140, 130, 120), -1)

    # High frequency details for blur score (Laplacian variance >= 100) and contrast (std >= 20)
    np.random.seed(42)
    noise = np.random.randint(-40, 40, (300, 300, 3), dtype=np.int16)
    noisy_canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Optic disc feature
    cv2.circle(noisy_canvas, (190, 150), 22, (230, 220, 200), -1)

    # Blood vessel lines
    cv2.line(noisy_canvas, (190, 150), (100, 100), (40, 30, 160), 2)
    cv2.line(noisy_canvas, (190, 150), (100, 200), (40, 30, 160), 2)
    cv2.line(noisy_canvas, (190, 150), (130, 150), (40, 30, 160), 2)

    cv2.imwrite(img_path, noisy_canvas)
    # Sanity check: Ensure fixture actually passes quality gate
    q = check_image_quality(img_path)
    assert q["overall_pass"] is True, f"Fixture failed quality check: {q}"
    return img_path


@pytest.fixture
def blurry_retinal_image(tmp_path) -> str:
    """Creates a blurry retinal fundus image failing the blur check (blur < 100)."""
    img_path = str(tmp_path / "blurry_retina.png")
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 100, (140, 130, 120), -1)
    # Severe Gaussian blur drops Laplacian variance well below 100
    blurry = cv2.GaussianBlur(canvas, (35, 35), sigmaX=15)
    cv2.imwrite(img_path, blurry)
    q = check_image_quality(img_path)
    assert q["overall_pass"] is False
    assert q["blur_score"] < 100.0
    return img_path


@pytest.fixture
def dark_image(tmp_path) -> str:
    """Creates an underexposed image failing brightness check (< 30)."""
    img_path = str(tmp_path / "dark_retina.png")
    canvas = np.full((300, 300, 3), 15, dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 100, (25, 25, 25), -1)
    cv2.imwrite(img_path, canvas)
    q = check_image_quality(img_path)
    assert q["overall_pass"] is False
    assert q["brightness_mean"] < 30.0
    return img_path


@pytest.fixture
def small_image(tmp_path) -> str:
    """Creates an image failing minimum resolution check (< 224x224)."""
    img_path = str(tmp_path / "small_image.png")
    canvas = np.full((120, 120, 3), 128, dtype=np.uint8)
    cv2.imwrite(img_path, canvas)
    q = check_image_quality(img_path)
    assert q["overall_pass"] is False
    assert q["resolution_ok"] is False
    return img_path


@pytest.fixture
def low_contrast_image(tmp_path) -> str:
    """Creates an image failing contrast check (std < 20)."""
    img_path = str(tmp_path / "low_contrast.png")
    canvas = np.full((300, 300, 3), 128, dtype=np.uint8)
    # Very slight difference: std well below 20
    cv2.circle(canvas, (150, 150), 100, (132, 132, 132), -1)
    cv2.imwrite(img_path, canvas)
    q = check_image_quality(img_path)
    assert q["overall_pass"] is False
    assert q["contrast_std"] < 20.0
    return img_path



@pytest.fixture
def dummy_eye_model() -> nn.Module:
    """Returns a deterministic DenseNet201 model on CPU for testing inference."""
    torch.manual_seed(1234)
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    model.eval()
    return model


class TestScreeningQualityRejection:
    """Verifies that low quality or invalid images are strictly stopped before ML inference."""

    def test_blurry_image_rejected_without_preprocessing_or_inference(
        self, blurry_retinal_image, dummy_eye_model
    ):
        with patch("ml.eye_model.preprocess_image") as mock_prep:
            with patch.object(dummy_eye_model, "forward", wraps=dummy_eye_model.forward) as mock_forward:
                result = screen_eye_image(
                    image_path=blurry_retinal_image,
                    model=dummy_eye_model,
                )

        # 1. Verification of backend rejection contract
        assert result["status"] == "rejected"
        assert "reason" in result
        assert "quality_checks" in result
        assert "blur" in result["reason"].lower()

        # 2. Strict guarantee: disease prediction must NOT be returned
        assert "prediction" not in result
        assert "class_index" not in result
        assert "confidence" not in result
        assert "all_probabilities" not in result
        assert "risk_level" not in result

        # 3. Strict guarantee: Preprocessing and forward pass were NEVER called
        mock_prep.assert_not_called()
        mock_forward.assert_not_called()

    def test_dark_image_rejected_without_inference(self, dark_image, dummy_eye_model):
        with patch("ml.eye_model.preprocess_image") as mock_prep:
            result = screen_eye_image(image_path=dark_image, model=dummy_eye_model)

        assert result["status"] == "rejected"
        assert result["quality_checks"]["overall_pass"] is False
        assert result["quality_checks"]["brightness_mean"] < 30.0
        assert "prediction" not in result
        mock_prep.assert_not_called()

    def test_low_contrast_image_rejected_without_inference(
        self, low_contrast_image, dummy_eye_model
    ):
        with patch("ml.eye_model.preprocess_image") as mock_prep:
            result = screen_eye_image(image_path=low_contrast_image, model=dummy_eye_model)

        assert result["status"] == "rejected"
        assert result["quality_checks"]["overall_pass"] is False
        assert result["quality_checks"]["contrast_std"] < 20.0
        assert "prediction" not in result
        mock_prep.assert_not_called()

    def test_small_resolution_image_rejected_without_inference(
        self, small_image, dummy_eye_model
    ):
        with patch("ml.eye_model.preprocess_image") as mock_prep:
            result = screen_eye_image(image_path=small_image, model=dummy_eye_model)

        assert result["status"] == "rejected"
        assert "resolution" in result["reason"].lower()
        assert result["quality_checks"]["overall_pass"] is False
        assert "prediction" not in result
        mock_prep.assert_not_called()

    def test_no_retinal_region_rejected_without_inference(
        self, valid_retinal_image, dummy_eye_model
    ):
        with patch("cv2.HoughCircles", return_value=None):
            with patch("ml.eye_model.preprocess_image") as mock_prep:
                result = screen_eye_image(
                    image_path=valid_retinal_image, model=dummy_eye_model
                )

        assert result["status"] == "rejected"
        assert "retinal" in result["reason"].lower()
        assert result["quality_checks"]["retinal_region_detected"] is False
        assert result["quality_checks"]["overall_pass"] is False
        assert "prediction" not in result
        mock_prep.assert_not_called()

    def test_missing_file_rejected_without_inference(self, tmp_path, dummy_eye_model):
        missing_path = str(tmp_path / "does_not_exist.png")
        with patch("ml.eye_model.preprocess_image") as mock_prep:
            result = screen_eye_image(image_path=missing_path, model=dummy_eye_model)

        assert result["status"] == "rejected"
        assert "not found" in result["reason"].lower()
        assert result["quality_checks"]["overall_pass"] is False
        mock_prep.assert_not_called()

    def test_corrupted_file_rejected_without_inference(self, tmp_path, dummy_eye_model):
        corrupt_path = str(tmp_path / "corrupt.png")
        with open(corrupt_path, "wb") as f:
            f.write(b"not a valid png binary content")

        with patch("ml.eye_model.preprocess_image") as mock_prep:
            result = screen_eye_image(image_path=corrupt_path, model=dummy_eye_model)

        assert result["status"] == "rejected"
        assert result["quality_checks"]["overall_pass"] is False
        mock_prep.assert_not_called()

    def test_empty_zero_byte_file_rejected_without_inference(
        self, tmp_path, dummy_eye_model
    ):
        empty_path = str(tmp_path / "empty.png")
        with open(empty_path, "wb") as f:
            pass

        with patch("ml.eye_model.preprocess_image") as mock_prep:
            result = screen_eye_image(image_path=empty_path, model=dummy_eye_model)

        assert result["status"] == "rejected"
        assert "empty" in result["reason"].lower()
        assert result["quality_checks"]["overall_pass"] is False
        mock_prep.assert_not_called()


class TestScreeningQualityPassAndInference:
    """Verifies that quality-passing images proceed to preprocessing and 4-class classification."""

    def test_valid_image_succeeds_and_returns_complete_contract(
        self, valid_retinal_image, dummy_eye_model
    ):
        result = screen_eye_image(
            image_path=valid_retinal_image,
            model=dummy_eye_model,
        )

        # 1. Top-level status
        assert result["status"] == "success"

        # 2. Disease classification contract keys
        assert result["prediction"] in CLASS_NAMES
        assert isinstance(result["class_index"], int)
        assert 0 <= result["class_index"] < NUM_CLASSES
        assert CLASS_NAMES[result["class_index"]] == result["prediction"]

        # 3. Confidence & Risk Level
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["risk_level"] in ["HIGH", "MODERATE", "LOW"]

        # 4. Probability distribution across 4 canonical classes
        assert isinstance(result["all_probabilities"], dict)
        assert list(result["all_probabilities"].keys()) == CLASS_NAMES
        prob_sum = sum(result["all_probabilities"].values())
        assert pytest.approx(prob_sum, abs=0.01) == 1.0
        assert (
            result["all_probabilities"][result["prediction"]]
            == pytest.approx(result["confidence"], abs=1e-3)
        )

        # 5. Quality checks payload included
        assert result["quality_checks"]["overall_pass"] is True
        assert result["quality_checks"]["error_message"] is None
        assert result["quality_checks"]["resolution_ok"] is True
        assert result["quality_checks"]["retinal_region_detected"] is True

    def test_quality_pass_invokes_preprocessing_and_forward_pass(
        self, valid_retinal_image, dummy_eye_model
    ):
        with patch("ml.eye_model.preprocess_image", wraps=ml.eye_model.preprocess_image) as mock_prep:
            with patch.object(dummy_eye_model, "forward", wraps=dummy_eye_model.forward) as mock_forward:
                result = screen_eye_image(
                    image_path=valid_retinal_image,
                    model=dummy_eye_model,
                )

        assert result["status"] == "success"
        mock_prep.assert_called_once_with(valid_retinal_image)
        mock_forward.assert_called_once()

    def test_screen_eye_image_cached_model_path(self, valid_retinal_image, tmp_path):
        # Save a valid checkpoint to test weights_path resolution
        model = build_densenet201_model(num_classes=NUM_CLASSES)
        ckpt_path = str(tmp_path / "valid_test_ckpt.pth")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "classes": CLASS_NAMES,
            },
            ckpt_path,
        )

        result = screen_eye_image(
            image_path=valid_retinal_image,
            weights_path=ckpt_path,
            use_cache=True,
        )
        assert result["status"] == "success"
        assert result["prediction"] in CLASS_NAMES


class TestErrorDistinctionAndRouting:
    """Verifies that model/checkpoint failures are strictly distinct from quality failures."""

    def test_missing_checkpoint_raises_filenotfound_not_quality_rejection(
        self, valid_retinal_image, tmp_path
    ):
        missing_ckpt = str(tmp_path / "nonexistent_model.pth")
        # Quality passes, so model loading must be attempted and raise FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            screen_eye_image(
                image_path=valid_retinal_image,
                weights_path=missing_ckpt,
                model=None,
            )
        assert "not found" in str(exc_info.value).lower()

    def test_corrupt_checkpoint_raises_ioerror_not_quality_rejection(
        self, valid_retinal_image, tmp_path
    ):
        corrupt_ckpt = str(tmp_path / "corrupt_model.pth")
        with open(corrupt_ckpt, "wb") as f:
            f.write(b"not a valid torch checkpoint")

        with pytest.raises(IOError):
            screen_eye_image(
                image_path=valid_retinal_image,
                weights_path=corrupt_ckpt,
                model=None,
            )

    def test_predict_eye_disease_validate_quality_flag_routing(
        self, valid_retinal_image, blurry_retinal_image, dummy_eye_model
    ):
        # When validate_quality is True, routes through screen_eye_image
        rej_result = predict_eye_disease(
            image_input=blurry_retinal_image,
            model=dummy_eye_model,
            validate_quality=True,
        )
        assert rej_result["status"] == "rejected"
        assert "blur" in rej_result["reason"].lower()

        succ_result = predict_eye_disease(
            image_input=valid_retinal_image,
            model=dummy_eye_model,
            validate_quality=True,
        )
        assert succ_result["status"] == "success"
        assert succ_result["prediction"] in CLASS_NAMES

    def test_predict_eye_disease_validate_quality_with_tensor_raises_typeerror(
        self, dummy_eye_model
    ):
        dummy_tensor = torch.zeros(1, 3, 224, 224)
        with pytest.raises(TypeError) as exc_info:
            predict_eye_disease(
                image_input=dummy_tensor,
                model=dummy_eye_model,
                validate_quality=True,
            )
        assert "file path" in str(exc_info.value).lower()


class TestScreeningGradCamIntegration:
    """Verifies end-to-end integration between eye screening inference and Grad-CAM."""

    def test_screen_eye_image_with_include_gradcam_success(
        self, valid_retinal_image, dummy_eye_model
    ):
        result = screen_eye_image(
            image_path=valid_retinal_image,
            model=dummy_eye_model,
            include_gradcam=True,
        )

        # 1. Status and core contract
        assert result["status"] == "success"
        assert result["prediction"] in CLASS_NAMES
        assert isinstance(result["class_index"], int)
        assert result["quality_checks"]["overall_pass"] is True

        # 2. Grad-CAM payload
        assert "gradcam_image" in result
        gradcam_b64 = result["gradcam_image"]
        assert isinstance(gradcam_b64, str)
        assert not gradcam_b64.startswith("data:")

        # 3. Binary PNG validation
        png_bytes = base64.b64decode(gradcam_b64)
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    def test_predicted_class_index_propagation_to_gradcam(
        self, valid_retinal_image, dummy_eye_model
    ):
        with patch("ml.gradcam.generate_gradcam", wraps=ml.gradcam.generate_gradcam) as spy_gradcam:
            result = screen_eye_image(
                image_path=valid_retinal_image,
                model=dummy_eye_model,
                include_gradcam=True,
            )

        # Verify exact class index returned by classification was passed to generate_gradcam
        assert spy_gradcam.called
        call_args = spy_gradcam.call_args
        passed_idx = call_args.kwargs.get("predicted_class_idx")
        if passed_idx is None and len(call_args.args) >= 2:
            passed_idx = call_args.args[1]
        assert passed_idx == result["class_index"]

    def test_consistent_model_instance_reuse(
        self, valid_retinal_image, dummy_eye_model
    ):
        with patch("ml.gradcam.generate_gradcam", wraps=ml.gradcam.generate_gradcam) as spy_gradcam:
            _ = screen_eye_image(
                image_path=valid_retinal_image,
                model=dummy_eye_model,
                include_gradcam=True,
            )

        # Verify the exact same model instance was passed without re-instantiation
        passed_model = spy_gradcam.call_args.kwargs.get("model")
        assert passed_model is dummy_eye_model

    def test_original_image_unmodified_on_disk(
        self, valid_retinal_image, dummy_eye_model
    ):
        with open(valid_retinal_image, "rb") as f:
            hash_before = hashlib.sha256(f.read()).hexdigest()

        _ = screen_eye_image(
            image_path=valid_retinal_image,
            model=dummy_eye_model,
            include_gradcam=True,
        )

        with open(valid_retinal_image, "rb") as f:
            hash_after = hashlib.sha256(f.read()).hexdigest()

        assert hash_before == hash_after, "Grad-CAM modified the original input file on disk!"

    def test_gradcam_failure_behavior_raises_clearly(
        self, valid_retinal_image, dummy_eye_model
    ):
        with patch("ml.gradcam.generate_gradcam", side_effect=RuntimeError("Hook failure")):
            with pytest.raises(RuntimeError) as exc_info:
                screen_eye_image(
                    image_path=valid_retinal_image,
                    model=dummy_eye_model,
                    include_gradcam=True,
                )
            assert "Hook failure" in str(exc_info.value)

    def test_backend_response_contract_compatibility(
        self, valid_retinal_image, dummy_eye_model
    ):
        # Case A: Without Grad-CAM (standard screening output)
        res_no_cam = screen_eye_image(
            image_path=valid_retinal_image,
            model=dummy_eye_model,
            include_gradcam=False,
        )
        expected_keys_no_cam = {
            "status",
            "prediction",
            "class_index",
            "confidence",
            "risk_level",
            "all_probabilities",
            "quality_checks",
        }
        assert set(res_no_cam.keys()) == expected_keys_no_cam

        # Case B: With Grad-CAM (matches POST /api/v1/predict_eye backend payload)
        res_with_cam = screen_eye_image(
            image_path=valid_retinal_image,
            model=dummy_eye_model,
            include_gradcam=True,
        )
        assert "gradcam_image" in res_with_cam
        assert set(res_with_cam.keys()) == expected_keys_no_cam | {"gradcam_image"}

    def test_rejected_image_does_not_execute_gradcam(
        self, blurry_retinal_image, dummy_eye_model
    ):
        with patch("ml.gradcam.generate_gradcam") as mock_cam:
            result = screen_eye_image(
                image_path=blurry_retinal_image,
                model=dummy_eye_model,
                include_gradcam=True,
            )

        assert result["status"] == "rejected"
        assert "gradcam_image" not in result
        mock_cam.assert_not_called()

