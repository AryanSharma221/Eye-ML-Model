"""
Automated Smoke and Regression Tests for Trained Checkpoint Inference Compatibility.

Verifies:
1. Physical existence and strict loadability of trained checkpoints.
2. End-to-end screening workflow (quality gate -> preprocessing -> inference -> gradcam).
3. Public contract compliance and JSON serializability.
4. Probability distribution properties (finite, sum to ~1.0, bounded [0, 1]).
5. Rejection of invalid inputs and failure distinction for missing checkpoints.
"""

import json
import os
import pytest
import torch

from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES, load_split_manifest
from ml.eye_model import (
    DEFAULT_CHECKPOINT_PATH,
    build_densenet201_model,
    clear_model_cache,
    load_eye_model_checkpoint,
    predict_eye_disease,
    screen_eye_image,
)
from ml.quality_gate import check_image_quality

BEST_CHECKPOINT_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "weights", "eye_densenet201_best.pth"
)
MANIFEST_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "dataset", "split_manifest.json"
)


@pytest.mark.smoke
class TestTrainedCheckpointArtifacts:
    """Verifies that the trained model weights exist and are structurally valid."""

    def test_checkpoint_files_exist_and_non_empty(self):
        assert os.path.exists(BEST_CHECKPOINT_PATH), f"Missing {BEST_CHECKPOINT_PATH}"
        assert os.path.getsize(BEST_CHECKPOINT_PATH) > 50 * 1024 * 1024  # > 50 MB

        assert os.path.exists(DEFAULT_CHECKPOINT_PATH), f"Missing {DEFAULT_CHECKPOINT_PATH}"
        assert os.path.getsize(DEFAULT_CHECKPOINT_PATH) > 50 * 1024 * 1024

    def test_checkpoint_strict_loader_compatibility(self):
        model = build_densenet201_model(num_classes=NUM_CLASSES)
        loaded = load_eye_model_checkpoint(model, BEST_CHECKPOINT_PATH, device=torch.device("cpu"))
        assert loaded is not None

        # Verify checkpoint payload metadata
        ckpt_data = torch.load(BEST_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        assert ckpt_data.get("classes") == list(CLASS_NAMES)
        assert ckpt_data.get("best_acc") is not None
        assert ckpt_data["best_acc"] > 0.80


@pytest.mark.smoke
class TestInferenceCompatibilitySmoke:
    """Verifies real inference predictions on held-out test images."""

    @pytest.fixture(autouse=True)
    def clean_cache(self):
        clear_model_cache()
        yield
        clear_model_cache()

    def test_end_to_end_smoke_inference_on_held_out_image(self):
        if not os.path.exists(MANIFEST_PATH):
            pytest.skip("Manifest not found")

        manifest = load_split_manifest(MANIFEST_PATH)
        test_records = manifest["splits"]["test"]

        # Find first image that passes quality gate
        passing_image = None
        for r in test_records:
            if os.path.exists(r["filepath"]):
                q = check_image_quality(r["filepath"])
                if q["overall_pass"]:
                    passing_image = r["filepath"]
                    break

        if not passing_image:
            pytest.skip("No quality-passing image found in test split sample")

        # Execute screen_eye_image with Grad-CAM
        result = screen_eye_image(passing_image, weights_path=BEST_CHECKPOINT_PATH, include_gradcam=True)

        # 1. Output contract validation
        assert result["status"] == "success"
        assert result["prediction"] in CLASS_NAMES
        assert 0 <= result["class_index"] < NUM_CLASSES
        assert CLASS_NAMES[result["class_index"]] == result["prediction"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["risk_level"] in ["LOW", "MODERATE", "HIGH"]

        # 2. Probability distribution checks
        probs = result["all_probabilities"]
        assert len(probs) == NUM_CLASSES
        for name in CLASS_NAMES:
            assert name in probs
            assert 0.0 <= probs[name] <= 1.0

        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 1e-3, f"Probabilities sum to {prob_sum}"
        assert abs(result["confidence"] - probs[result["prediction"]]) < 1e-3

        # 3. Grad-CAM checks
        assert "gradcam_image" in result
        assert isinstance(result["gradcam_image"], str)
        assert len(result["gradcam_image"]) > 1000

        # 4. JSON serializability
        serialized = json.dumps(result)
        assert serialized is not None

    def test_missing_checkpoint_raises_filenotfound(self, tmp_path):
        dummy_img = str(tmp_path / "dummy.png")
        import numpy as np
        import cv2
        cv2.imwrite(dummy_img, np.zeros((224, 224, 3), dtype=np.uint8))

        with pytest.raises(FileNotFoundError):
            predict_eye_disease(dummy_img, weights_path="nonexistent_checkpoint_file.pth")
