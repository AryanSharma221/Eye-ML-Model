"""
Automated Unit Tests for DenseNet201 Eye Disease Architecture, Checkpoint Loader & Inference.

Verifies:
1. Model creation and 4-class output tensor shape.
2. Canonical class ordering and index mapping.
3. Explicit CPU initialization.
4. Automatic CUDA selection when CUDA is available (with mock verification).
5. Missing checkpoint error handling.
6. Valid checkpoint loading (raw state_dict and container dict with metadata).
7. Incompatible classifier head shape detection.
8. Missing or unexpected checkpoint keys detection (strict loading).
9. Corrupted checkpoint file detection.
10. Checkpoint class-order metadata validation when metadata exists.
11. Inference return schema: exact 5 required keys.
12. Probability vector properties (range [0, 1], canonical order, sum ≈ 1.0).
13. Confidence matches selected class probability and correct index mapping.
14. Risk level mapping rules (Normal -> LOW; Non-normal >=0.80 -> HIGH, >=0.50 -> MODERATE, <0.50 -> LOW).
15. Deterministic inference with controlled weights.
16. Input format support: both image path and preprocessed tensor.
17. Full JSON serialization of the inference output.
18. Device-aware in-memory model caching.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import torch
import torch.nn as nn

from ml.eye_model import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    DENSENET201_IN_FEATURES,
    build_densenet201_model,
    load_eye_model_checkpoint,
    get_model,
    get_cached_model,
    clear_model_cache,
    get_default_device,
    compute_risk_level,
    predict_eye_disease,
)


class TestEyeModelArchitecture(unittest.TestCase):
    """Tests covering model construction, forward pass dimensions, and device placement."""

    def test_class_names_and_mappings(self):
        expected_classes = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
        self.assertEqual(CLASS_NAMES, expected_classes)
        self.assertEqual(NUM_CLASSES, 4)

        for idx, name in enumerate(expected_classes):
            self.assertEqual(CLASS_TO_IDX[name], idx)
            self.assertEqual(IDX_TO_CLASS[idx], name)

    def test_model_creation_and_output_shape(self):
        model = build_densenet201_model(num_classes=4)
        self.assertIsInstance(model.classifier, nn.Linear)
        self.assertEqual(model.classifier.in_features, DENSENET201_IN_FEATURES)
        self.assertEqual(model.classifier.out_features, 4)

        # Forward pass on synthetic input tensor [1, 3, 224, 224]
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, torch.Size([1, 4]))
        self.assertTrue(output.is_floating_point())
        self.assertTrue(torch.isfinite(output).all().item())

    def test_cpu_device_initialization(self):
        cpu_device = torch.device("cpu")
        model = get_model(device=cpu_device)
        first_param = next(model.parameters())
        self.assertEqual(first_param.device, cpu_device)

    @patch("torch.cuda.is_available")
    def test_cuda_device_selection_when_available(self, mock_cuda):
        mock_cuda.return_value = True
        device = get_default_device()
        self.assertEqual(device.type, "cuda")

    @patch("torch.cuda.is_available")
    def test_cpu_device_selection_when_cuda_unavailable(self, mock_cuda):
        mock_cuda.return_value = False
        device = get_default_device()
        self.assertEqual(device.type, "cpu")


class TestEyeModelCheckpointLoader(unittest.TestCase):
    """Tests covering checkpoint loading, device mapping, and strict schema validation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cpu_device = torch.device("cpu")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _save_dummy_checkpoint(self, payload: dict, filename: str = "ckpt.pth") -> str:
        path = os.path.join(self.temp_dir, filename)
        torch.save(payload, path)
        return path

    def test_missing_checkpoint_raises_filenotfound(self):
        missing_path = os.path.join(self.temp_dir, "nonexistent.pth")
        model = build_densenet201_model()
        with self.assertRaises(FileNotFoundError) as ctx:
            load_eye_model_checkpoint(model, missing_path, device=self.cpu_device)
        self.assertIn("not found", str(ctx.exception).lower())

    def test_empty_string_path_raises_valueerror(self):
        model = build_densenet201_model()
        with self.assertRaises(ValueError):
            load_eye_model_checkpoint(model, "", device=self.cpu_device)

    def test_directory_path_raises_isadirectoryerror(self):
        model = build_densenet201_model()
        with self.assertRaises(IsADirectoryError):
            load_eye_model_checkpoint(model, self.temp_dir, device=self.cpu_device)

    def test_corrupted_checkpoint_raises_ioerror(self):
        corrupt_path = os.path.join(self.temp_dir, "corrupt.pth")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_PYTORCH_PICKLE_STREAM_123456789")

        model = build_densenet201_model()
        with self.assertRaises(IOError):
            load_eye_model_checkpoint(model, corrupt_path, device=self.cpu_device)

    def test_empty_checkpoint_file_raises_ioerror(self):
        empty_path = os.path.join(self.temp_dir, "empty.pth")
        with open(empty_path, "wb"):
            pass

        model = build_densenet201_model()
        with self.assertRaises(IOError) as ctx:
            load_eye_model_checkpoint(model, empty_path, device=self.cpu_device)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_load_valid_raw_state_dict(self):
        source_model = build_densenet201_model()
        ckpt_path = self._save_dummy_checkpoint(source_model.state_dict(), "raw_weights.pth")

        target_model = build_densenet201_model()
        loaded_model = load_eye_model_checkpoint(target_model, ckpt_path, device=self.cpu_device)
        self.assertIs(loaded_model, target_model)

        for p1, p2 in zip(source_model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_load_valid_container_checkpoint_with_metadata(self):
        source_model = build_densenet201_model()
        payload = {
            "epoch": 10,
            "best_acc": 0.945,
            "classes": CLASS_NAMES,
            "state_dict": source_model.state_dict(),
        }
        ckpt_path = self._save_dummy_checkpoint(payload, "container_weights.pth")

        loaded_model = get_model(weights_path=ckpt_path, device=self.cpu_device)
        self.assertIsNotNone(loaded_model)

        for p1, p2 in zip(source_model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_checkpoint_class_order_mismatch_raises_valueerror(self):
        source_model = build_densenet201_model()
        invalid_classes = ["normal", "cataract", "glaucoma", "diabetic_retinopathy"]
        payload = {
            "classes": invalid_classes,
            "state_dict": source_model.state_dict(),
        }
        ckpt_path = self._save_dummy_checkpoint(payload, "bad_classes.pth")

        model = build_densenet201_model()
        with self.assertRaises(ValueError) as ctx:
            load_eye_model_checkpoint(model, ckpt_path, device=self.cpu_device)
        self.assertIn("class order mismatch", str(ctx.exception).lower())

    def test_incompatible_classifier_shape_raises_valueerror(self):
        source_model = build_densenet201_model()
        state = source_model.state_dict()
        state["classifier.weight"] = torch.randn(10, DENSENET201_IN_FEATURES)
        state["classifier.bias"] = torch.randn(10)
        ckpt_path = self._save_dummy_checkpoint(state, "incompatible_head.pth")

        model = build_densenet201_model()
        with self.assertRaises(ValueError) as ctx:
            load_eye_model_checkpoint(model, ckpt_path, device=self.cpu_device)
        self.assertIn("classifier weight shape mismatch", str(ctx.exception).lower())

    def test_missing_or_unexpected_keys_raises_runtimeerror(self):
        source_model = build_densenet201_model()
        state = source_model.state_dict()
        del state["classifier.bias"]
        ckpt_path = self._save_dummy_checkpoint(state, "missing_key.pth")

        model = build_densenet201_model()
        with self.assertRaises(RuntimeError) as ctx:
            load_eye_model_checkpoint(model, ckpt_path, device=self.cpu_device)
        self.assertIn("incompatible state_dict keys", str(ctx.exception).lower())


class TestEyeModelInference(unittest.TestCase):
    """Tests covering the inference pipeline, risk mapping, caching, and serialization."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cpu_device = torch.device("cpu")
        clear_model_cache()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_model_cache()

    def _create_synthetic_image_file(self, filename: str = "sample.png") -> str:
        path = os.path.join(self.temp_dir, filename)
        img = Image.new("RGB", (250, 250), color=(120, 130, 140))
        img.save(path)
        return path

    def _create_controlled_model(self, target_class_idx: int) -> nn.Module:
        """Creates a mock model that outputs high logit for target_class_idx."""
        class MockClassifierModel(nn.Module):
            def __init__(self, pred_idx: int):
                super().__init__()
                self.pred_idx = pred_idx

            def forward(self, x):
                logits = torch.full((x.size(0), 4), -5.0)
                logits[:, self.pred_idx] = 10.0
                return logits

        model = MockClassifierModel(target_class_idx)
        model.eval()
        return model

    # 1. Output key contract
    def test_inference_returns_exact_contract_keys(self):
        model = self._create_controlled_model(target_class_idx=0)
        img_path = self._create_synthetic_image_file()
        result = predict_eye_disease(img_path, model=model, device=self.cpu_device)

        expected_keys = {
            "prediction",
            "class_index",
            "confidence",
            "risk_level",
            "all_probabilities",
        }
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertIsInstance(result["prediction"], str)
        self.assertIsInstance(result["class_index"], int)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["risk_level"], str)
        self.assertIsInstance(result["all_probabilities"], dict)

    # 2. Probability properties: length, canonical ordering, range, sum ≈ 1.0
    def test_probability_distribution_properties(self):
        model = self._create_controlled_model(target_class_idx=2)
        dummy_tensor = torch.randn(1, 3, 224, 224)
        result = predict_eye_disease(dummy_tensor, model=model, device=self.cpu_device)

        prob_dict = result["all_probabilities"]
        self.assertEqual(list(prob_dict.keys()), CLASS_NAMES)
        self.assertEqual(len(prob_dict), 4)

        prob_values = list(prob_dict.values())
        for p in prob_values:
            self.assertTrue(0.0 <= p <= 1.0)

        # Sum of probabilities should be ~1.0
        self.assertAlmostEqual(sum(prob_values), 1.0, places=3)

    # 3. Correct predicted class and matching confidence
    def test_predicted_class_and_confidence_matching(self):
        # Target index 1: "diabetic_retinopathy"
        model = self._create_controlled_model(target_class_idx=1)
        dummy_tensor = torch.randn(1, 3, 224, 224)
        result = predict_eye_disease(dummy_tensor, model=model, device=self.cpu_device)

        self.assertEqual(result["prediction"], "diabetic_retinopathy")
        self.assertEqual(result["class_index"], 1)
        self.assertGreater(result["confidence"], 0.99)
        self.assertEqual(result["confidence"], result["all_probabilities"]["diabetic_retinopathy"])

    # 4. Risk level mapping rules
    def test_risk_level_mapping_rules(self):
        # Normal is always LOW risk regardless of confidence
        self.assertEqual(compute_risk_level("normal", 0.99), "LOW")
        self.assertEqual(compute_risk_level("normal", 0.50), "LOW")

        # Non-normal classes
        self.assertEqual(compute_risk_level("glaucoma", 0.85), "HIGH")
        self.assertEqual(compute_risk_level("glaucoma", 0.80), "HIGH")
        self.assertEqual(compute_risk_level("cataract", 0.79), "MODERATE")
        self.assertEqual(compute_risk_level("cataract", 0.50), "MODERATE")
        self.assertEqual(compute_risk_level("diabetic_retinopathy", 0.49), "LOW")
        self.assertEqual(compute_risk_level("diabetic_retinopathy", 0.20), "LOW")

    # 5. CPU inference
    def test_cpu_inference_execution(self):
        model = self._create_controlled_model(target_class_idx=3)  # normal
        dummy_tensor = torch.randn(1, 3, 224, 224)
        result = predict_eye_disease(dummy_tensor, model=model, device=self.cpu_device)
        self.assertEqual(result["prediction"], "normal")
        self.assertEqual(result["risk_level"], "LOW")

    # 6. Preprocessed tensor vs. file path inputs
    def test_input_variants_path_and_tensor(self):
        model = self._create_controlled_model(target_class_idx=0)
        img_path = self._create_synthetic_image_file()

        # Call with path
        res1 = predict_eye_disease(img_path, model=model, device=self.cpu_device)
        self.assertEqual(res1["prediction"], "cataract")

        # Call with tensor
        tensor = torch.randn(1, 3, 224, 224)
        res2 = predict_eye_disease(tensor, model=model, device=self.cpu_device)
        self.assertEqual(res2["prediction"], "cataract")

    # 7. Invalid tensor shape raises ValueError
    def test_invalid_tensor_shape_raises_valueerror(self):
        model = self._create_controlled_model(target_class_idx=0)
        invalid_tensor = torch.randn(1, 1, 100, 100)
        with self.assertRaises(ValueError):
            predict_eye_disease(invalid_tensor, model=model, device=self.cpu_device)

    # 8. Missing and corrupted checkpoint files fail clearly
    def test_missing_checkpoint_fails_clearly(self):
        dummy_tensor = torch.randn(1, 3, 224, 224)
        missing_path = os.path.join(self.temp_dir, "nonexistent.pth")
        with self.assertRaises(FileNotFoundError):
            predict_eye_disease(dummy_tensor, weights_path=missing_path, device=self.cpu_device)

    def test_corrupted_checkpoint_fails_clearly(self):
        corrupt_path = os.path.join(self.temp_dir, "bad.pth")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_WEIGHT_FILE")

        dummy_tensor = torch.randn(1, 3, 224, 224)
        with self.assertRaises(IOError):
            predict_eye_disease(dummy_tensor, weights_path=corrupt_path, device=self.cpu_device)

    # 9. JSON serialization of the output
    def test_json_serializability(self):
        model = self._create_controlled_model(target_class_idx=2)
        dummy_tensor = torch.randn(1, 3, 224, 224)
        result = predict_eye_disease(dummy_tensor, model=model, device=self.cpu_device)

        serialized = json.dumps(result)
        self.assertIsInstance(serialized, str)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["prediction"], "glaucoma")
        self.assertEqual(deserialized["class_index"], 2)

    # 10. Device-aware in-memory model caching
    def test_model_caching_and_clearing(self):
        source_model = build_densenet201_model()
        ckpt_path = os.path.join(self.temp_dir, "cache_test.pth")
        torch.save(source_model.state_dict(), ckpt_path)

        m1 = get_cached_model(weights_path=ckpt_path, device=self.cpu_device)
        m2 = get_cached_model(weights_path=ckpt_path, device=self.cpu_device)
        self.assertIs(m1, m2)

        clear_model_cache()
        m3 = get_cached_model(weights_path=ckpt_path, device=self.cpu_device)
        self.assertIsNot(m1, m3)


if __name__ == "__main__":
    unittest.main()
