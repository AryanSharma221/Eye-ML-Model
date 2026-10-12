"""
Focused Unit and Integration Tests for Input Validation and Data Contracts.

Validates:
1. Image input boundaries (empty paths, directories, unsupported formats, corrupted bytes,
   channel counts, grayscale/RGBA, non-finite pixel values, dimension guardrails).
2. Vital-sign input hardening (missing values, string types, boolean types, NaN, Inf,
   unphysical limits, assess_* standalone contract validation).
3. Eye model output contract validation (4 classes, finite, sum to 1.0, index alignment,
   confidence consistency, failure detection on invalid outputs).
4. Grad-CAM input and output contract verification (class index bounds, PNG magic bytes,
   Base64 roundtrip validity, hook cleanup invariance).
5. Standalone CLI contract validation with valid and invalid parameters.
"""

import base64
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    SUPPORTED_IMAGE_EXTENSIONS,
    get_default_checkpoint_path,
    resolve_device,
)
from ml.dataset_loader import load_split_manifest
from ml.exceptions import (
    GradCAMGenerationError,
    InvalidImageInputError,
    ModelInferenceError,
    PhysiologicalTypeValidationError,
    PhysiologicalValidationError,
)
from ml.eye_model import (
    get_cached_model,
    predict_eye_disease,
)
from ml.gradcam import generate_gradcam
from ml.physio_model import (
    STATUS_ABNORMAL,
    STATUS_HIGHER,
    STATUS_LOWER,
    STATUS_NORMAL,
    analyze_vitals,
    assess_heart_rate,
    assess_overall_status,
    assess_spo2,
    assess_temperature,
    compute_vital_risk_level,
)
from ml.preprocessing import preprocess_image
from ml.quality_gate import check_image_quality
from ml.screening_pipeline import main as pipeline_cli_main, run_screening


class TestImageInputHardening(unittest.TestCase):
    """Verifies edge case handling and boundaries for image inputs across modules."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_image(self, filename: str, size=(250, 250), mode="RGB", color=(100, 150, 200)) -> str:
        img = Image.new(mode, size, color)
        path = os.path.join(self.temp_dir, filename)
        img.save(path)
        return path

    def test_missing_and_empty_paths(self):
        # None path
        q_res = check_image_quality(None)  # type: ignore
        self.assertFalse(q_res["overall_pass"])
        self.assertIn("non-empty string", q_res["error_message"])

        # Empty string
        q_res_empty = check_image_quality("")
        self.assertFalse(q_res_empty["overall_pass"])
        self.assertIn("non-empty string", q_res_empty["error_message"])

        # Whitespace
        q_res_ws = check_image_quality("   ")
        self.assertFalse(q_res_ws["overall_pass"])
        self.assertIn("non-empty string", q_res_ws["error_message"])

        # Preprocessing raises TypeError/ValueError
        with self.assertRaises(TypeError):
            preprocess_image(None)  # type: ignore
        with self.assertRaises(ValueError):
            preprocess_image("")
        with self.assertRaises(ValueError):
            preprocess_image("   ")

    def test_directory_passed_as_image(self):
        dir_path = os.path.join(self.temp_dir, "test_subdir")
        os.makedirs(dir_path, exist_ok=True)

        q_res = check_image_quality(dir_path)
        self.assertFalse(q_res["overall_pass"])
        self.assertIn("regular file", q_res["error_message"].lower())

        with self.assertRaises(IsADirectoryError):
            preprocess_image(dir_path)

        with self.assertRaises(IsADirectoryError):
            generate_gradcam(dir_path, predicted_class_idx=0)

        # Screening pipeline traps directory safely
        s_res = run_screening(image_path=dir_path)
        self.assertEqual(s_res["eye_screening"]["status"], "error")
        self.assertIn("directory", s_res["eye_screening"]["reason"].lower())

    def test_unsupported_file_extensions(self):
        pdf_file = os.path.join(self.temp_dir, "scan.pdf")
        with open(pdf_file, "wb") as f:
            f.write(b"%PDF-1.4 header bytes")

        q_res = check_image_quality(pdf_file)
        self.assertFalse(q_res["overall_pass"])
        self.assertIn("unsupported file format", q_res["error_message"].lower())

        with self.assertRaises(InvalidImageInputError):
            preprocess_image(pdf_file)

        with self.assertRaises(InvalidImageInputError):
            generate_gradcam(pdf_file, predicted_class_idx=0)

    def test_corrupted_and_truncated_image(self):
        corrupt_file = os.path.join(self.temp_dir, "broken.png")
        with open(corrupt_file, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nGARBAGE_PAYLOAD_NOT_A_PNG")

        q_res = check_image_quality(corrupt_file)
        self.assertFalse(q_res["overall_pass"])
        self.assertIn("failed to decode", q_res["error_message"].lower())

        with self.assertRaises(InvalidImageInputError):
            preprocess_image(corrupt_file)

        with self.assertRaises(InvalidImageInputError):
            generate_gradcam(corrupt_file, predicted_class_idx=0)

    def test_grayscale_and_rgba_images_handled_cleanly(self):
        gray_path = self._create_image("test_gray.png", size=(250, 250), mode="L", color=128)
        rgba_path = self._create_image("test_rgba.png", size=(250, 250), mode="RGBA", color=(100, 150, 200, 255))

        # Preprocessing converts both to 3-channel [1, 3, 224, 224]
        tensor_gray = preprocess_image(gray_path)
        self.assertEqual(tensor_gray.shape, torch.Size([1, 3, 224, 224]))
        self.assertTrue(torch.isfinite(tensor_gray).all())

        tensor_rgba = preprocess_image(rgba_path)
        self.assertEqual(tensor_rgba.shape, torch.Size([1, 3, 224, 224]))
        self.assertTrue(torch.isfinite(tensor_rgba).all())

    def test_extremely_large_image_rejected(self):
        huge_path = os.path.join(self.temp_dir, "huge.png")
        huge_canvas = np.zeros((200, 5200, 3), dtype=np.uint8)
        cv2.imwrite(huge_path, huge_canvas)

        q_res = check_image_quality(huge_path)
        self.assertFalse(q_res["overall_pass"])
        self.assertIn("exceed maximum allowable resolution", q_res["error_message"].lower())

        with self.assertRaises(InvalidImageInputError):
            preprocess_image(huge_path)

        with self.assertRaises(InvalidImageInputError):
            generate_gradcam(huge_path, predicted_class_idx=0)

    def test_non_finite_tensor_input_rejected_in_inference(self):
        nan_tensor = torch.randn(1, 3, 224, 224)
        nan_tensor[0, 0, 10, 10] = float("nan")

        with self.assertRaises(ValueError) as ctx:
            predict_eye_disease(nan_tensor)
        self.assertIn("non-finite", str(ctx.exception).lower())

        inf_tensor = torch.randn(1, 3, 224, 224)
        inf_tensor[0, 1, 20, 20] = float("inf")

        with self.assertRaises(ValueError) as ctx:
            predict_eye_disease(inf_tensor)
        self.assertIn("non-finite", str(ctx.exception).lower())

    def test_non_floating_point_tensor_rejected_in_inference(self):
        int_tensor = torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.int64)
        with self.assertRaises(TypeError) as ctx:
            predict_eye_disease(int_tensor)
        self.assertIn("floating-point", str(ctx.exception).lower())


class TestVitalSignInputHardening(unittest.TestCase):
    """Verifies vital sign contract validation and boundary checking."""

    def test_missing_vital_signs(self):
        with self.assertRaises(PhysiologicalTypeValidationError):
            analyze_vitals(None, 98.0, 36.8)  # type: ignore
        with self.assertRaises(PhysiologicalTypeValidationError):
            analyze_vitals(72.0, None, 36.8)  # type: ignore
        with self.assertRaises(PhysiologicalTypeValidationError):
            analyze_vitals(72.0, 98.0, None)  # type: ignore

    def test_incompatible_types_rejected(self):
        with self.assertRaises(PhysiologicalTypeValidationError):
            analyze_vitals("72.0", 98.0, 36.8)  # type: ignore
        with self.assertRaises(PhysiologicalTypeValidationError):
            analyze_vitals([72.0], 98.0, 36.8)  # type: ignore
        with self.assertRaises(PhysiologicalTypeValidationError):
            analyze_vitals(72.0, {"val": 98.0}, 36.8)  # type: ignore

    def test_boolean_values_specifically_rejected(self):
        # Python bool is an int subclass; must reject explicitly
        with self.assertRaises(PhysiologicalTypeValidationError):
            analyze_vitals(True, 98.0, 36.8)  # type: ignore
        with self.assertRaises(PhysiologicalTypeValidationError):
            analyze_vitals(72.0, False, 36.8)  # type: ignore
        with self.assertRaises(PhysiologicalTypeValidationError):
            assess_heart_rate(True)  # type: ignore
        with self.assertRaises(PhysiologicalTypeValidationError):
            assess_spo2(False)  # type: ignore
        with self.assertRaises(PhysiologicalTypeValidationError):
            assess_temperature(True)  # type: ignore

    def test_nan_and_infinite_values_rejected(self):
        # NaN
        with self.assertRaises(PhysiologicalValidationError):
            analyze_vitals(float("nan"), 98.0, 36.8)
        with self.assertRaises(PhysiologicalValidationError):
            assess_heart_rate(float("nan"))
        with self.assertRaises(PhysiologicalValidationError):
            assess_spo2(float("nan"))
        with self.assertRaises(PhysiologicalValidationError):
            assess_temperature(float("nan"))

        # Infinity
        with self.assertRaises(PhysiologicalValidationError):
            analyze_vitals(float("inf"), 98.0, 36.8)
        with self.assertRaises(PhysiologicalValidationError):
            analyze_vitals(72.0, float("-inf"), 36.8)
        with self.assertRaises(PhysiologicalValidationError):
            assess_heart_rate(float("inf"))

    def test_values_outside_plausible_limits_rejected(self):
        # Heart rate: [20, 300]
        with self.assertRaises(PhysiologicalValidationError):
            assess_heart_rate(19.9)
        with self.assertRaises(PhysiologicalValidationError):
            assess_heart_rate(300.1)

        # SpO2: [0, 100]
        with self.assertRaises(PhysiologicalValidationError):
            assess_spo2(-0.1)
        with self.assertRaises(PhysiologicalValidationError):
            assess_spo2(100.1)

        # Temp: [25, 45]
        with self.assertRaises(PhysiologicalValidationError):
            assess_temperature(24.9)
        with self.assertRaises(PhysiologicalValidationError):
            assess_temperature(45.1)

    def test_status_and_risk_aggregation_input_validation(self):
        with self.assertRaises(PhysiologicalTypeValidationError):
            assess_overall_status(123, STATUS_NORMAL, STATUS_NORMAL)  # type: ignore
        with self.assertRaises(PhysiologicalValidationError):
            assess_overall_status("InvalidStatus", STATUS_NORMAL, STATUS_NORMAL)

        with self.assertRaises(PhysiologicalTypeValidationError):
            compute_vital_risk_level(456)  # type: ignore
        with self.assertRaises(PhysiologicalValidationError):
            compute_vital_risk_level("UnrecognizedStatus")


class TestModelOutputContractValidation(unittest.TestCase):
    """Verifies that model inference strictly validates output contracts."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = os.path.join(_PROJECT_ROOT, "ml", "weights", "eye_model.pth")
        cls.has_checkpoint = os.path.exists(cls.checkpoint_path)
        cls.device = resolve_device(None)

    def test_real_model_inference_contract_invariants(self):
        if not self.has_checkpoint:
            self.skipTest("Trained checkpoint not present on disk.")

        manifest_path = os.path.join(_PROJECT_ROOT, "ml", "dataset", "split_manifest.json")
        manifest_data = load_split_manifest(manifest_path)
        test_records = manifest_data.get("splits", {}).get("test", [])
        sample_img = None
        for r in test_records:
            fp = r.get("filepath", "")
            if os.path.exists(fp) and check_image_quality(fp)["overall_pass"]:
                sample_img = fp
                break

        if not sample_img:
            self.skipTest("No passing test image found in dataset split.")

        res = predict_eye_disease(
            image_input=sample_img,
            weights_path=self.checkpoint_path,
            device=self.device,
            use_cache=True,
        )

        # 1. Output contains all required fields
        self.assertIn("prediction", res)
        self.assertIn("class_index", res)
        self.assertIn("confidence", res)
        self.assertIn("risk_level", res)
        self.assertIn("all_probabilities", res)

        # 2. Exactly four class probabilities are present
        probs_dict = res["all_probabilities"]
        self.assertEqual(len(probs_dict), 4)
        for cls_name in CLASS_NAMES:
            self.assertIn(cls_name, probs_dict)

        # 3. All probabilities finite and in [0, 1]
        for cls_name, p in probs_dict.items():
            self.assertTrue(math.isfinite(p))
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

        # 4. Probabilities sum to 1.0 within tolerance
        prob_sum = sum(probs_dict.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)

        # 5. Class index in [0, 3]
        self.assertIn(res["class_index"], [0, 1, 2, 3])

        # 6. Prediction matches class_index
        self.assertEqual(res["prediction"], CLASS_NAMES[res["class_index"]])

        # 7. Confidence matches probability of predicted class
        self.assertAlmostEqual(res["confidence"], probs_dict[res["prediction"]], places=4)

        # 8. Risk level is one of canonical values
        self.assertIn(res["risk_level"], ["LOW", "MODERATE", "HIGH"])


class TestGradCAMInputOutputValidation(unittest.TestCase):
    """Verifies Grad-CAM input validation and output PNG Base64 integrity."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = os.path.join(_PROJECT_ROOT, "ml", "weights", "eye_model.pth")
        cls.has_checkpoint = os.path.exists(cls.checkpoint_path)
        cls.device = resolve_device(None)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        img = Image.new("RGB", (300, 300), (120, 140, 160))
        self.test_img = os.path.join(self.temp_dir, "gradcam_test.png")
        img.save(self.test_img)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_class_indices(self):
        with self.assertRaises(ValueError):
            generate_gradcam(self.test_img, predicted_class_idx=-1)
        with self.assertRaises(ValueError):
            generate_gradcam(self.test_img, predicted_class_idx=4)
        with self.assertRaises(TypeError):
            generate_gradcam(self.test_img, predicted_class_idx=True)  # type: ignore
        with self.assertRaises(TypeError):
            generate_gradcam(self.test_img, predicted_class_idx="0")  # type: ignore

    def test_invalid_alpha_values(self):
        with self.assertRaises(ValueError):
            generate_gradcam(self.test_img, predicted_class_idx=0, alpha=float("nan"))
        with self.assertRaises(ValueError):
            generate_gradcam(self.test_img, predicted_class_idx=0, alpha=float("inf"))

    def test_output_png_magic_bytes_and_roundtrip(self):
        if not self.has_checkpoint:
            self.skipTest("Trained checkpoint not present on disk.")

        b64_output = generate_gradcam(
            image_path=self.test_img,
            predicted_class_idx=0,
            weights_path=self.checkpoint_path,
            device=self.device,
        )

        self.assertIsInstance(b64_output, str)
        self.assertTrue(len(b64_output) > 100)

        # Decode Base64 and verify PNG signature \x89PNG\r\n\x1a\n
        raw_bytes = base64.b64decode(b64_output, validate=True)
        self.assertEqual(raw_bytes[:8], b"\x89PNG\r\n\x1a\n")

        # Confirm OpenCV can decode visualization back to an image of original dimensions
        nparr = np.frombuffer(raw_bytes, np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape[0], 300)
        self.assertEqual(decoded.shape[1], 300)


class TestPipelineAndCLIContractValidation(unittest.TestCase):
    """Verifies pipeline decoupling and CLI argument validation."""

    def test_pipeline_independence_image_fail_vitals_succeed(self):
        # Non-existent image, valid vitals
        res = run_screening(
            image_path="nonexistent_fundus_photo_xyz.png",
            heart_rate=75.0,
            spo2=98.0,
            temperature=36.8,
        )
        self.assertEqual(res["eye_screening"]["status"], "error")
        self.assertEqual(res["vitals_screening"]["status"], "success")
        self.assertEqual(res["status"], "partial")

    def test_pipeline_independence_vitals_fail_image_succeed(self):
        # Valid synthetic image fixture, invalid vitals
        temp_dir = tempfile.mkdtemp()
        try:
            img = Image.new("RGB", (300, 300), (120, 130, 140))
            valid_path = os.path.join(temp_dir, "test.png")
            img.save(valid_path)

            res = run_screening(
                image_path=valid_path,
                heart_rate=500.0,  # Unphysical
                spo2=98.0,
                temperature=36.8,
            )
            self.assertEqual(res["vitals_screening"]["status"], "error")
            # Eye screening either rejected by QG or evaluated, never aborted by vitals
            self.assertIn(res["eye_screening"]["status"], ["rejected", "success", "error"])
            self.assertIn(res["status"], ["partial", "rejected", "error"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cli_execution_with_nan_vitals_fails(self):
        # Executing CLI with --hr nan should exit with code 1
        cmd = [
            sys.executable,
            "-m",
            "ml.screening_pipeline",
            "--hr",
            "nan",
            "--spo2",
            "98.0",
            "--temp",
            "36.8",
        ]
        proc = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)

    def test_cli_missing_all_arguments_returns_exit_code_2(self):
        cmd = [sys.executable, "-m", "ml.screening_pipeline"]
        proc = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("No inputs provided", proc.stderr)


if __name__ == "__main__":
    unittest.main()
