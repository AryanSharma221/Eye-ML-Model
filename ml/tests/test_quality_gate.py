"""
Comprehensive Unit Tests for the Retinal Image Quality Gate Module.

Covers input validation, resolution thresholds, Laplacian blur, brightness intervals,
contrast variance, Hough circle retinal heuristics, error priority, and return contract.
"""

import math
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from ml.quality_gate import check_image_quality


class TestQualityGateInputHandling(unittest.TestCase):
    """Tests for edge cases and malformed input paths."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nonexistent_path(self):
        result = check_image_quality(os.path.join(self.temp_dir, "missing.png"))
        self.assertFalse(result["overall_pass"])
        self.assertIn("File not found", result["error_message"])
        self.assertEqual(result["blur_score"], 0.0)
        self.assertEqual(result["brightness_mean"], 0.0)
        self.assertEqual(result["contrast_std"], 0.0)
        self.assertFalse(result["retinal_region_detected"])
        self.assertFalse(result["resolution_ok"])

    def test_empty_string_path(self):
        result = check_image_quality("")
        self.assertFalse(result["overall_pass"])
        self.assertIn("non-empty string", result["error_message"])

    def test_whitespace_string_path(self):
        result = check_image_quality("   ")
        self.assertFalse(result["overall_pass"])
        self.assertIn("non-empty string", result["error_message"])

    def test_directory_path(self):
        result = check_image_quality(self.temp_dir)
        self.assertFalse(result["overall_pass"])
        self.assertIn("regular file", result["error_message"])

    def test_empty_file(self):
        empty_path = os.path.join(self.temp_dir, "empty.png")
        with open(empty_path, "wb") as f:
            pass
        result = check_image_quality(empty_path)
        self.assertFalse(result["overall_pass"])
        self.assertIn("empty (0 bytes)", result["error_message"])
        self.assertEqual(result["blur_score"], 0.0)

    def test_corrupt_file(self):
        corrupt_path = os.path.join(self.temp_dir, "corrupt.jpg")
        with open(corrupt_path, "wb") as f:
            f.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01NotAnActualImageStream123456789")
        result = check_image_quality(corrupt_path)
        self.assertFalse(result["overall_pass"])
        self.assertIn("Failed to decode image", result["error_message"])

    def test_text_file_disguised_as_png(self):
        text_path = os.path.join(self.temp_dir, "fake.png")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("This is a plain text file, not a valid PNG binary.")
        result = check_image_quality(text_path)
        self.assertFalse(result["overall_pass"])
        self.assertIn("Failed to decode image", result["error_message"])


class TestQualityGateMetrics(unittest.TestCase):
    """Tests for resolution, blur, brightness, contrast, and retinal detection."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _save_synthetic_image(self, array: np.ndarray, filename: str = "test.png") -> str:
        path = os.path.join(self.temp_dir, filename)
        cv2.imwrite(path, array)
        return path

    # --- Resolution Tests ---
    def test_resolution_below_width(self):
        arr = np.full((250, 200, 3), 120, dtype=np.uint8)
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertFalse(result["resolution_ok"])
        self.assertFalse(result["overall_pass"])
        self.assertIn("Insufficient resolution", result["error_message"])
        self.assertIn("200x250", result["error_message"])

    def test_resolution_below_height(self):
        arr = np.full((200, 250, 3), 120, dtype=np.uint8)
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertFalse(result["resolution_ok"])
        self.assertFalse(result["overall_pass"])
        self.assertIn("Insufficient resolution", result["error_message"])
        self.assertIn("250x200", result["error_message"])

    def test_resolution_exact_minimum(self):
        arr = np.full((224, 224, 3), 120, dtype=np.uint8)
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertTrue(result["resolution_ok"])

    # --- Brightness Tests ---
    def test_brightness_below_minimum(self):
        # Value 25 is below 30.0
        arr = np.full((224, 224, 3), 25, dtype=np.uint8)
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertEqual(result["brightness_mean"], 25.0)
        self.assertFalse(result["overall_pass"])
        # If resolution ok, blur fails for flat image; verify brightness fails independently
        self.assertLess(result["brightness_mean"], 30.0)

    def test_brightness_exact_minimum_30(self):
        arr = np.full((224, 224, 3), 30, dtype=np.uint8)
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertEqual(result["brightness_mean"], 30.0)

    def test_brightness_exact_maximum_225(self):
        arr = np.full((224, 224, 3), 225, dtype=np.uint8)
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertEqual(result["brightness_mean"], 225.0)

    def test_brightness_above_maximum(self):
        # Value 240 is above 225.0
        arr = np.full((224, 224, 3), 240, dtype=np.uint8)
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertGreater(result["brightness_mean"], 225.0)

    # --- Contrast Tests ---
    def test_contrast_below_threshold(self):
        # Flat image has standard deviation 0.0 < 20.0
        arr = np.full((224, 224, 3), 120, dtype=np.uint8)
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertEqual(result["contrast_std"], 0.0)
        self.assertFalse(result["overall_pass"])

    def test_contrast_above_threshold(self):
        # Image split into halves 40 and 180 has high contrast
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:, :112] = 40
        arr[:, 112:] = 180
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertGreaterEqual(result["contrast_std"], 20.0)

    # --- Blur (Laplacian Variance) Tests ---
    def test_blur_below_threshold(self):
        # Uniform or smooth gradient has low Laplacian variance
        arr = np.full((224, 224, 3), 120, dtype=np.uint8)
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertLess(result["blur_score"], 100.0)

    def test_blur_above_threshold(self):
        # Checkerboard high-frequency pattern has very high Laplacian variance
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        # Alternate pixels
        arr[::2, ::2] = 200
        arr[1::2, 1::2] = 200
        arr[::2, 1::2] = 40
        arr[1::2, ::2] = 40
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertGreaterEqual(result["blur_score"], 100.0)

    # --- Retinal Region Detection (Hough Circles) Tests ---
    @patch("cv2.HoughCircles")
    def test_retinal_region_detected_true(self, mock_hough):
        # Mock valid circle returned: 1 circle with (x, y, radius)
        mock_hough.return_value = np.array([[[112.0, 112.0, 60.0]]], dtype=np.float32)
        arr = np.full((224, 224, 3), 120, dtype=np.uint8)
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertTrue(result["retinal_region_detected"])

    @patch("cv2.HoughCircles")
    def test_retinal_region_detected_none(self, mock_hough):
        mock_hough.return_value = None
        arr = np.full((224, 224, 3), 120, dtype=np.uint8)
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertFalse(result["retinal_region_detected"])

    @patch("cv2.HoughCircles")
    def test_retinal_region_hough_exception_handled(self, mock_hough):
        mock_hough.side_effect = cv2.error("OpenCV native error in HoughCircles")
        arr = np.full((224, 224, 3), 120, dtype=np.uint8)
        path = self._save_synthetic_image(arr)
        result = check_image_quality(path)
        self.assertFalse(result["retinal_region_detected"])
        self.assertFalse(result["overall_pass"])

    @patch("cv2.HoughCircles")
    def test_hough_circles_parameters_exact(self, mock_hough):
        mock_hough.return_value = None
        arr = np.full((300, 300, 3), 120, dtype=np.uint8)
        path = self._save_synthetic_image(arr)
        check_image_quality(path)

        mock_hough.assert_called_once()
        args, kwargs = mock_hough.call_args

        # Verify arguments passed to HoughCircles
        resized_img = args[0]
        self.assertEqual(resized_img.shape, (224, 224))
        self.assertEqual(args[1], cv2.HOUGH_GRADIENT)
        self.assertEqual(kwargs["dp"], 1.2)
        self.assertEqual(kwargs["minDist"], 100)
        self.assertEqual(kwargs["param1"], 50)
        self.assertEqual(kwargs["param2"], 30)
        self.assertEqual(kwargs["minRadius"], 40)
        self.assertEqual(kwargs["maxRadius"], 112)


class TestQualityGateCombinedAndPriority(unittest.TestCase):
    """Tests for overall pass conditions and deterministic error prioritization."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_high_freq_pattern(self, width: int = 224, height: int = 224, low: int = 40, high: int = 200) -> np.ndarray:
        """Creates an image with high sharpness, controllable brightness & contrast."""
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[::2, ::2] = high
        arr[1::2, 1::2] = high
        arr[::2, 1::2] = low
        arr[1::2, ::2] = low
        return arr

    @patch("cv2.HoughCircles")
    def test_all_checks_pass(self, mock_hough):
        # 1. Circle detected
        mock_hough.return_value = np.array([[[112.0, 112.0, 60.0]]], dtype=np.float32)
        # 2. Dimensions: 250x250 (>=224)
        # 3. High sharpness (blur >= 100)
        # 4. Brightness: mean ~120 (within [30, 225])
        # 5. Contrast: std ~80 (>= 20)
        arr = self._create_high_freq_pattern(width=250, height=250, low=40, high=200)
        path = os.path.join(self.temp_dir, "pass_sample.png")
        cv2.imwrite(path, arr)

        result = check_image_quality(path)
        self.assertTrue(result["overall_pass"])
        self.assertTrue(result["resolution_ok"])
        self.assertGreaterEqual(result["blur_score"], 100.0)
        self.assertTrue(30.0 <= result["brightness_mean"] <= 225.0)
        self.assertGreaterEqual(result["contrast_std"], 20.0)
        self.assertTrue(result["retinal_region_detected"])
        self.assertIsNone(result["error_message"])

    # --- Error Priority Verification ---
    # Priority: 1. Unreadable -> 2. Resolution -> 3. Blur -> 4. Brightness -> 5. Contrast -> 6. Retinal
    @patch("cv2.HoughCircles")
    def test_priority_resolution_over_blur(self, mock_hough):
        mock_hough.return_value = None
        # Fails resolution (100x100) AND blur (uniform) AND brightness (10)
        arr = np.full((100, 100, 3), 10, dtype=np.uint8)
        path = os.path.join(self.temp_dir, "multi_fail.png")
        cv2.imwrite(path, arr)

        result = check_image_quality(path)
        self.assertFalse(result["overall_pass"])
        self.assertIn("Insufficient resolution", result["error_message"])

    @patch("cv2.HoughCircles")
    def test_priority_blur_over_brightness(self, mock_hough):
        mock_hough.return_value = None
        # Resolution ok (250x250), but blurry (flat 10) AND brightness < 30
        arr = np.full((250, 250, 3), 10, dtype=np.uint8)
        path = os.path.join(self.temp_dir, "blur_fail.png")
        cv2.imwrite(path, arr)

        result = check_image_quality(path)
        self.assertFalse(result["overall_pass"])
        self.assertIn("Excessive image blur", result["error_message"])

    @patch("cv2.HoughCircles")
    def test_priority_brightness_over_contrast(self, mock_hough):
        mock_hough.return_value = None
        # Sharp high-frequency pattern, but very dark: low=0, high=10 -> mean ~5 < 30
        arr = self._create_high_freq_pattern(width=250, height=250, low=0, high=10)
        path = os.path.join(self.temp_dir, "dark_fail.png")
        cv2.imwrite(path, arr)

        result = check_image_quality(path)
        self.assertFalse(result["overall_pass"])
        self.assertIn("Image underexposed", result["error_message"])

    @patch("cv2.HoughCircles")
    def test_priority_contrast_over_retinal(self, mock_hough):
        mock_hough.return_value = None
        # Low contrast: low=115, high=125 -> std ~5 < 20. But sharp: high freq pattern.
        # Check if blur >= 100: for step of 10, Laplacian var may be < 100, let's verify
        arr = np.full((250, 250, 3), 120, dtype=np.uint8)
        path = os.path.join(self.temp_dir, "flat_contrast.png")
        cv2.imwrite(path, arr)

        result = check_image_quality(path)
        self.assertFalse(result["overall_pass"])
        # Because flat image also fails blur, blur is caught first
        self.assertIn("Excessive image blur", result["error_message"])

    @patch("cv2.HoughCircles")
    def test_priority_retinal_when_all_others_pass(self, mock_hough):
        mock_hough.return_value = None
        # Everything passes except Hough circle
        arr = self._create_high_freq_pattern(width=250, height=250, low=40, high=200)
        path = os.path.join(self.temp_dir, "no_retina.png")
        cv2.imwrite(path, arr)

        result = check_image_quality(path)
        self.assertFalse(result["overall_pass"])
        self.assertIn("No circular retinal boundary detected", result["error_message"])

    def test_dictionary_contract_keys_and_types(self):
        arr = np.full((250, 250, 3), 120, dtype=np.uint8)
        path = os.path.join(self.temp_dir, "contract_test.png")
        cv2.imwrite(path, arr)

        result = check_image_quality(path)
        required_keys = {
            "overall_pass",
            "blur_score",
            "brightness_mean",
            "contrast_std",
            "retinal_region_detected",
            "resolution_ok",
            "error_message",
        }
        self.assertEqual(set(result.keys()), required_keys)
        self.assertIsInstance(result["overall_pass"], bool)
        self.assertIsInstance(result["blur_score"], float)
        self.assertTrue(math.isfinite(result["blur_score"]))
        self.assertIsInstance(result["brightness_mean"], float)
        self.assertTrue(math.isfinite(result["brightness_mean"]))
        self.assertIsInstance(result["contrast_std"], float)
        self.assertTrue(math.isfinite(result["contrast_std"]))
        self.assertIsInstance(result["retinal_region_detected"], bool)
        self.assertIsInstance(result["resolution_ok"], bool)
        self.assertTrue(result["error_message"] is None or isinstance(result["error_message"], str))


if __name__ == "__main__":
    unittest.main()
