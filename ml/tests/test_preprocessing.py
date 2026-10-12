"""
Automated Unit Tests for the Retinal Image Preprocessing Module.

Verifies:
1. Valid RGB image returns a PyTorch tensor.
2. Output shape is exactly [1, 3, 224, 224].
3. Output dtype is floating point (torch.float32).
4. Output contains all finite values.
5. Grayscale input is correctly converted to 3-channel RGB.
6. RGBA input is correctly converted to 3-channel RGB.
7. Non-square images are resized to 224x224.
8. Invalid file paths raise meaningful exceptions.
9. Corrupted and 0-byte image files raise meaningful exceptions.
10. Normalization matches ImageNet mean and std within numerical tolerances.
11. Repeated calls do not accumulate additional batch dimensions.
12. Output tensor is structurally compatible with DenseNet201 without downloading weights.
"""

import os
import shutil
import tempfile
import unittest
import torch
from torchvision import models
from PIL import Image

from ml.preprocessing import preprocess_image, IMAGENET_MEAN, IMAGENET_STD


class TestImagePreprocessing(unittest.TestCase):
    """Test suite covering preprocessing transformation behavior and error handling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_image(
        self,
        size=(300, 300),
        mode="RGB",
        color=(128, 128, 128),
        filename="test.png",
    ) -> str:
        """Helper to create and save a synthetic test image."""
        img = Image.new(mode, size, color)
        path = os.path.join(self.temp_dir, filename)
        img.save(path)
        return path

    # 1. Valid RGB image returns a tensor
    def test_valid_rgb_returns_tensor(self):
        path = self._create_image((300, 300), "RGB", (100, 150, 200), "rgb.jpg")
        result = preprocess_image(path)
        self.assertIsInstance(result, torch.Tensor)

    # 2. Output shape is exactly [1, 3, 224, 224]
    def test_output_shape(self):
        path = self._create_image((400, 400), "RGB", (50, 60, 70), "shape.png")
        tensor = preprocess_image(path)
        self.assertEqual(tensor.shape, torch.Size([1, 3, 224, 224]))

    # 3. Output dtype is floating point
    def test_output_dtype_floating_point(self):
        path = self._create_image((250, 250), "RGB", (10, 20, 30), "dtype.png")
        tensor = preprocess_image(path)
        self.assertTrue(tensor.is_floating_point())
        self.assertEqual(tensor.dtype, torch.float32)

    # 4. Output contains finite values
    def test_output_values_finite(self):
        path = self._create_image((300, 300), "RGB", (128, 128, 128), "finite.png")
        tensor = preprocess_image(path)
        self.assertTrue(torch.isfinite(tensor).all().item())

    # 5. Grayscale input is converted correctly
    def test_grayscale_input_converted_to_3_channels(self):
        path = self._create_image((250, 250), "L", 180, "gray.png")
        tensor = preprocess_image(path)
        self.assertEqual(tensor.shape, torch.Size([1, 3, 224, 224]))
        # In a grayscale image, the unnormalized RGB channels are identical.
        # After ImageNet normalization, channels differ only because of channel-specific mean/std.
        self.assertTrue(torch.isfinite(tensor).all().item())

    # 6. RGBA input is converted correctly
    def test_rgba_input_converted_to_3_channels(self):
        path = self._create_image((250, 250), "RGBA", (100, 150, 200, 255), "rgba.png")
        tensor = preprocess_image(path)
        self.assertEqual(tensor.shape, torch.Size([1, 3, 224, 224]))
        self.assertEqual(tensor.size(1), 3)

    # 7. Images with non-square dimensions are resized correctly
    def test_non_square_dimensions_resized(self):
        path_wide = self._create_image((600, 200), "RGB", (70, 80, 90), "wide.png")
        tensor_wide = preprocess_image(path_wide)
        self.assertEqual(tensor_wide.shape, torch.Size([1, 3, 224, 224]))

        path_tall = self._create_image((200, 700), "RGB", (70, 80, 90), "tall.png")
        tensor_tall = preprocess_image(path_tall)
        self.assertEqual(tensor_tall.shape, torch.Size([1, 3, 224, 224]))

    # 8. Invalid file paths raise meaningful exceptions
    def test_invalid_paths_raise_exceptions(self):
        # Non-existent file
        missing_path = os.path.join(self.temp_dir, "missing_image.png")
        with self.assertRaises(FileNotFoundError):
            preprocess_image(missing_path)

        # Empty string path
        with self.assertRaises(ValueError):
            preprocess_image("")

        # Whitespace path
        with self.assertRaises(ValueError):
            preprocess_image("   ")

        # Non-string argument
        with self.assertRaises(TypeError):
            preprocess_image(12345)

        # Directory path
        with self.assertRaises(IsADirectoryError):
            preprocess_image(self.temp_dir)

    # 9. Corrupted image files raise meaningful exceptions
    def test_corrupted_and_empty_files_raise_exceptions(self):
        # Empty file (0 bytes)
        empty_path = os.path.join(self.temp_dir, "empty.png")
        with open(empty_path, "wb"):
            pass
        with self.assertRaises(IOError) as ctx:
            preprocess_image(empty_path)
        self.assertIn("empty", str(ctx.exception).lower())

        # Corrupted header / truncated file
        corrupt_path = os.path.join(self.temp_dir, "corrupt.jpg")
        with open(corrupt_path, "wb") as f:
            f.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00garbagebytesnotanimage")
        with self.assertRaises(IOError):
            preprocess_image(corrupt_path)

    # 10. Normalization matches specified ImageNet mean and std within numerical tolerances
    def test_normalization_values_match_imagenet(self):
        # Create uniform solid color image: R=128, G=64, B=200
        r_val, g_val, b_val = 128, 64, 200
        path = self._create_image((224, 224), "RGB", (r_val, g_val, b_val), "norm.png")
        tensor = preprocess_image(path)

        # Expected unnormalized float values in [0.0, 1.0]:
        expected_r = r_val / 255.0
        expected_g = g_val / 255.0
        expected_b = b_val / 255.0

        # Expected normalized values: (channel - mean) / std
        expected_norm_r = (expected_r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        expected_norm_g = (expected_g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        expected_norm_b = (expected_b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

        # Extract actual tensor channel values (all pixels in channel should be uniform)
        actual_norm_r = tensor[0, 0, 0, 0].item()
        actual_norm_g = tensor[0, 1, 0, 0].item()
        actual_norm_b = tensor[0, 2, 0, 0].item()

        self.assertAlmostEqual(actual_norm_r, expected_norm_r, places=3)
        self.assertAlmostEqual(actual_norm_g, expected_norm_g, places=3)
        self.assertAlmostEqual(actual_norm_b, expected_norm_b, places=3)

    # 11. Repeated calls do not accidentally add extra batch dimensions
    def test_repeated_calls_stable_shape(self):
        path1 = self._create_image((250, 250), "RGB", (100, 100, 100), "rep1.png")
        path2 = self._create_image((350, 350), "RGB", (200, 200, 200), "rep2.png")

        t1 = preprocess_image(path1)
        t2 = preprocess_image(path2)
        t3 = preprocess_image(path1)

        self.assertEqual(t1.shape, torch.Size([1, 3, 224, 224]))
        self.assertEqual(t2.shape, torch.Size([1, 3, 224, 224]))
        self.assertEqual(t3.shape, torch.Size([1, 3, 224, 224]))

    # 12. Output is suitable as input to DenseNet201 without downloading model weights
    def test_densenet201_structural_compatibility(self):
        path = self._create_image((250, 250), "RGB", (120, 120, 120), "densenet.png")
        input_tensor = preprocess_image(path)

        # Instantiate DenseNet201 architecture with weights=None (no network download)
        model = models.densenet201(weights=None)
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)

        # Standard DenseNet201 ImageNet classification head outputs [1, 1000]
        self.assertEqual(output.shape, torch.Size([1, 1000]))
        self.assertTrue(output.is_floating_point())
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
