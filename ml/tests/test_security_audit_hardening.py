"""
Focused Security and Hardening Test Suite.

Validates defensive safeguards against:
1. Unsafe path handling and embedded null-byte injections.
2. Protection of critical project assets (.pth weights, .py files, dataset manifests) against accidental overwrites.
3. Crash-resilient atomic file writing and automatic cleanup of temporary files on failure.
4. Recursive telemetry sanitization preventing leaks of biometric image blobs and physiological vital-sign data.
5. Error message sanitization preventing sensitive filesystem or username disclosure.
6. Safe checkpoint loading with strict validation of architecture, parameter shapes, and weights_only security.
"""

import json
import os
import stat
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from ml.config import PathsConfig
from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES
from ml.exceptions import (
    CheckpointCorruptError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    InvalidImageInputError,
)
from ml.eye_model import build_densenet201_model, load_eye_model_checkpoint
from ml.gradcam import generate_gradcam
from ml.monitoring import sanitize_details, sanitize_error_message, sanitize_path
from ml.preprocessing import preprocess_image
from ml.quality_gate import check_image_quality
from ml.screening_pipeline import main as screening_cli_main, run_screening
from ml.security import (
    PROTECTED_EXTENSIONS,
    PROTECTED_FILENAMES,
    safe_atomic_write_file,
    safe_atomic_write_json,
    validate_safe_path,
)


class TestPathHandlingAndNullBytes(unittest.TestCase):
    """Verifies that malicious or malformed paths do not trigger unhandled exceptions."""

    def test_quality_gate_null_byte_rejection(self):
        """Quality gate rejects null-byte paths cleanly as sentinel failures without crashing."""
        res = check_image_quality("invalid\x00_path.jpg")
        self.assertFalse(res["overall_pass"])
        self.assertIn("null byte", res["error_message"].lower())

    def test_preprocessing_null_byte_rejection(self):
        """Preprocessing raises InvalidImageInputError on null-byte paths."""
        with self.assertRaises(InvalidImageInputError) as ctx:
            preprocess_image("image\x00_test.png")
        self.assertIn("null byte", str(ctx.exception).lower())

    def test_checkpoint_loader_null_byte_rejection(self):
        """Checkpoint loader raises CheckpointCorruptError on null-byte paths."""
        model = build_densenet201_model(num_classes=4)
        with self.assertRaises(CheckpointCorruptError) as ctx:
            load_eye_model_checkpoint(model, "weights\x00_bad.pth", device=torch.device("cpu"))
        self.assertIn("null byte", str(ctx.exception).lower())

    def test_gradcam_null_byte_rejection(self):
        """Grad-CAM generator raises ValueError on null-byte paths."""
        with self.assertRaises(ValueError) as ctx:
            generate_gradcam("retina\x00.jpg", predicted_class_idx=0)
        self.assertIn("null byte", str(ctx.exception).lower())

    def test_screening_pipeline_null_byte_rejection(self):
        """run_screening gracefully reports errors for null-byte image inputs."""
        res = run_screening(image_path="test\x00_image.jpg")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["eye_screening"]["status"], "error")
        self.assertIn("null byte", res["eye_screening"]["reason"].lower())

    def test_directory_passed_as_image_file_handled(self):
        """Passing a directory instead of a file produces a clean handled error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = check_image_quality(tmp_dir)
            self.assertFalse(res["overall_pass"])
            self.assertTrue(
                "regular file" in res["error_message"].lower() or "directory" in res["error_message"].lower()
            )


class TestProtectedAssetOverwritePrevention(unittest.TestCase):
    """Verifies that generic output writers cannot overwrite checkpoints, manifests, or code."""

    def test_validate_safe_path_blocks_protected_extensions(self):
        """Targeting .pth, .pt, or .py files for generic output is blocked."""
        for ext in (".pth", ".pt", ".py", ".pyc", ".git"):
            with self.assertRaises(ValueError) as ctx:
                validate_safe_path(f"output_result{ext}", disallow_critical=True)
            self.assertIn("protected", str(ctx.exception).lower())

    def test_validate_safe_path_blocks_protected_filenames(self):
        """Targeting critical manifest and model files by name is blocked."""
        for fname in ("split_manifest.json", "eye_model.pth", "eye_densenet201_best.pth"):
            with self.assertRaises(ValueError) as ctx:
                validate_safe_path(fname, disallow_critical=True)
            self.assertIn("protected", str(ctx.exception).lower())

    def test_cli_output_cannot_overwrite_checkpoint(self):
        """Screening CLI refuses to overwrite a .pth weights file."""
        exit_code = screening_cli_main([
            "--hr", "72",
            "--spo2", "98",
            "--temp", "36.8",
            "--output", "corrupted_weights.pth",
        ])
        self.assertEqual(exit_code, 1)

    def test_cli_output_cannot_overwrite_manifest(self):
        """Screening CLI refuses to overwrite split_manifest.json."""
        exit_code = screening_cli_main([
            "--hr", "72",
            "--spo2", "98",
            "--temp", "36.8",
            "--output", "split_manifest.json",
        ])
        self.assertEqual(exit_code, 1)


class TestSafeAtomicFileWrites(unittest.TestCase):
    """Verifies atomic write guarantees and cleanup of temporary files."""

    def test_atomic_write_text_and_json(self):
        """Atomic writing of text and JSON completes cleanly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            text_target = os.path.join(tmp_dir, "test_out.txt")
            safe_atomic_write_file(text_target, "Hello Security", mode="w")
            self.assertTrue(os.path.isfile(text_target))
            with open(text_target, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "Hello Security")

            json_target = os.path.join(tmp_dir, "test_out.json")
            safe_atomic_write_json(json_target, {"status": "ok", "value": 42})
            self.assertTrue(os.path.isfile(json_target))
            with open(json_target, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertEqual(data["status"], "ok")

    def test_temp_file_cleanup_on_write_failure(self):
        """If an I/O error occurs during write, temporary files are cleanly removed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_path = os.path.join(tmp_dir, "failed_target.json")

            # Simulate failure during json.dumps or file write
            with patch("ml.security.open", side_effect=IOError("Simulated disk error")):
                with self.assertRaises(IOError):
                    safe_atomic_write_file(target_path, "data")

            # Verify no partial files or .tmp files remain in directory
            lingering_files = os.listdir(tmp_dir)
            self.assertEqual(lingering_files, [])

    def test_atomic_write_refuses_existing_directory(self):
        """Writing to an existing directory path raises IsADirectoryError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(IsADirectoryError):
                safe_atomic_write_file(tmp_dir, "content")


class TestSensitiveDataAndLogSanitization(unittest.TestCase):
    """Verifies that PII, raw vitals, and image payloads are sanitized from telemetry."""

    def test_recursive_nested_details_sanitization(self):
        """Nested dictionaries and lists are deeply sanitized."""
        raw_details = {
            "patient_info": {
                "name": "Jane Doe",
                "patient_id": "PT-99812",
                "ssn": "123-45-6789",
            },
            "credentials": {
                "api_key": "sk-live-secret-token-12345",
                "auth_bearer": "bearer 98765",
            },
            "vitals_payload": {
                "heart_rate": 78.5,
                "spo2": 97.0,
                "temperature": 36.9,
            },
            "nested_list": [
                {"token": "secret-abc"},
                {"image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."},
            ],
            "safe_metadata": {
                "status": "SUCCESS",
                "model": "DenseNet201",
            },
        }

        sanitized = sanitize_details(raw_details)

        # 1. PII Redaction
        self.assertEqual(sanitized["patient_info"]["name"], "[REDACTED_PII]")
        self.assertEqual(sanitized["patient_info"]["patient_id"], "[REDACTED_PII]")
        self.assertEqual(sanitized["patient_info"]["ssn"], "[REDACTED_PII]")

        # 2. Secret Redaction
        self.assertEqual(sanitized["credentials"]["api_key"], "[REDACTED]")
        self.assertEqual(sanitized["credentials"]["auth_bearer"], "[REDACTED]")

        # 3. Physiological Data Masking
        self.assertEqual(sanitized["vitals_payload"]["heart_rate"], "[PHYSIO_DATA_MASKED]")
        self.assertEqual(sanitized["vitals_payload"]["spo2"], "[PHYSIO_DATA_MASKED]")
        self.assertEqual(sanitized["vitals_payload"]["temperature"], "[PHYSIO_DATA_MASKED]")

        # 4. Nested List Sanitization
        self.assertEqual(sanitized["nested_list"][0]["token"], "[REDACTED]")
        self.assertEqual(sanitized["nested_list"][1]["image_data"], "[BLOB_DATA_MASKED]")

        # 5. Non-sensitive data preserved
        self.assertEqual(sanitized["safe_metadata"]["status"], "SUCCESS")

    def test_error_message_path_sanitization(self):
        """Error messages strip absolute Windows and POSIX directory structures."""
        win_err = "Cannot access C:\\Users\\Administrator\\OphthalmicData\\patient_01.jpg: Access Denied"
        sanitized_win = sanitize_error_message(win_err)
        self.assertNotIn("Administrator", sanitized_win)
        self.assertIn("[PATH]\\patient_01.jpg", sanitized_win)

        posix_err = "Failed reading /home/jdoe/clinical_vault/scans/retina_42.png: File Corrupted"
        sanitized_posix = sanitize_error_message(posix_err)
        self.assertNotIn("jdoe", sanitized_posix)
        self.assertIn("[PATH]/retina_42.png", sanitized_posix)

        secret_err = "Authentication failed with api_key: sk-proj-123456789abcdef"
        sanitized_secret = sanitize_error_message(secret_err)
        self.assertNotIn("sk-proj-123456789abcdef", sanitized_secret)
        self.assertIn("api_key=[REDACTED]", sanitized_secret)


class TestCheckpointIntegrityVerification(unittest.TestCase):
    """Verifies that model loading strictly rejects invalid checkpoints."""

    def test_corrupted_checkpoint_fails_cleanly(self):
        """Corrupted/truncated binary raises CheckpointCorruptError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_file = os.path.join(tmp_dir, "corrupt.pth")
            with open(corrupt_file, "wb") as f:
                f.write(b"NOT_A_VALID_PYTORCH_FILE_HEADER")

            model = build_densenet201_model(num_classes=4)
            with self.assertRaises(CheckpointCorruptError):
                load_eye_model_checkpoint(model, corrupt_file, device=torch.device("cpu"))

    def test_empty_checkpoint_fails_cleanly(self):
        """Empty 0-byte checkpoint raises CheckpointCorruptError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_file = os.path.join(tmp_dir, "empty.pth")
            with open(empty_file, "wb") as f:
                pass

            model = build_densenet201_model(num_classes=4)
            with self.assertRaises(CheckpointCorruptError):
                load_eye_model_checkpoint(model, empty_file, device=torch.device("cpu"))

    def test_incompatible_architecture_metadata_rejected(self):
        """Checkpoint declaring incompatible architecture (e.g. ResNet50) is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_arch_file = os.path.join(tmp_dir, "resnet.pth")
            model = build_densenet201_model(num_classes=4)
            payload = {
                "state_dict": model.state_dict(),
                "classes": list(CLASS_NAMES),
                "architecture": "ResNet50",
            }
            torch.save(payload, bad_arch_file)

            with self.assertRaises(CheckpointIncompatibleError) as ctx:
                load_eye_model_checkpoint(model, bad_arch_file, device=torch.device("cpu"))
            self.assertIn("architecture mismatch", str(ctx.exception).lower())

    def test_mismatched_class_count_rejected(self):
        """Checkpoint with incompatible classifier head dimension is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_head_file = os.path.join(tmp_dir, "bad_head.pth")
            model = build_densenet201_model(num_classes=4)
            state = model.state_dict()
            state["classifier.weight"] = torch.randn(8, 1920)  # 8 classes instead of 4
            state["classifier.bias"] = torch.randn(8)
            payload = {
                "state_dict": state,
                "classes": list(CLASS_NAMES),
                "architecture": "DenseNet201",
            }
            torch.save(payload, bad_head_file)

            with self.assertRaises(CheckpointIncompatibleError) as ctx:
                load_eye_model_checkpoint(model, bad_head_file, device=torch.device("cpu"))
            self.assertIn("classifier weight shape mismatch", str(ctx.exception).lower())

    def test_original_trained_checkpoint_preserved(self):
        """Verifies that the canonical trained checkpoint file exists and has not been mutated."""
        paths = PathsConfig.from_env()
        ckpt_path = paths.checkpoint_path
        self.assertTrue(os.path.isfile(ckpt_path), f"Checkpoint missing: {ckpt_path}")
        self.assertEqual(os.path.getsize(ckpt_path), 73754057)


if __name__ == "__main__":
    unittest.main()
