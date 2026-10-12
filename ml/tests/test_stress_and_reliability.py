"""
Unit and Integration Tests for ML Pipeline Reliability and Stress Invariants.

Verifies:
1. Model inference determinism across repeated executions.
2. Hook cleanup invariance in Grad-CAM.
3. Concurrency safety and numerical equivalence between sequential and multi-threaded runs.
4. Malformed and corrupted input fault isolation.
5. Logging handler idempotency and telemetry buffer bounding.
6. Existence and schema consistency of stress test reports.
"""

import concurrent.futures
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.config import get_default_checkpoint_path, resolve_device
from ml.dataset_loader import load_split_manifest
from ml.exceptions import CheckpointNotFoundError
from ml.eye_model import get_cached_model, predict_eye_disease
from ml.gradcam import generate_gradcam, get_target_layer
from ml.monitoring import _EVENT_RING_BUFFER, configure_logging, ml_logger
from ml.physio_model import analyze_vitals
from ml.quality_gate import check_image_quality
from ml.screening_pipeline import run_screening


class TestPipelineReliabilityAndStress(unittest.TestCase):
    """Test suite verifying reliability invariants and stress resilience."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = (
            os.path.join(_PROJECT_ROOT, "ml", "weights", "eye_model.pth")
            if os.path.exists(os.path.join(_PROJECT_ROOT, "ml", "weights", "eye_model.pth"))
            else get_default_checkpoint_path()
        )
        cls.manifest_path = os.path.join(_PROJECT_ROOT, "ml", "dataset", "split_manifest.json")
        cls.device = resolve_device(None)

        manifest_data = load_split_manifest(cls.manifest_path)
        test_records = manifest_data.get("splits", {}).get("test", [])
        cls.valid_images = [r["filepath"] for r in test_records if os.path.exists(r.get("filepath", ""))]
        
        cls.sample_image = None
        for img_path in cls.valid_images:
            if check_image_quality(img_path)["overall_pass"]:
                cls.sample_image = img_path
                break

        cls.eval_dir = os.path.join(_PROJECT_ROOT, "ml", "evaluation_results")
        cls.stress_json_path = os.path.join(cls.eval_dir, "stress_test_report.json")
        cls.stress_md_path = os.path.join(cls.eval_dir, "STRESS_TEST_REPORT.md")

    def test_repeated_inference_numerical_determinism(self):
        """Verify repeated inference on the same input yields identical output logits & probs."""
        if not self.sample_image:
            self.skipTest("No sample test image available.")

        runs = []
        for _ in range(5):
            res = predict_eye_disease(
                image_input=self.sample_image,
                weights_path=self.checkpoint_path,
                device=self.device,
                use_cache=True,
            )
            runs.append(res)

        first_pred = runs[0]["prediction"]
        first_conf = runs[0]["confidence"]
        first_probs = runs[0]["all_probabilities"]

        for r in runs[1:]:
            self.assertEqual(r["prediction"], first_pred)
            self.assertAlmostEqual(r["confidence"], first_conf, places=4)
            for k in first_probs:
                self.assertAlmostEqual(r["all_probabilities"][k], first_probs[k], places=4)

    def test_gradcam_hook_registration_and_cleanup_invariance(self):
        """Verify that Grad-CAM registers and safely removes all forward and backward hooks."""
        if not self.sample_image:
            self.skipTest("No sample test image available.")

        active_model = get_cached_model(weights_path=self.checkpoint_path, device=self.device)
        target_layer = get_target_layer(active_model)

        self.assertEqual(len(target_layer._forward_hooks), 0)
        self.assertEqual(len(target_layer._backward_hooks), 0)

        # Execute Grad-CAM
        b64_str = generate_gradcam(
            image_path=self.sample_image,
            predicted_class_idx=0,
            model=active_model,
            device=self.device,
        )
        self.assertTrue(len(b64_str) > 100)

        # Hooks must be cleaned up
        self.assertEqual(len(target_layer._forward_hooks), 0)
        self.assertEqual(len(target_layer._backward_hooks), 0)

        # Model parameters requires_grad must remain False
        for p in active_model.parameters():
            self.assertFalse(p.requires_grad)

    def test_concurrent_screening_requests_consistency(self):
        """Verify concurrent multi-threaded requests return identical results to sequential execution."""
        if not self.sample_image:
            self.skipTest("No sample test image available.")

        # Baseline sequential
        seq_res = run_screening(
            image_path=self.sample_image,
            heart_rate=72.0,
            spo2=98.0,
            temperature=36.8,
            weights_path=self.checkpoint_path,
            device=self.device,
            include_gradcam=True,
            use_cache=True,
        )
        seq_pred = seq_res["eye_screening"]["prediction"]
        seq_conf = seq_res["eye_screening"]["confidence"]

        # 4 concurrent threads
        def _worker():
            return run_screening(
                image_path=self.sample_image,
                heart_rate=72.0,
                spo2=98.0,
                temperature=36.8,
                weights_path=self.checkpoint_path,
                device=self.device,
                include_gradcam=True,
                use_cache=True,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_worker) for _ in range(8)]
            for f in concurrent.futures.as_completed(futures):
                c_res = f.result()
                self.assertEqual(c_res["eye_screening"]["prediction"], seq_pred)
                self.assertAlmostEqual(c_res["eye_screening"]["confidence"], seq_conf, places=3)
                self.assertIsNotNone(c_res["eye_screening"]["gradcam_image"])

    def test_corrupted_and_empty_file_resilience(self):
        """Verify pipeline traps 0-byte and random byte files without unhandled process crashes."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_zero:
            zero_path = f_zero.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_bin:
            f_bin.write(os.urandom(1024))
            bin_path = f_bin.name

        try:
            res_zero = run_screening(image_path=zero_path, weights_path=self.checkpoint_path)
            self.assertIn(res_zero["eye_screening"]["status"], ("rejected", "error"))

            res_bin = run_screening(image_path=bin_path, weights_path=self.checkpoint_path)
            self.assertIn(res_bin["eye_screening"]["status"], ("rejected", "error"))
        finally:
            if os.path.exists(zero_path):
                os.remove(zero_path)
            if os.path.exists(bin_path):
                os.remove(bin_path)

    def test_logging_idempotency_and_ring_buffer(self):
        """Verify configure_logging is idempotent and telemetry ring buffer is capped."""
        configure_logging(level="INFO")
        count1 = len(ml_logger.handlers)
        configure_logging(level="INFO")
        count2 = len(ml_logger.handlers)
        self.assertEqual(count1, count2)

        self.assertLessEqual(len(_EVENT_RING_BUFFER), 500)

    def test_stress_report_artifacts_exist_and_valid(self):
        """Verify that stress_test_report.json and STRESS_TEST_REPORT.md are properly generated."""
        self.assertTrue(os.path.exists(self.stress_json_path))
        self.assertTrue(os.path.exists(self.stress_md_path))

        with open(self.stress_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("baseline", data)
        self.assertIn("repeated_execution", data)
        self.assertIn("concurrency", data)
        self.assertIn("resource_stability", data)
        self.assertEqual(data["resource_stability"]["stability_verdict"], "STABLE")


if __name__ == "__main__":
    unittest.main()
