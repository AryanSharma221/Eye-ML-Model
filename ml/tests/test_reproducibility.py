"""
Unit and Integration Tests for ML Pipeline Reproducibility and Device Consistency.

Verifies:
1. Cross-device numerical parity between CPU and CUDA inference.
2. Same-process determinism across repeated executions.
3. Existence and schema consistency of reproducibility benchmark reports.
4. Portability of device resolution policies and checkpoint paths.
5. Fail-fast error handling when checkpoints are missing or misconfigured.
"""

import json
import math
import os
import sys
import unittest

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.config import CLASS_NAMES, get_default_checkpoint_path, resolve_device
from ml.dataset_loader import load_split_manifest
from ml.exceptions import CheckpointNotFoundError
from ml.eye_model import get_cached_model, predict_eye_disease
from ml.quality_gate import check_image_quality


class TestPipelineReproducibility(unittest.TestCase):
    """Test suite verifying cross-environment numerical consistency and artifact portability."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = os.path.join(_PROJECT_ROOT, "ml", "weights", "eye_model.pth")
        cls.has_checkpoint = os.path.isfile(cls.checkpoint_path)
        cls.has_cuda = torch.cuda.is_available()

        cls.manifest_path = os.path.join(_PROJECT_ROOT, "ml", "dataset", "split_manifest.json")
        manifest_data = load_split_manifest(cls.manifest_path)
        test_records = manifest_data.get("splits", {}).get("test", [])

        cls.sample_image = None
        for r in test_records:
            fp = r.get("filepath", "")
            if os.path.isfile(fp) and check_image_quality(fp)["overall_pass"]:
                cls.sample_image = fp
                break

        cls.eval_dir = os.path.join(_PROJECT_ROOT, "ml", "evaluation_results")
        cls.report_json_path = os.path.join(cls.eval_dir, "reproducibility_report.json")
        cls.report_md_path = os.path.join(cls.eval_dir, "REPRODUCIBILITY_REPORT.md")

    def test_reproducibility_report_artifacts_exist_and_valid(self):
        """Verifies that reproducibility reports are generated and adhere to schema contracts."""
        self.assertTrue(os.path.isfile(self.report_json_path), f"Missing {self.report_json_path}")
        self.assertTrue(os.path.isfile(self.report_md_path), f"Missing {self.report_md_path}")

        with open(self.report_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["metadata"]["status"], "VERIFIED")
        self.assertIn("environment", data)
        self.assertIn("test_cohort", data)
        self.assertIn("mode_benchmarks", data)
        self.assertIn("numerical_consistency", data)
        self.assertIn("portability_checks", data)

        # Check numerical consistency assertions
        cons = data["numerical_consistency"]
        if "cpu_vs_cuda" in cons:
            cpu_cuda = cons["cpu_vs_cuda"]
            self.assertEqual(cpu_cuda["prediction_agreement_pct"], 100.0)
            self.assertLessEqual(cpu_cuda["max_probability_diff"], 0.005)

        self.assertEqual(cons["repeated_same_process"]["prediction_agreement_pct"], 100.0)

    def test_cross_device_cpu_cuda_numerical_consistency(self):
        """Asserts live CPU vs CUDA inference produces identical predictions within tolerance."""
        if not self.has_checkpoint:
            self.skipTest("No checkpoint file present.")
        if not self.has_cuda:
            self.skipTest("CUDA device unavailable on host.")
        if not self.sample_image:
            self.skipTest("No sample image available.")

        # CPU run
        cpu_res = predict_eye_disease(
            image_input=self.sample_image,
            weights_path=self.checkpoint_path,
            device="cpu",
            use_cache=True,
        )

        # CUDA run
        cuda_res = predict_eye_disease(
            image_input=self.sample_image,
            weights_path=self.checkpoint_path,
            device="cuda",
            use_cache=True,
        )

        # Prediction and risk level agreement
        self.assertEqual(cpu_res["prediction"], cuda_res["prediction"])
        self.assertEqual(cpu_res["class_index"], cuda_res["class_index"])
        self.assertEqual(cpu_res["risk_level"], cuda_res["risk_level"])

        # Probability distribution delta
        for cname in CLASS_NAMES:
            p_cpu = cpu_res["all_probabilities"][cname]
            p_cuda = cuda_res["all_probabilities"][cname]
            delta = abs(p_cpu - p_cuda)
            self.assertLess(delta, 0.005, f"Probability delta for {cname} exceeded 0.005: {delta}")

    def test_same_process_repeated_determinism(self):
        """Verifies repeated inference on the same device yields identical outputs."""
        if not self.has_checkpoint or not self.sample_image:
            self.skipTest("Checkpoint or sample image not available.")

        device = resolve_device(None)
        run1 = predict_eye_disease(self.sample_image, weights_path=self.checkpoint_path, device=device, use_cache=True)
        run2 = predict_eye_disease(self.sample_image, weights_path=self.checkpoint_path, device=device, use_cache=True)

        self.assertEqual(run1["prediction"], run2["prediction"])
        self.assertAlmostEqual(run1["confidence"], run2["confidence"], places=4)
        for cname in CLASS_NAMES:
            self.assertAlmostEqual(run1["all_probabilities"][cname], run2["all_probabilities"][cname], places=4)

    def test_device_resolution_policies(self):
        """Confirms that device resolution strictly adheres to requested parameters."""
        cpu = resolve_device("cpu")
        self.assertEqual(str(cpu), "cpu")

        if self.has_cuda:
            cuda = resolve_device("cuda")
            self.assertEqual(str(cuda), "cuda")
            auto = resolve_device("auto")
            self.assertEqual(str(auto), "cuda")

    def test_missing_checkpoint_fails_fast(self):
        """Confirms missing checkpoint raises CheckpointNotFoundError immediately."""
        fake_path = os.path.join(self.eval_dir, "nonexistent_weights_123.pth")
        with self.assertRaises(CheckpointNotFoundError):
            predict_eye_disease(self.sample_image, weights_path=fake_path)


if __name__ == "__main__":
    unittest.main()
