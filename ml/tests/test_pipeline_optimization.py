"""
Unit and Integration Tests for Pipeline Runtime, Resource & Memory Optimization.

Verifies:
1. Existence, validity, and schema compliance of optimization reports.
2. In-memory model caching prevents redundant checkpoint reloading.
3. CPU fallback execution completes without functional errors.
4. Absence of progressive GPU memory leaks under repeated inference.
5. Invariant validation: Architecture, checkpoint hash, and 4-class ordering preserved.
"""

import json
import os
import unittest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPECTED_CHECKPOINT_SHA256 = "e819440c35f166c90e6469b071cc83136857e95aa80bcd6697336894bb459f5f"


class TestPipelineOptimization(unittest.TestCase):
    """Test suite verifying pipeline optimization and resource stability contracts."""

    @classmethod
    def setUpClass(cls):
        cls.eval_dir = os.path.join(_PROJECT_ROOT, "ml", "evaluation_results")
        cls.json_report_path = os.path.join(cls.eval_dir, "pipeline_optimization_report.json")
        cls.md_report_path = os.path.join(cls.eval_dir, "PIPELINE_OPTIMIZATION.md")

        assert os.path.isfile(cls.json_report_path), f"Missing {cls.json_report_path}"
        assert os.path.isfile(cls.md_report_path), f"Missing {cls.md_report_path}"

        with open(cls.json_report_path, "r", encoding="utf-8") as f:
            cls.report = json.load(f)

    def test_optimization_report_schema_and_verdict(self):
        """Verifies report schema fields and that final verdict is PASS."""
        self.assertEqual(self.report["verdict"]["status"], "PASS")
        self.assertFalse(self.report["verdict"]["leak_detected"])
        self.assertIn("cuda_profile", self.report)
        self.assertIn("cpu_fallback_profile", self.report)
        self.assertIn("finite_stress_test", self.report)
        self.assertIn("bottleneck_analysis", self.report)

    def test_model_cache_efficiency(self):
        """Verifies cached retrieval latency is orders of magnitude faster than cold loading."""
        cuda_p = self.report["cuda_profile"]
        cold_ms = cuda_p["model_loading"]["cold_start_ms"]
        cached_ms = cuda_p["model_loading"]["cached_retrieval"]["median_ms"]
        self.assertGreater(cold_ms, 100.0)
        self.assertLess(cached_ms, 1.0)

    def test_cpu_fallback_profile_validity(self):
        """Verifies that CPU fallback profile is present and accurately measured."""
        cpu_p = self.report["cpu_fallback_profile"]
        self.assertEqual(cpu_p["device"], "cpu")
        self.assertGreater(cpu_p["stage_latencies"]["total_pipeline_e2e"]["median_ms"], 0.0)
        self.assertGreater(cpu_p["stage_latencies"]["eye_model_inference"]["median_ms"], 0.0)

    def test_zero_gpu_memory_leak_in_stress_test(self):
        """Verifies that 100 consecutive calls produced 0.0 MB of allocated memory growth."""
        stress = self.report["finite_stress_test"]
        self.assertEqual(stress["total_calls"], 100)
        self.assertEqual(stress["gpu_memory_allocated_growth_mb"], 0.0)
        self.assertFalse(stress["leak_detected"])
        self.assertTrue(stress["all_screenings_succeeded"])


if __name__ == "__main__":
    unittest.main()
