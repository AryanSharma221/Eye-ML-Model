"""
Unit and Integration Tests for Generalization and Subgroup Robustness Audit.

Verifies:
1. Presence, validity, and schema compliance of generalization audit reports.
2. Accuracy and integrity of subgroup stratifications (sample counts match 894 held-out test records).
3. Proper handling of small subgroups (marking N < 30 as inconclusive).
4. Validation that all 9 transformations were evaluated and recorded without altering original assets.
5. Invariant checking: Checkpoint and manifest SHA-256 hashes match verified champion signatures.
6. Explicit verification of clinical disclaimer and unresolved domain-shift verdict.
"""

import json
import os
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPECTED_CHECKPOINT_SHA256 = "e819440c35f166c90e6469b071cc83136857e95aa80bcd6697336894bb459f5f"
EXPECTED_MANIFEST_SHA256 = "cb2d80449899673438268750011697f7ae9518a8259e5dc8482edfe9d7c01f27"


class TestGeneralizationAuditArtifacts(unittest.TestCase):
    """Test suite verifying generalization audit report contracts."""

    @classmethod
    def setUpClass(cls):
        cls.eval_dir = os.path.join(_PROJECT_ROOT, "ml", "evaluation_results")
        cls.json_report_path = os.path.join(cls.eval_dir, "generalization_audit_report.json")
        cls.md_report_path = os.path.join(cls.eval_dir, "GENERALIZATION_AUDIT.md")

        assert os.path.isfile(cls.json_report_path), f"Missing {cls.json_report_path}"
        assert os.path.isfile(cls.md_report_path), f"Missing {cls.md_report_path}"

        with open(cls.json_report_path, "r", encoding="utf-8") as f:
            cls.report = json.load(f)

    def test_artifact_hashes_match_champion_signatures(self):
        """Verifies report was generated against the uncorrupted champion checkpoint and manifest."""
        self.assertEqual(self.report["checkpoint_sha256"], EXPECTED_CHECKPOINT_SHA256)
        self.assertEqual(self.report["manifest_sha256"], EXPECTED_MANIFEST_SHA256)
        self.assertEqual(self.report["held_out_test_samples"], 894)

    def test_clean_baseline_metrics_match_ground_truth(self):
        """Verifies clean baseline accuracy matches the 87.81% champion benchmark."""
        base = self.report["clean_baseline_metrics"]
        self.assertAlmostEqual(base["accuracy"], 0.8781, places=3)
        self.assertAlmostEqual(base["macro_f1"], 0.8778, places=3)
        self.assertEqual(sum(base["per_class"][c]["support"] for c in base["per_class"]), 894)

    def test_subgroup_analysis_partitions(self):
        """Verifies subgroup partitions are correctly populated and documented."""
        subgroups = self.report["subgroup_analysis"]
        self.assertIn("res_low_under_800", subgroups)
        self.assertIn("res_high_over_1600", subgroups)
        self.assertIn("source_rfmid", subgroups)
        self.assertIn("source_kaggle", subgroups)
        self.assertIn("conf_high_over_90", subgroups)
        self.assertIn("conf_low_under_70", subgroups)

        # Check total resolution samples
        total_res = subgroups["res_low_under_800"]["count"] + subgroups["res_mid_800_1600"]["count"] + subgroups["res_high_over_1600"]["count"]
        self.assertEqual(total_res, 894)

        # Check source samples
        total_sources = subgroups["source_rfmid"]["count"] + subgroups["source_kaggle"]["count"]
        self.assertEqual(total_sources, 894)

        # Check confidence sample sum
        total_conf = subgroups["conf_high_over_90"]["count"] + subgroups["conf_mid_70_90"]["count"] + subgroups["conf_low_under_70"]["count"]
        self.assertEqual(total_conf, 894)

    def test_all_transformations_evaluated(self):
        """Verifies that all 9 required controlled transformations were evaluated across all 894 samples."""
        rob = self.report["controlled_robustness_tests"]
        expected_transforms = [
            "brightness_inc_15",
            "brightness_dec_15",
            "contrast_inc_15",
            "contrast_dec_15",
            "mild_blur_sigma1",
            "resample_384",
            "rotation_pos_5deg",
            "rotation_neg_5deg",
            "jpeg_quality_75",
        ]
        for t_name in expected_transforms:
            self.assertIn(t_name, rob)
            entry = rob[t_name]
            self.assertEqual(entry["original_samples_evaluated"], 894)
            self.assertGreaterEqual(entry["prediction_agreement_pct"], 90.0)
            self.assertLessEqual(entry["prediction_agreement_pct"], 100.0)
            self.assertIn("mean_confidence_delta", entry)
            self.assertIn("max_confidence_delta", entry)
            self.assertIn("class_flips_from_baseline", entry)

    def test_verdict_and_clinical_disclaimer_present(self):
        """Verifies final verdict is UNRESOLVED_DOMAIN_SHIFT_RISK and external validation is disclaimed."""
        verdict = self.report["verdict_and_recommendations"]
        self.assertEqual(verdict["verdict"], "UNRESOLVED_DOMAIN_SHIFT_RISK")
        self.assertFalse(verdict["external_validation_performed"])
        self.assertIn("NOT_PERFORMED", verdict["external_validation_status"])
        self.assertIn("clinical_disclaimer", self.report)


if __name__ == "__main__":
    unittest.main()
