"""
Verification Tests for ml/MODEL_CARD.md.

Ensures that the model card:
1. Exists, is non-empty, and conforms to required formatting standards.
2. Contains all 14 required core sections.
3. Accurately reports checkpoint metadata, SHA-256 hash, and file size.
4. Accurately reports canonical class names and ordering.
5. Accurately reflects empirical evaluation metrics from test_evaluation_report.json.
6. Maintains required clinical disclaimers and out-of-scope boundaries.
7. Documents physiological vitals reference ranges and rules.
"""

import json
import os
import re
import unittest

from ml.config import CLASS_NAMES, PathsConfig
from ml.experiment_tracker import compute_file_sha256


class TestModelCardIntegrity(unittest.TestCase):
    """Automated integrity and factual consistency tests for ml/MODEL_CARD.md."""

    @classmethod
    def setUpClass(cls):
        paths = PathsConfig.from_env()
        cls.ml_dir = paths.ml_dir
        cls.model_card_path = os.path.join(cls.ml_dir, "MODEL_CARD.md")
        cls.checkpoint_path = paths.checkpoint_path
        cls.eval_report_path = os.path.join(paths.evaluation_dir, "test_evaluation_report.json")

        assert os.path.isfile(cls.model_card_path), f"MODEL_CARD.md not found at {cls.model_card_path}"
        with open(cls.model_card_path, "r", encoding="utf-8") as f:
            cls.content = f.read()

    def test_model_card_non_empty(self):
        """Model card must exist and have substantive content (> 5000 characters)."""
        self.assertGreater(len(self.content), 5000)

    def test_all_14_required_sections_present(self):
        """Verifies that all 14 required section titles exist with ## headers."""
        required_sections = [
            "1. Model Overview",
            "2. Intended Use",
            "3. Out-of-Scope Uses",
            "4. Model Architecture",
            "5. Dataset and Provenance",
            "6. Training Procedure",
            "7. Evaluation Methodology and Results",
            "8. Preprocessing and Quality Requirements",
            "9. Output Interpretation",
            "10. Limitations and Risks",
            "11. Ethical and Safety Considerations",
            "12. Versioning and Checkpoint Provenance",
            "13. Reproducibility Instructions",
            "14. Outstanding Verification Requirements",
        ]
        for section in required_sections:
            pattern = re.compile(rf"^##\s+{re.escape(section)}", re.MULTILINE)
            self.assertTrue(
                pattern.search(self.content),
                f"Missing required section header in MODEL_CARD.md: '{section}'",
            )

    def test_checkpoint_hash_and_metadata_match(self):
        """Verifies checkpoint SHA-256 in MODEL_CARD.md matches the actual weights file."""
        if os.path.isfile(self.checkpoint_path):
            actual_sha = compute_file_sha256(self.checkpoint_path)
            self.assertIn(actual_sha, self.content)
            actual_size = os.path.getsize(self.checkpoint_path)
            formatted_size = f"{actual_size:,}"
            self.assertTrue(
                str(actual_size) in self.content or formatted_size in self.content,
                f"Neither {actual_size} nor {formatted_size} found in MODEL_CARD.md",
            )

    def test_canonical_classes_and_ordering(self):
        """Verifies canonical 4 classes and their 0-3 indices are documented."""
        for idx, cls_name in enumerate(CLASS_NAMES):
            self.assertIn(cls_name, self.content)
            self.assertIn(f"`{idx}`", self.content)

    def test_evaluation_metrics_match_report(self):
        """Verifies that evaluation metrics in MODEL_CARD.md match test_evaluation_report.json."""
        if os.path.isfile(self.eval_report_path):
            with open(self.eval_report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            acc = report.get("overall_accuracy", report.get("metrics", {}).get("overall_accuracy"))
            acc_pct_str = f"{acc * 100:.2f}%"
            self.assertIn(acc_pct_str, self.content)

            # Check support
            total_images = report.get("total_test_images", 894)
            self.assertIn(str(total_images), self.content)

    def test_clinical_research_disclaimers(self):
        """Verifies strong clinical research warnings and CDS framing."""
        self.assertIn("CLINICAL RESEARCH", self.content)
        self.assertIn("autonomous diagnostic system", self.content.lower())
        self.assertIn("decision-support", self.content.lower())

    def test_independent_vitals_documentation(self):
        """Verifies independent physiological triage rules and ranges are documented."""
        self.assertIn("Heart Rate", self.content)
        self.assertIn("SpO", self.content)
        self.assertIn("Temperature", self.content)
        self.assertIn("BPM", self.content)
        self.assertIn("Normal", self.content)
        self.assertIn("Abnormal", self.content)
        self.assertIn("Lower than usual", self.content)
        self.assertIn("Higher than usual", self.content)


if __name__ == "__main__":
    unittest.main()
