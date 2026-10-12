"""
Automated Tests for the Multimodal Health Screening Demonstration Runner (ml/demo.py).

Verifies:
1. CLI help and argument parsing.
2. Pipeline validation mode (--mode validate).
3. Real inference mode with sample discovery (--mode real --sample).
4. Grad-CAM PNG file generation and disk persistence.
5. Error handling when checkpoint or image is unavailable.
6. Safe sample workflow and schema validation.
"""

import json
import os
import subprocess
import sys
import pytest

from ml.config import CLASS_NAMES, ML_DIR
from ml.demo import (
    find_sample_test_image,
    parse_cli_args,
    run_batch_demonstration,
    run_pipeline_validation,
    run_real_screening_demo,
)
from ml.exceptions import CheckpointNotFoundError


class TestDemoCLIAndValidation:
    """Verifies CLI argument handling and validation mode."""

    def test_parse_cli_defaults(self):
        args = parse_cli_args([])
        assert args.mode == "real"
        assert args.image is None
        assert args.sample is False
        assert args.sample_class == "normal"
        assert args.no_gradcam is False

    def test_parse_cli_custom_flags(self):
        args = parse_cli_args([
            "--mode", "validate",
            "--hr", "80",
            "--spo2", "99",
            "--temp", "37.1",
            "--sample",
            "--sample-class", "cataract",
            "--no-gradcam",
        ])
        assert args.mode == "validate"
        assert args.heart_rate == 80.0
        assert args.spo2 == 99.0
        assert args.temperature == 37.1
        assert args.sample is True
        assert args.sample_class == "cataract"
        assert args.no_gradcam is True

    def test_run_pipeline_validation_succeeds(self, tmp_path):
        out_dir = str(tmp_path / "val_results")
        res = run_pipeline_validation(output_dir=out_dir)

        assert res["mode"] == "pipeline_validation"
        assert res["overall_status"] == "PASS"
        assert "subsystems" in res
        assert res["subsystems"]["module_imports"]["status"] == "PASS"
        assert res["subsystems"]["quality_gate"]["status"] == "PASS"
        assert res["subsystems"]["vitals_engine"]["status"] == "PASS"
        assert res["subsystems"]["schema_contract"]["status"] == "PASS"

        # Verify output artifact created
        artifact_path = os.path.join(out_dir, "validation_result.json")
        assert os.path.isfile(artifact_path)
        with open(artifact_path, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        assert persisted["overall_status"] == "PASS"


class TestSampleWorkflowAndRealInference:
    """Verifies the safe sample discovery and real screening execution."""

    def test_find_sample_test_image(self):
        sample = find_sample_test_image(class_name="normal")
        assert sample is not None
        assert os.path.isfile(sample["filepath"])
        assert sample["class_name"] == "normal"

    def test_real_demo_missing_checkpoint_raises(self, tmp_path):
        fake_ckpt = str(tmp_path / "missing_model.pth")
        with pytest.raises(CheckpointNotFoundError):
            run_real_screening_demo(
                weights_path=fake_ckpt,
                output_dir=str(tmp_path),
            )

    def test_real_demo_with_sample_image(self, tmp_path):
        out_dir = str(tmp_path / "real_demo_out")
        res = run_real_screening_demo(
            output_dir=out_dir,
            sample_class="normal",
            heart_rate=72.0,
            spo2=98.0,
            temperature=36.8,
            include_gradcam=True,
        )

        assert res["status"] in ("success", "partial")
        assert "overall_risk_level" in res
        assert "eye_screening" in res
        assert "vitals_screening" in res

        eye = res["eye_screening"]
        assert eye["status"] == "success"
        assert eye["prediction"] in CLASS_NAMES
        assert 0.0 <= eye["confidence"] <= 1.0

        vitals = res["vitals_screening"]
        assert vitals["status"] == "success"
        assert vitals["overall_status"] == "Normal"

        # Verify artifacts written
        json_file = os.path.join(out_dir, "demo_result.json")
        assert os.path.isfile(json_file)

        # Grad-CAM saved PNG check
        gradcam_saved = eye.get("gradcam_saved_file")
        if gradcam_saved:
            full_cam_path = os.path.join(out_dir, gradcam_saved)
            assert os.path.isfile(full_cam_path)
            assert os.path.getsize(full_cam_path) > 1000


class TestCLIExecutionProcess:
    """Verifies demo.py CLI invocations via subprocess."""

    def test_cli_help(self):
        cmd = [sys.executable, "-m", "ml.demo", "--help"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "Multimodal Health Screening" in proc.stdout
        assert "--mode" in proc.stdout

    def test_cli_validate_mode(self, tmp_path):
        out_dir = str(tmp_path / "cli_val")
        cmd = [
            sys.executable, "-m", "ml.demo",
            "--mode", "validate",
            "--output-dir", out_dir,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "PIPELINE VALIDATION MODE" in proc.stdout
        assert os.path.isfile(os.path.join(out_dir, "validation_result.json"))

    def test_cli_real_sample_normal(self, tmp_path):
        out_dir = str(tmp_path / "cli_real")
        cmd = [
            sys.executable, "-m", "ml.demo",
            "--mode", "real",
            "--sample",
            "--sample-class", "normal",
            "--output-dir", out_dir,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "REAL INFERENCE MODE" in proc.stdout
        assert "RETINAL FUNDUS SCREENING" in proc.stdout
        assert os.path.isfile(os.path.join(out_dir, "demo_result.json"))
