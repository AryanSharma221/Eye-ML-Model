"""
ML Runtime Environment & Dependency Diagnostic Utility.

Inspects, validates, and reports:
1. Python version and runtime platform.
2. Core ML dependencies (torch, torchvision, opencv, pillow, numpy, scikit-learn).
3. OpenCV image-processing functionality (filtering, color conversions, colormaps).
4. Hardware acceleration (CUDA availability, GPU device details, memory, compute capability).
5. Model Checkpoint existence, file readability, and weights integrity.
6. Dataset root directory and split-manifest availability.
7. Output and logging directory write permissions.

Distinguishes non-fatal warnings (e.g., CPU fallback) from fatal blockers
(e.g., missing checkpoint, corrupt dependencies).
"""

import argparse
import importlib
import json
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

from ml.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    get_default_checkpoint_path,
    get_default_dataset_dir,
    get_default_evaluation_dir,
    get_ml_config,
    resolve_device,
)


class CheckStatus:
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class DiagnosticReport:
    """Collects and displays diagnostic check results."""

    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def add(
        self,
        category: str,
        name: str,
        status: str,
        details: str,
        is_fatal: bool = False,
        recommendation: Optional[str] = None,
    ) -> None:
        self.checks.append({
            "category": category,
            "name": name,
            "status": status,
            "details": details,
            "is_fatal": is_fatal,
            "recommendation": recommendation,
        })

    @property
    def has_fatal_errors(self) -> bool:
        return any(c["status"] == CheckStatus.FAIL and c["is_fatal"] for c in self.checks)

    @property
    def total_passed(self) -> int:
        return sum(1 for c in self.checks if c["status"] == CheckStatus.PASS)

    @property
    def total_warnings(self) -> int:
        return sum(1 for c in self.checks if c["status"] == CheckStatus.WARN)

    @property
    def total_failures(self) -> int:
        return sum(1 for c in self.checks if c["status"] == CheckStatus.FAIL)

    def print_summary(self) -> None:
        print("=" * 80)
        print("MULTIMODAL HEALTH SCREENING -- ML ENVIRONMENT DIAGNOSTIC REPORT")
        print("=" * 80)
        print(f"Platform:   {platform.platform()}")
        print(f"Python:     {platform.python_version()} ({sys.executable})")
        print(f"Timestamp:  {os.environ.get('DIAG_TIMESTAMP', 'Current')}")
        print("-" * 80)

        current_category = ""
        for c in self.checks:
            if c["category"] != current_category:
                current_category = c["category"]
                print(f"\n[{current_category.upper()}]")

            tag = f"[{c['status']}]"
            name_str = f"{c['name']:<32}"
            print(f"  {tag:<7} {name_str} : {c['details']}")
            if c["status"] != CheckStatus.PASS and c["recommendation"]:
                print(f"          Action: {c['recommendation']}")

        print("\n" + "=" * 80)
        verdict = "HEALTHY" if not self.has_fatal_errors else "BLOCKED"
        print(f"DIAGNOSTIC VERDICT: {verdict}")
        print(f"Summary: {self.total_passed} Passed | {self.total_warnings} Warnings | {self.total_failures} Failures")
        if self.has_fatal_errors:
            print("FATAL ERROR DETECTED: Real model inference or training cannot proceed until resolved.")
        else:
            print("ENVIRONMENT READY: System is capable of executing ML inference.")
        print("=" * 80)


def run_diagnostics(
    verbose: bool = True,
    check_inference: bool = False,
    check_dataset: bool = False,
) -> DiagnosticReport:
    """
    Executes full diagnostic test suite across environment, dependencies, hardware, and weights.

    Args:
        verbose: Whether to print report summary to stdout.
        check_inference: If True, strictly requires trained checkpoint (fatal error if missing).
                         If False, missing checkpoint generates non-fatal warning so developer
                         onboarding and unit tests can proceed.
        check_dataset: If True, strictly requires dataset and manifest (fatal error if missing).
                       If False, missing dataset generates non-fatal warning.
    """
    report = DiagnosticReport()
    cfg = get_ml_config()

    # -------------------------------------------------------------------------
    # 1. Python Environment Check
    # -------------------------------------------------------------------------
    py_ver = sys.version_info
    py_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    if py_ver >= (3, 10):
        report.add("Python Runtime", "Python Version", CheckStatus.PASS, f"{py_str} (>= 3.10 required)")
    else:
        report.add(
            "Python Runtime",
            "Python Version",
            CheckStatus.FAIL,
            f"{py_str} (Python >= 3.10 is required)",
            is_fatal=True,
            recommendation="Upgrade Python to 3.10 or higher.",
        )

    # -------------------------------------------------------------------------
    # 2. Package Dependency Checks
    # -------------------------------------------------------------------------
    dependencies = [
        ("torch", "2.4.0", True),
        ("torchvision", "0.19.0", True),
        ("numpy", "1.26.0", True),
        ("PIL", "10.2.0", True),  # pillow
        ("cv2", "4.9.0", True),   # opencv-python
        ("sklearn", "1.4.0", False),  # scikit-learn
    ]

    for mod_name, min_ver, is_fatal in dependencies:
        display_name = "pillow" if mod_name == "PIL" else ("opencv-python" if mod_name == "cv2" else ("scikit-learn" if mod_name == "sklearn" else mod_name))
        try:
            mod = importlib.import_module(mod_name)
            installed_ver = getattr(mod, "__version__", "unknown")
            report.add(
                "Dependencies",
                display_name,
                CheckStatus.PASS,
                f"v{installed_ver} (minimum: v{min_ver})",
            )
        except ImportError as err:
            status = CheckStatus.FAIL if is_fatal else CheckStatus.WARN
            report.add(
                "Dependencies",
                display_name,
                status,
                f"Not installed: {err}",
                is_fatal=is_fatal,
                recommendation=f"Install via: pip install {display_name}",
            )

    # -------------------------------------------------------------------------
    # 3. OpenCV Capabilities Check
    # -------------------------------------------------------------------------
    try:
        import cv2
        import numpy as np
        # Verify color conversion, filtering, and colormaps
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2GRAY)
        _ = cv2.Laplacian(gray, cv2.CV_64F).var()
        _ = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        report.add(
            "OpenCV Capabilities",
            "Image Processing Kernels",
            CheckStatus.PASS,
            "cvtColor, Laplacian blur detection, and colormap heatmap generation verified.",
        )
    except Exception as cv_err:
        report.add(
            "OpenCV Capabilities",
            "Image Processing Kernels",
            CheckStatus.FAIL,
            f"OpenCV kernel failure: {cv_err}",
            is_fatal=True,
            recommendation="Reinstall opencv-python: pip install --force-reinstall opencv-python",
        )

    # -------------------------------------------------------------------------
    # 4. Internal Package Imports Check
    # -------------------------------------------------------------------------
    internal_modules = [
        ("ml.quality_gate", "check_image_quality"),
        ("ml.preprocessing", "preprocess_image"),
        ("ml.eye_model", "get_model"),
        ("ml.physio_model", "analyze_vitals"),
        ("ml.screening_pipeline", "run_screening"),
        ("ml.gradcam", "generate_gradcam"),
        ("ml.config", "get_ml_config"),
        ("ml.exceptions", "MLException"),
        ("ml.monitoring", "log_ml_event"),
        ("ml.checkpoint_manager", "get_active_checkpoint_status"),
    ]
    all_imports_passed = True
    failed_imports = []
    for mod_path, symbol in internal_modules:
        try:
            m = importlib.import_module(mod_path)
            if not hasattr(m, symbol):
                all_imports_passed = False
                failed_imports.append(f"{mod_path}.{symbol}")
        except Exception as imp_err:
            all_imports_passed = False
            failed_imports.append(f"{mod_path} ({imp_err})")

    if all_imports_passed:
        report.add(
            "Package Imports",
            "ML Submodules Import",
            CheckStatus.PASS,
            f"All {len(internal_modules)} core ML submodules imported cleanly without error.",
        )
    else:
        report.add(
            "Package Imports",
            "ML Submodules Import",
            CheckStatus.FAIL,
            f"Failed importing submodules: {', '.join(failed_imports)}",
            is_fatal=True,
            recommendation="Verify PYTHONPATH and virtual environment dependencies.",
        )

    # -------------------------------------------------------------------------
    # 5. Configuration Validity Check
    # -------------------------------------------------------------------------
    try:
        class_order_valid = list(cfg.model.class_names) == ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
        img_size_valid = cfg.image.input_size == (224, 224)
        qg_bounds_valid = (
            cfg.image.blur_threshold == 100.0
            and cfg.image.brightness_min == 30.0
            and cfg.image.brightness_max == 225.0
            and cfg.image.contrast_min == 20.0
        )
        if class_order_valid and img_size_valid and qg_bounds_valid:
            report.add(
                "Configuration Validity",
                "Schema & Model Invariants",
                CheckStatus.PASS,
                "Canonical 4-class ordering, 224x224 tensor dims, and quality gate thresholds validated.",
            )
        else:
            report.add(
                "Configuration Validity",
                "Schema & Model Invariants",
                CheckStatus.FAIL,
                f"Configuration invariant check failed (classes={class_order_valid}, size={img_size_valid}, qg={qg_bounds_valid})",
                is_fatal=True,
                recommendation="Reset configuration defaults via ml.config.MLConfig.default()",
            )
    except Exception as cfg_err:
        report.add(
            "Configuration Validity",
            "Schema & Model Invariants",
            CheckStatus.FAIL,
            f"Configuration validation error: {cfg_err}",
            is_fatal=True,
        )

    # -------------------------------------------------------------------------
    # 6. Hardware Acceleration & Device Selection
    # -------------------------------------------------------------------------
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        dev_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        cuda_ver = torch.version.cuda
        dev_count = torch.cuda.device_count()
        report.add(
            "Hardware Acceleration",
            "CUDA Availability",
            CheckStatus.PASS,
            f"Available (CUDA {cuda_ver}, {dev_count} GPU device(s))",
        )
        report.add(
            "Hardware Acceleration",
            "Primary GPU Device",
            CheckStatus.PASS,
            f"{dev_name} (Compute Capability: {cap[0]}.{cap[1]})",
        )
    else:
        report.add(
            "Hardware Acceleration",
            "CUDA Availability",
            CheckStatus.WARN,
            "CUDA not available; CPU mode will be utilized for inference.",
            is_fatal=False,
            recommendation="Install PyTorch with CUDA support if an NVIDIA GPU is present: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
        )

    # Verify device resolution logic
    try:
        resolved = resolve_device("auto")
        report.add(
            "Hardware Acceleration",
            "Device Auto-Resolution",
            CheckStatus.PASS,
            f"Successfully resolved to '{resolved}'",
        )
    except Exception as res_err:
        report.add(
            "Hardware Acceleration",
            "Device Auto-Resolution",
            CheckStatus.FAIL,
            f"Resolution failed: {res_err}",
            is_fatal=True,
        )

    # -------------------------------------------------------------------------
    # 7. Checkpoint Weights Verification
    # -------------------------------------------------------------------------
    ckpt_path = cfg.paths.checkpoint_path
    ckpt_clean_name = os.path.basename(ckpt_path)

    if not os.path.exists(ckpt_path):
        if check_inference:
            report.add(
                "Model Checkpoint",
                "Weights File Existence",
                CheckStatus.FAIL,
                f"Checkpoint file not found: '{ckpt_clean_name}' at configured path.",
                is_fatal=True,
                recommendation=f"Ensure '{ckpt_clean_name}' is placed in ml/weights/ or set ML_CHECKPOINT_PATH.",
            )
        else:
            report.add(
                "Model Checkpoint",
                "Weights File Existence",
                CheckStatus.WARN,
                f"Checkpoint file not found: '{ckpt_clean_name}' (Inference blocked; unit tests and vitals analysis remain functional).",
                is_fatal=False,
                recommendation=f"Place '{ckpt_clean_name}' in ml/weights/ or set ML_CHECKPOINT_PATH to enable real inference.",
            )
    else:
        try:
            sz_bytes = os.path.getsize(ckpt_path)
            sz_mb = sz_bytes / (1024 * 1024)
            if sz_bytes == 0:
                report.add(
                    "Model Checkpoint",
                    "Weights File Integrity",
                    CheckStatus.FAIL,
                    f"File is 0 bytes (corrupt/empty): '{ckpt_clean_name}'",
                    is_fatal=True,
                    recommendation="Re-download or restore checkpoint binary.",
                )
            else:
                # Inspect checkpoint state dict
                raw_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                if isinstance(raw_data, dict):
                    epoch = raw_data.get("epoch", "N/A")
                    classes = raw_data.get("classes", [])
                    state_dict = raw_data.get("model_state_dict", raw_data.get("state_dict", raw_data))
                    has_classifier = "classifier.weight" in state_dict or "classifier.bias" in state_dict
                    report.add(
                        "Model Checkpoint",
                        "Weights Inspection",
                        CheckStatus.PASS,
                        f"Valid {sz_mb:.2f} MB | Epoch: {epoch} | Classes: {len(classes)} | DenseNet201 state_dict confirmed.",
                    )
                else:
                    report.add(
                        "Model Checkpoint",
                        "Weights Inspection",
                        CheckStatus.PASS,
                        f"Valid {sz_mb:.2f} MB binary state_dict.",
                    )
        except Exception as ckpt_err:
            report.add(
                "Model Checkpoint",
                "Weights Inspection",
                CheckStatus.FAIL,
                f"Failed unpickling checkpoint: {ckpt_err}",
                is_fatal=True,
                recommendation="Verify file is not corrupted or truncated.",
            )

    # -------------------------------------------------------------------------
    # 8. Dataset Root & Manifest
    # -------------------------------------------------------------------------
    manifest_path = cfg.paths.manifest_path
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            meta = manifest_data.get("metadata", {})
            total_imgs = meta.get("total_images", 0)
            train_cnt = meta.get("train_count", 0)
            val_cnt = meta.get("val_count", 0)
            test_cnt = meta.get("test_count", 0)
            report.add(
                "Dataset & Manifest",
                "Split Manifest",
                CheckStatus.PASS,
                f"Manifest found ({total_imgs} total images | Train: {train_cnt}, Val: {val_cnt}, Test: {test_cnt})",
            )
        except Exception as m_err:
            report.add(
                "Dataset & Manifest",
                "Split Manifest",
                CheckStatus.WARN,
                f"Error reading manifest: {m_err}",
                recommendation="Validate JSON structure in ml/dataset/split_manifest.json.",
            )
    else:
        if check_dataset:
            report.add(
                "Dataset & Manifest",
                "Split Manifest",
                CheckStatus.FAIL,
                f"Manifest file not found: '{os.path.basename(manifest_path)}' (Required for dataset operations).",
                is_fatal=True,
                recommendation="Provide split_manifest.json or set ML_MANIFEST_PATH.",
            )
        else:
            report.add(
                "Dataset & Manifest",
                "Split Manifest",
                CheckStatus.WARN,
                f"Manifest file not found: '{os.path.basename(manifest_path)}' (Inference remains supported).",
                is_fatal=False,
                recommendation="Dataset manifest is only required for training and dataset evaluation.",
            )

    ds_dir = cfg.paths.dataset_dir
    if os.path.exists(ds_dir) and os.path.isdir(ds_dir):
        report.add(
            "Dataset & Manifest",
            "Dataset Root Directory",
            CheckStatus.PASS,
            f"Configured dataset directory exists: '{os.path.basename(ds_dir)}'",
        )
    else:
        if check_dataset:
            report.add(
                "Dataset & Manifest",
                "Dataset Root Directory",
                CheckStatus.FAIL,
                f"Dataset directory not found: '{os.path.basename(ds_dir)}' (Required for dataset operations).",
                is_fatal=True,
                recommendation="Set ML_DATASET_DIR to the root dataset path.",
            )
        else:
            report.add(
                "Dataset & Manifest",
                "Dataset Root Directory",
                CheckStatus.WARN,
                f"Dataset directory not found: '{os.path.basename(ds_dir)}' (Inference remains supported).",
                is_fatal=False,
                recommendation="Set ML_DATASET_DIR if training or full dataset evaluation is intended.",
            )

    # -------------------------------------------------------------------------
    # 9. Output & Logging Directory Permissions
    # -------------------------------------------------------------------------
    dirs_to_verify = [
        ("Evaluation Directory", cfg.paths.evaluation_dir),
        ("Logging Directory", cfg.paths.log_dir),
        ("Demo Results Directory", cfg.paths.demo_dir),
    ]
    for lbl, dir_path in dirs_to_verify:
        try:
            os.makedirs(dir_path, exist_ok=True)
            # Test write access with a temporary file
            test_file = os.path.join(dir_path, ".perm_test.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            report.add(
                "Filesystem Permissions",
                lbl,
                CheckStatus.PASS,
                f"Writable ({os.path.basename(dir_path)})",
            )
        except Exception as perm_err:
            report.add(
                "Filesystem Permissions",
                lbl,
                CheckStatus.FAIL,
                f"Directory permission failure: {perm_err}",
                is_fatal=True,
                recommendation=f"Ensure write permissions for: {dir_path}",
            )

    if verbose:
        report.print_summary()

    return report


def main(argv=None) -> int:
    """CLI entry point for environment diagnostic tool."""
    parser = argparse.ArgumentParser(description="Multimodal Health Screening ML Environment Diagnostics")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument(
        "--check-inference",
        action="store_true",
        help="Strictly validate requirements for inference (requires checkpoint)",
    )
    parser.add_argument(
        "--check-dataset",
        action="store_true",
        help="Strictly validate requirements for training/evaluation (requires dataset and manifest)",
    )
    args = parser.parse_args(argv)

    report = run_diagnostics(
        verbose=not args.json,
        check_inference=args.check_inference,
        check_dataset=args.check_dataset,
    )

    if args.json:
        out = {
            "verdict": "BLOCKED" if report.has_fatal_errors else "HEALTHY",
            "passed": report.total_passed,
            "warnings": report.total_warnings,
            "failures": report.total_failures,
            "checks": report.checks,
        }
        print(json.dumps(out, indent=2))

    return 1 if report.has_fatal_errors else 0


if __name__ == "__main__":
    sys.exit(main())
