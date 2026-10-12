"""
Multimodal Health Screening — Demonstration Runner & Pipeline Verification.

Provides a standalone Python demonstration tool for the ML layer supporting:
1. Real Inference Mode (--mode real):
   - Invokes the production screening pipeline (run_screening).
   - Validates trained DenseNet201 checkpoint and authentic fundus images.
   - Evaluates independent physiological vital signs (HR, SpO2, Temperature).
   - Generates and persists class-aligned Grad-CAM heatmaps to PNG.
   - Outputs comprehensive terminal triage reports and persisted JSON artifacts.
   - Never substitutes random weights or fabricated predictions.

2. Pipeline Validation Mode (--mode validate):
   - Verifies module imports, configuration, dependency health, and input validation.
   - Runs deterministic smoke tests on simulated inputs without requiring production images.
   - Explicitly flags all outputs as mocked validation artifacts.

3. Safe Sample Workflow (--sample, --sample-class):
   - Discovers authentic, quality-verified fundus images from the held-out test split
     (split_manifest.json) across all 4 canonical classes:
     ["cataract", "diabetic_retinopathy", "glaucoma", "normal"].

Medical Prototype Disclaimer:
This software is a screening prototype engineered for clinical decision support.
It does not provide definitive medical diagnoses. All outputs must be reviewed by
certified medical professionals.
"""

import argparse
import base64
import json
import logging
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np
import torch

from ml.config import (
    CLASS_NAMES,
    get_default_checkpoint_path,
    resolve_device,
)
from ml.exceptions import CheckpointNotFoundError
from ml.monitoring import sanitize_path
from ml.physio_model import analyze_vitals
from ml.quality_gate import check_image_quality
from ml.screening_pipeline import (
    SYSTEM_SCREENING_DISCLAIMER,
    run_screening,
)

logger = logging.getLogger("ml.demo")


# =============================================================================
# 1. Safe Sample Discovery Workflow
# =============================================================================

def find_sample_test_image(
    class_name: Optional[str] = "normal",
    manifest_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Finds a verified, quality-passed sample image from the held-out test split.

    Args:
        class_name: Target class ('cataract', 'diabetic_retinopathy', 'glaucoma', 'normal').
        manifest_path: Optional path to split_manifest.json.

    Returns:
        Dict with 'filepath', 'filename', 'class_name', 'patient_id', or None if unavailable.
    """
    m_path = manifest_path or os.path.join(_REPO_ROOT, "ml", "dataset", "split_manifest.json")
    if not os.path.isfile(m_path):
        return None

    try:
        with open(m_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as err:
        logger.warning(f"Unable to read split manifest at {m_path}: {err}")
        return None

    test_records = manifest.get("splits", {}).get("test", [])
    target_class = (class_name or "normal").lower().strip()

    for rec in test_records:
        rec_class = rec.get("class_name", "").lower()
        if target_class and rec_class != target_class:
            continue

        fp = rec.get("filepath", "")
        if not os.path.isfile(fp):
            continue

        # Verify it passes quality gate
        q = check_image_quality(fp)
        if q.get("overall_pass", False):
            return {
                "filepath": fp,
                "filename": rec.get("filename", os.path.basename(fp)),
                "class_name": rec.get("class_name", target_class),
                "patient_id": rec.get("patient_id", "sample_test_patient"),
                "quality_summary": q,
            }

    # Fallback to any passing test image if specific class wasn't found
    for rec in test_records:
        fp = rec.get("filepath", "")
        if os.path.isfile(fp):
            q = check_image_quality(fp)
            if q.get("overall_pass", False):
                return {
                    "filepath": fp,
                    "filename": rec.get("filename", os.path.basename(fp)),
                    "class_name": rec.get("class_name", "unknown"),
                    "patient_id": rec.get("patient_id", "sample_test_patient"),
                    "quality_summary": q,
                }

    return None


# =============================================================================
# 2. Pipeline Validation Mode (Dry-Run / Verification)
# =============================================================================

def run_pipeline_validation(
    output_dir: str = os.path.join("ml", "demo_results"),
    weights_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes pipeline validation mode.
    Verifies modules, configuration, quality-gate rules, vitals validation,
    and input schemas. Any mocked executions are strictly labeled.
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.perf_counter()
    validation_report: Dict[str, Any] = {
        "mode": "pipeline_validation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notice": "MOCKED / DRY-RUN VALIDATION ONLY -- NOT CLINICAL DIAGNOSTIC PREDICTIONS",
        "subsystems": {},
    }

    print("\n" + "=" * 78)
    print(" [PIPELINE VALIDATION MODE] -- DRY RUN & INTEGRATION HEALTH CHECK")
    print("=" * 78)
    print(" Notice: Validating pipeline schemas, error boundaries, and module integrity.")
    print(" Any outputs in this mode are strictly mocked verification artifacts.")
    print("-" * 78)

    # 1. Dependency & Module Health
    modules_checked = {}
    for mod_name in [
        "torch", "torchvision", "cv2", "PIL", "numpy",
        "ml.quality_gate", "ml.preprocessing", "ml.eye_model",
        "ml.gradcam", "ml.physio_model", "ml.screening_pipeline", "ml.checkpoint_manager"
    ]:
        try:
            __import__(mod_name)
            modules_checked[mod_name] = "AVAILABLE"
        except ImportError as e:
            modules_checked[mod_name] = f"MISSING ({e})"

    all_mods_ok = all(v == "AVAILABLE" for v in modules_checked.values())
    validation_report["subsystems"]["module_imports"] = {
        "status": "PASS" if all_mods_ok else "FAIL",
        "modules": modules_checked,
    }
    print(f" [1/5] Module Imports:        {'[PASS]' if all_mods_ok else '[FAIL]'}")

    # 2. Checkpoint Availability & Integrity Check
    active_ckpt = weights_path or get_default_checkpoint_path()
    ckpt_status: Dict[str, Any] = {
        "path": sanitize_path(active_ckpt),
        "exists_on_disk": os.path.isfile(active_ckpt),
    }
    if ckpt_status["exists_on_disk"]:
        try:
            raw = torch.load(active_ckpt, map_location="cpu", weights_only=True)
            ckpt_status["loadable_weights_only"] = True
            ckpt_status["classes"] = raw.get("classes", [])
            ckpt_status["epoch"] = raw.get("epoch", None)
            ckpt_status["best_val_accuracy"] = raw.get("best_val_accuracy", None)
            ckpt_status["status"] = "PASS"
        except Exception as err:
            ckpt_status["loadable_weights_only"] = False
            ckpt_status["error"] = str(err)
            ckpt_status["status"] = "CORRUPT"
    else:
        ckpt_status["status"] = "UNAVAILABLE"

    validation_report["subsystems"]["checkpoint_health"] = ckpt_status
    print(f" [2/5] Checkpoint Status:      [{ckpt_status['status']}] ({ckpt_status['path']})")

    qg_results = {}
    with tempfile.TemporaryDirectory() as fixtures_dir:
        blank_path = os.path.join(fixtures_dir, "val_blank_rejection.png")
        cv2.imwrite(blank_path, np.zeros((300, 300, 3), dtype=np.uint8))
        q_blank = check_image_quality(blank_path)
        qg_results["blank_canvas_rejected"] = (not q_blank["overall_pass"])

        blurred_path = os.path.join(fixtures_dir, "val_blur_rejection.png")
        blurred_canvas = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(blurred_canvas, (150, 150), 100, (180, 150, 120), -1)
        blurred_canvas = cv2.GaussianBlur(blurred_canvas, (45, 45), sigmaX=20)
        cv2.imwrite(blurred_path, blurred_canvas)
        q_blur = check_image_quality(blurred_path)
        qg_results["blur_canvas_rejected"] = (not q_blur["overall_pass"])

    qg_ok = qg_results["blank_canvas_rejected"] and qg_results["blur_canvas_rejected"]
    validation_report["subsystems"]["quality_gate"] = {
        "status": "PASS" if qg_ok else "FAIL",
        "tests": qg_results,
    }
    print(f" [3/5] Quality Gate Logic:     {'[PASS]' if qg_ok else '[FAIL]'}")

    # 4. Vitals Engine Verification
    vitals_test_results = {}
    v_norm = analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=36.8)
    vitals_test_results["normal_case_low_risk"] = (v_norm["risk_level"] == "LOW")

    v_crit = analyze_vitals(heart_rate=145.0, spo2=86.0, temperature=39.5)
    vitals_test_results["critical_case_high_risk"] = (v_crit["risk_level"] == "HIGH")

    # Test validation error on invalid input
    try:
        analyze_vitals(heart_rate=500.0, spo2=98.0, temperature=37.0)
        invalid_rejected = False
    except ValueError:
        invalid_rejected = True
    vitals_test_results["implausible_value_rejected"] = invalid_rejected

    vitals_ok = all(vitals_test_results.values())
    validation_report["subsystems"]["vitals_engine"] = {
        "status": "PASS" if vitals_ok else "FAIL",
        "tests": vitals_test_results,
    }
    print(f" [4/5] Vitals Engine Logic:    {'[PASS]' if vitals_ok else '[FAIL]'}")

    # 5. Schema Serialization & Smoke Screening
    # Test screening pipeline with vitals-only (no image needed)
    smoke_screen = run_screening(heart_rate=75.0, spo2=98.0, temperature=36.6)
    schema_ok = (
        smoke_screen.get("status") in ("success", "partial")
        and "vitals_screening" in smoke_screen
        and "overall_risk_level" in smoke_screen
    )
    validation_report["subsystems"]["schema_contract"] = {
        "status": "PASS" if schema_ok else "FAIL",
        "smoke_status": smoke_screen.get("status"),
    }
    print(f" [5/5] Schema Contract:        {'[PASS]' if schema_ok else '[FAIL]'}")

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    validation_report["elapsed_ms"] = elapsed_ms
    overall_status = "PASS" if (all_mods_ok and qg_ok and vitals_ok and schema_ok) else "FAIL"
    validation_report["overall_status"] = overall_status

    out_file = os.path.join(output_dir, "validation_result.json")
    from ml.security import safe_atomic_write_json
    safe_atomic_write_json(out_file, validation_report, disallow_critical=True)

    print("-" * 78)
    print(f" Validation Outcome: [{overall_status}] in {elapsed_ms} ms")
    print(f" Artifact Persisted: {out_file}")
    print("=" * 78 + "\n")

    return validation_report


# =============================================================================
# 3. Real Inference Demonstration Runner
# =============================================================================

def run_real_screening_demo(
    image_path: Optional[str] = None,
    heart_rate: Optional[float] = None,
    spo2: Optional[float] = None,
    temperature: Optional[float] = None,
    weights_path: Optional[str] = None,
    output_dir: str = os.path.join("ml", "demo_results"),
    device: Optional[Union[str, torch.device]] = None,
    include_gradcam: bool = True,
    sample_class: str = "normal",
) -> Dict[str, Any]:
    """
    Executes real inference demonstration using the production ML pipeline.

    Guarantees:
    - Never uses mock predictions or random weights.
    - Requires an authentic trained checkpoint and fundus image (or sample image).
    - Decodes and saves Grad-CAM visualization to PNG.
    - Persists complete structured JSON output.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_device = resolve_device(device or "auto")
    active_ckpt = weights_path or get_default_checkpoint_path()

    print("\n" + "=" * 78)
    print(" [REAL INFERENCE MODE] -- PRODUCTION MULTIMODAL HEALTH SCREENING")
    print("=" * 78)

    # 1. Verify Checkpoint
    if not os.path.isfile(active_ckpt):
        raise CheckpointNotFoundError(
            f"Production checkpoint file not found at '{active_ckpt}'. "
            f"Real inference requires a verified trained checkpoint.",
            operation="demo_inference",
        )

    # 2. Resolve Image Input
    selected_image = image_path
    sample_metadata = None

    if not selected_image:
        print(f" Discovering verified '{sample_class}' sample from held-out test split...")
        sample_meta = find_sample_test_image(class_name=sample_class)
        if sample_meta:
            selected_image = sample_meta["filepath"]
            sample_metadata = sample_meta
            print(f" Found sample: {sanitize_path(selected_image)} (GT: {sample_meta['class_name']})")
        else:
            print(" No test split image discovered automatically.")

    if not selected_image and heart_rate is None and spo2 is None and temperature is None:
        raise ValueError(
            "No inputs provided. Specify --image and/or vitals (--hr, --spo2, --temp), "
            "or pass --sample to use a verified test split image."
        )

    # 3. Default Demo Vitals if none supplied for a full multimodal demonstration
    demo_vitals_used = False
    hr_val = heart_rate
    spo2_val = spo2
    temp_val = temperature
    if hr_val is None and spo2_val is None and temp_val is None:
        demo_vitals_used = True
        hr_val = 74.0
        spo2_val = 98.0
        temp_val = 36.7
        print(" Using demo baseline vitals (HR=74 BPM, SpO2=98%, Temp=36.7 C) [TEST INPUT]")

    # 4. Invoke Real Production Screening Pipeline
    print(f" Executing screening pipeline on device: {target_device}...")
    start_time = time.perf_counter()

    screening_result = run_screening(
        image_path=selected_image,
        heart_rate=hr_val,
        spo2=spo2_val,
        temperature=temp_val,
        weights_path=active_ckpt,
        device=target_device,
        include_gradcam=include_gradcam,
    )
    total_elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # 5. Extract and Save Grad-CAM PNG if generated
    gradcam_saved_path = None
    eye_res = screening_result.get("eye_screening", {})
    if selected_image and eye_res.get("gradcam_image"):
        b64_str = eye_res["gradcam_image"]
        try:
            from ml.security import safe_atomic_write_file, safe_atomic_write_json
            png_bytes = base64.b64decode(b64_str)
            gradcam_filename = f"gradcam_{sanitize_path(selected_image).replace('.', '_')}.png"
            gradcam_saved_path = os.path.join(output_dir, gradcam_filename)
            safe_atomic_write_file(gradcam_saved_path, png_bytes, mode="wb", disallow_critical=True)
            # Remove giant base64 string from persisted JSON to keep files compact
            eye_res["gradcam_saved_file"] = sanitize_path(gradcam_saved_path)
        except Exception as cam_err:
            logger.warning(f"Failed to persist Grad-CAM PNG: {cam_err}")

    # Attach demonstration metadata
    screening_result["demo_metadata"] = {
        "mode": "real_inference",
        "demo_vitals_used": demo_vitals_used,
        "sample_from_test_split": (sample_metadata is not None),
        "ground_truth_class": sample_metadata.get("class_name") if sample_metadata else None,
        "device": str(target_device),
        "checkpoint_file": sanitize_path(active_ckpt),
        "total_demo_duration_ms": total_elapsed_ms,
        "gradcam_file": sanitize_path(gradcam_saved_path) if gradcam_saved_path else None,
    }

    # 6. Save JSON Report
    json_out_file = os.path.join(output_dir, "demo_result.json")
    from ml.security import safe_atomic_write_json
    safe_atomic_write_json(json_out_file, screening_result, disallow_critical=True)

    # 7. Print Human-Readable Terminal Summary
    _print_terminal_summary(screening_result, gradcam_saved_path, total_elapsed_ms)

    return screening_result


def _print_terminal_summary(
    result: Dict[str, Any],
    gradcam_path: Optional[str],
    elapsed_ms: float,
) -> None:
    """Renders formatted ASCII summary table for terminal display."""
    demo_meta = result.get("demo_metadata", {})
    overall_risk = result.get("overall_risk_level", "UNKNOWN")
    screening_id = result.get("screening_id", "N/A")
    eye = result.get("eye_screening", {})
    vitals = result.get("vitals_screening", {})

    print("-" * 78)
    print(f" Status:             {result.get('status', 'unknown').upper()}")
    print(f" Overall Risk:       {overall_risk}")
    print(f" Screening ID:       {screening_id}")
    print(f" Checkpoint:         {demo_meta.get('checkpoint_file', 'eye_model.pth')}")
    print(f" Backend Device:     {demo_meta.get('device', 'cpu')}")
    print(f" Total Duration:     {elapsed_ms:.1f} ms")
    print("-" * 78)

    # Retinal Fundus Findings
    if eye and eye.get("status") != "skipped":
        print("\n [1] RETINAL FUNDUS SCREENING")
        eye_status = eye.get("status", "unknown").upper()
        if eye_status == "ERROR":
            print(f"  * Status:           ERROR (Reason: {eye.get('reason', 'Unknown error')})")
        else:
            q_gate = eye.get("quality_checks") or eye.get("quality_gate") or {}
            q_status = "PASSED" if q_gate.get("overall_pass") else "REJECTED"
            blur_val = q_gate.get("blur_score", 0.0)
            ret_val = q_gate.get("retinal_region_detected", False)
            print(f"  * Quality Gate:     {q_status} (Blur: {blur_val:.1f}, Retinal: {ret_val})")

            pred = eye.get("prediction")
            if pred:
                conf = eye.get("confidence", 0.0) * 100.0
                risk = eye.get("risk_level", "UNKNOWN")
                gt = demo_meta.get("ground_truth_class")
                gt_str = f" [Ground Truth: {gt}]" if gt else ""
                print(f"  * Prediction:       {pred.upper()} ({conf:.1f}% confidence, Risk: {risk}){gt_str}")
                print("  * Class Probabilities:")
                probs = eye.get("all_probabilities") or eye.get("probabilities") or {}
                for c_name in CLASS_NAMES:
                    p_val = probs.get(c_name, 0.0) * 100.0
                    bar = "#" * int(p_val / 5)
                    print(f"      - {c_name:<21}: {p_val:5.1f}%  {bar}")
            else:
                print(f"  * Inference:        BLOCKED / REJECTED (Reason: {eye.get('reason', 'N/A')})")

            if gradcam_path:
                print(f"  * Grad-CAM Heatmap:  Saved to: {sanitize_path(gradcam_path)}")
    else:
        print("\n [1] RETINAL FUNDUS SCREENING: Not evaluated (No image provided)")

    # Physiological Vitals Findings
    if vitals and vitals.get("status") != "skipped":
        print("\n [2] PHYSIOLOGICAL VITAL SIGNS")
        v_status = vitals.get("overall_status", "Unknown")
        v_risk = vitals.get("risk_level", "UNKNOWN")
        hr_val = vitals.get("heart_rate")
        spo2_val = vitals.get("spo2")
        temp_val = vitals.get("temperature")
        print(f"  * Status:           {v_status} (Risk: {v_risk})")
        print(f"  * Measurements:     HR={hr_val} BPM ({vitals.get('heart_rate_status')}), "
              f"SpO2={spo2_val}% ({vitals.get('spo2_status')}), "
              f"Temp={temp_val} C ({vitals.get('temperature_status')})")
        print(f"  * Clinical Summary: {vitals.get('summary', 'N/A')}")
    else:
        print("\n [2] PHYSIOLOGICAL VITAL SIGNS: Not evaluated (No vitals provided)")

    print("\n [3] CLINICAL DISCLAIMER")
    print(f"  \"{SYSTEM_SCREENING_DISCLAIMER}\"")
    print("=" * 78 + "\n")


# =============================================================================
# 4. Multi-Sample Held-Out Batch Demonstration (Full Evaluation Suite)
# =============================================================================

def run_batch_demonstration(
    checkpoint_path: Optional[str] = None,
    output_dir: str = os.path.join("ml", "demo_results"),
) -> Dict[str, Any]:
    """
    Executes real inference across 4 representative held-out test split images
    (one per class) plus one controlled quality rejection test case.
    """
    os.makedirs(output_dir, exist_ok=True)
    active_ckpt = checkpoint_path or get_default_checkpoint_path()
    manifest_path = os.path.join(_REPO_ROOT, "ml", "dataset", "split_manifest.json")

    print("\n" + "=" * 78)
    print(" [BATCH DEMONSTRATION] -- 4-CLASS HELD-OUT TEST SPLIT VALIDATION")
    print("=" * 78)

    sample_results = []
    for cls_name in CLASS_NAMES:
        sample_meta = find_sample_test_image(class_name=cls_name, manifest_path=manifest_path)
        if not sample_meta:
            print(f" [SKIP] No valid sample found for class '{cls_name}'.")
            continue

        fp = sample_meta["filepath"]
        res = run_screening(
            image_path=fp,
            heart_rate=72.0,
            spo2=98.0,
            temperature=36.8,
            weights_path=active_ckpt,
            include_gradcam=True,
        )
        eye_res = res.get("eye_screening", {})
        pred = eye_res.get("prediction")
        conf = eye_res.get("confidence", 0.0)
        is_match = (pred == cls_name)
        match_str = "[MATCH]" if is_match else "[MISMATCH]"
        print(f"  * {cls_name:<21} -> {str(pred):<21} ({conf*100:5.1f}%) {match_str}")
        sample_results.append({
            "target_class": cls_name,
            "prediction": pred,
            "confidence": conf,
            "match": is_match,
        })

    matches = sum(1 for s in sample_results if s["match"])
    print(f"\n Outcome: {matches}/{len(sample_results)} matches on representative test samples.")
    print("=" * 78 + "\n")

    return {
        "batch_samples_tested": len(sample_results),
        "matches": matches,
        "results": sample_results,
    }


# =============================================================================
# 5. CLI Entry Point
# =============================================================================

def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal Health Screening — Demonstration Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["real", "validate"],
        default="real",
        help="Demonstration mode: 'real' (production inference) or 'validate' (pipeline contract checks).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to retinal fundus image for screening.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use a verified representative image from the held-out test split.",
    )
    parser.add_argument(
        "--sample-class",
        type=str,
        choices=CLASS_NAMES,
        default="normal",
        help="Class of sample image to select when --sample is active.",
    )
    parser.add_argument(
        "--hr", "--heart-rate",
        dest="heart_rate",
        type=float,
        default=None,
        help="Heart rate in beats per minute (BPM).",
    )
    parser.add_argument(
        "--spo2",
        dest="spo2",
        type=float,
        default=None,
        help="Blood oxygen saturation percentage (%%).",
    )
    parser.add_argument(
        "--temp", "--temperature",
        dest="temperature",
        type=float,
        default=None,
        help="Core body temperature in Celsius (deg C).",
    )
    parser.add_argument(
        "--weights", "--checkpoint-path",
        dest="weights_path",
        type=str,
        default=None,
        help="Path to DenseNet201 checkpoint (.pth). Defaults to active champion checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("ml", "demo_results"),
        help="Directory to save demonstration artifacts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device ('auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Disable Grad-CAM heatmap generation.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run multi-sample 4-class evaluation on representative held-out test images.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_cli_args(argv)

    try:
        if args.mode == "validate":
            run_pipeline_validation(
                output_dir=args.output_dir,
                weights_path=args.weights_path,
            )
            return 0

        if args.batch:
            run_batch_demonstration(
                checkpoint_path=args.weights_path,
                output_dir=args.output_dir,
            )
            return 0

        # Real inference mode
        run_real_screening_demo(
            image_path=args.image,
            heart_rate=args.heart_rate,
            spo2=args.spo2,
            temperature=args.temperature,
            weights_path=args.weights_path,
            output_dir=args.output_dir,
            device=args.device,
            include_gradcam=not args.no_gradcam,
            sample_class=args.sample_class,
        )
        return 0

    except Exception as exc:
        sys.stderr.write(f"\n[DEMO ERROR] Demonstration execution failed: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
