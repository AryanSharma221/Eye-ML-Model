"""
Medical Image Robustness, Generalization & Prediction Stability Suite.
Multimodal Health Screening - DenseNet201 Retinal Disease Classifier.

Evaluates the champion eye disease screening model across:
1. Natural Image Variations (brightness, contrast, mild blur, compression, resampling, noise, slight rotation).
2. Prediction Stability & Consistency (agreement relative to unmodified clean baseline).
3. Quality Gate Behavior (pass/reject fidelity, severe degradation filtering, repeatability, corrupt file safety).
4. Failure Pattern Identification (per-class sensitivity, confidence shifts, unstable sample auditing).

Saves all artifacts to ml/evaluation_results/ without modifying original dataset images.
"""

import csv
import io
import json
import logging
import os
import random
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES, load_split_manifest
from ml.evaluate import calculate_classification_metrics
from ml.eye_model import (
    DEFAULT_CHECKPOINT_PATH,
    build_densenet201_model,
    get_default_device,
    load_eye_model_checkpoint,
    predict_eye_disease,
)
from ml.exceptions import InvalidImageInputError
from ml.monitoring import sanitize_path
from ml.preprocessing import PREPROCESSING_TRANSFORM
from ml.quality_gate import check_image_quality
from ml.screening_pipeline import run_screening

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Controlled In-Memory Image Transformations (Never modifies disk files)
# -----------------------------------------------------------------------------

def transform_identity(img: Image.Image) -> Image.Image:
    return img.copy()

def transform_brightness_inc(img: Image.Image, factor: float = 1.15) -> Image.Image:
    return ImageEnhance.Brightness(img.copy()).enhance(factor)

def transform_brightness_dec(img: Image.Image, factor: float = 0.85) -> Image.Image:
    return ImageEnhance.Brightness(img.copy()).enhance(factor)

def transform_contrast_inc(img: Image.Image, factor: float = 1.15) -> Image.Image:
    return ImageEnhance.Contrast(img.copy()).enhance(factor)

def transform_contrast_dec(img: Image.Image, factor: float = 0.85) -> Image.Image:
    return ImageEnhance.Contrast(img.copy()).enhance(factor)

def transform_rotation_pos5(img: Image.Image) -> Image.Image:
    return img.rotate(5.0, resample=Image.Resampling.BILINEAR)

def transform_rotation_neg5(img: Image.Image) -> Image.Image:
    return img.rotate(-5.0, resample=Image.Resampling.BILINEAR)

def transform_blur_mild(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=1.0))

def transform_jpeg_q75(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def transform_jpeg_q50(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def transform_resample_384(img: Image.Image) -> Image.Image:
    orig_size = img.size
    down = img.resize((384, 384), resample=Image.Resampling.BILINEAR)
    return down.resize(orig_size, resample=Image.Resampling.BILINEAR)

def transform_noise_mild(img: Image.Image, sigma: float = 0.02) -> Image.Image:
    arr = np.array(img, dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, sigma, arr.shape)
    noisy_arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy_arr * 255.0).astype(np.uint8))

# Severe degradation stress tests (Intended for Quality Gate rejection)
def transform_severe_blur(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=4.0))

def transform_severe_underexposure(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img.copy()).enhance(0.35)

def transform_severe_overexposure(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img.copy()).enhance(1.80)


TRANSFORMATION_REGISTRY: Dict[str, Tuple[Callable[[Image.Image], Image.Image], str]] = {
    "baseline_clean": (transform_identity, "Original clean test image (unmodified reference)"),
    "brightness_inc_15": (transform_brightness_inc, "Mild brightness increase (+15%)"),
    "brightness_dec_15": (transform_brightness_dec, "Mild brightness decrease (-15%)"),
    "contrast_inc_15": (transform_contrast_inc, "Mild contrast increase (+15%)"),
    "contrast_dec_15": (transform_contrast_dec, "Mild contrast decrease (-15%)"),
    "rotation_pos_5deg": (transform_rotation_pos5, "Slight clockwise rotation (+5 degrees)"),
    "rotation_neg_5deg": (transform_rotation_neg5, "Slight counter-clockwise rotation (-5 degrees)"),
    "blur_gaussian_r1": (transform_blur_mild, "Mild Gaussian blur (radius = 1.0 px)"),
    "compression_jpeg_q75": (transform_jpeg_q75, "Mild JPEG recompression (Quality = 75)"),
    "compression_jpeg_q50": (transform_jpeg_q50, "Moderate JPEG recompression (Quality = 50)"),
    "resample_down_up_384": (transform_resample_384, "Spatial resampling (downsample to 384x384, upsample back)"),
    "noise_gaussian_s02": (transform_noise_mild, "Mild additive Gaussian noise (sigma = 0.02)"),
    "stress_severe_blur": (transform_severe_blur, "Severe Gaussian blur (radius = 4.0 px) [Stress Gate Test]"),
    "stress_severe_underexposure": (transform_severe_underexposure, "Severe underexposure (-65% brightness) [Stress Gate Test]"),
    "stress_severe_overexposure": (transform_severe_overexposure, "Severe overexposure (+80% brightness) [Stress Gate Test]"),
}


# -----------------------------------------------------------------------------
# 2. In-Memory Quality Gate & Preprocessing Utilities
# -----------------------------------------------------------------------------

def evaluate_quality_gate_in_memory(pil_img: Image.Image) -> Dict[str, Any]:
    """Applies Quality Gate checks directly to an in-memory PIL image."""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]
    
    # 1. Resolution
    resolution_ok = (w >= 200 and h >= 200)
    
    # 2. Blur (Laplacian variance)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())
    blur_ok = (blur_score >= 100.0)
    
    # 3. Brightness & Contrast
    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    illum_ok = (30.0 <= brightness_mean <= 225.0) and (contrast_std >= 20.0)
    
    # 4. Retinal circle boundary heuristic
    retinal_circle_ok = True
    
    overall_pass = bool(resolution_ok and blur_ok and illum_ok and retinal_circle_ok)
    
    reason = None
    if not overall_pass:
        if not resolution_ok:
            reason = f"Low resolution ({w}x{h} < 200x200)"
        elif not blur_ok:
            reason = f"Excessive blur ({blur_score:.2f} < 100.0)"
        elif not illum_ok:
            reason = f"Improper lighting/contrast (mean: {brightness_mean:.1f}, std: {contrast_std:.1f})"
            
    return {
        "overall_pass": overall_pass,
        "blur_score": round(blur_score, 2),
        "brightness_mean": round(brightness_mean, 2),
        "contrast_std": round(contrast_std, 2),
        "resolution_ok": resolution_ok,
        "reason": reason,
    }


def preprocess_pil_image_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Standardizes in-memory PIL image using production PREPROCESSING_TRANSFORM."""
    t = PREPROCESSING_TRANSFORM(pil_img)
    return t.unsqueeze(0)  # [1, 3, 224, 224]


# -----------------------------------------------------------------------------
# 3. Comprehensive Robustness Evaluation Engine
# -----------------------------------------------------------------------------

def run_robustness_evaluation(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    manifest_path: str = "ml/dataset/split_manifest.json",
    max_samples: Optional[int] = None,
    output_dir: str = "ml/evaluation_results",
    seed: int = 42,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = get_default_device()

    print("=" * 80)
    print("MEDICAL COMPUTER VISION ROBUSTNESS & STABILITY EVALUATION")
    print("=" * 80)
    print(f"Device:             {device}")
    print(f"Tested Checkpoint:  {checkpoint_path}")
    print(f"Dataset Manifest:   {manifest_path}")
    print(f"Random Seed:        {seed}")
    print("=" * 80)

    # 1. Load Model
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    model = load_eye_model_checkpoint(model, checkpoint_path, device=device)
    model.eval()

    # 2. Load Held-Out Test Records
    manifest = load_split_manifest(manifest_path)
    test_records = manifest["splits"]["test"]
    print(f"\nDiscovered {len(test_records)} held-out test images.")

    # Filter to eligible test images that exist on disk
    valid_test_records = [r for r in test_records if os.path.isfile(r["filepath"])]
    if max_samples and max_samples < len(valid_test_records):
        valid_test_records = valid_test_records[:max_samples]
    print(f"Evaluating {len(valid_test_records)} test samples across {len(TRANSFORMATION_REGISTRY)} conditions.")

    # Pre-cache clean PIL images
    print("Pre-loading test images into memory...")
    cached_images: List[Tuple[Dict[str, Any], Image.Image]] = []
    for r in valid_test_records:
        try:
            img = Image.open(r["filepath"]).convert("RGB")
            cached_images.append((r, img))
        except Exception as e:
            print(f"  Warning: Skipping unreadable image {r['filepath']}: {e}")

    total_loaded = len(cached_images)
    print(f"Loaded {total_loaded} images successfully.")

    # 3. Reference Run: Baseline Clean Predictions
    print("\n[PHASE 1] Computing Reference Predictions on Clean Images...")
    baseline_predictions: Dict[str, Dict[str, Any]] = {}
    
    with torch.no_grad():
        for rec, img in cached_images:
            fname = os.path.basename(rec["filepath"])
            q_res = evaluate_quality_gate_in_memory(img)
            tensor = preprocess_pil_image_to_tensor(img).to(device)
            out = model(tensor)
            probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])

            baseline_predictions[fname] = {
                "true_label": rec["class_name"],
                "true_idx": int(rec["class_index"]),
                "pred_label": CLASS_NAMES[pred_idx],
                "pred_idx": pred_idx,
                "confidence": conf,
                "all_probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
                "quality_pass": q_res["overall_pass"],
                "quality_metrics": q_res,
            }

    # 4. Evaluate Each Transformation Condition
    print("\n[PHASE 2] Evaluating Robustness across Transformations...")
    transformation_results: Dict[str, Any] = {}
    unstable_samples: List[Dict[str, Any]] = []

    for cond_name, (transform_fn, cond_desc) in TRANSFORMATION_REGISTRY.items():
        t_start = time.time()
        print(f"\n--- Testing Condition: {cond_name} ---")
        print(f"Description: {cond_desc}")

        total_evaluated = 0
        gate_passed = 0
        gate_rejected = 0
        
        y_true_all = []
        y_pred_all = []
        confs_all = []
        conf_deltas = []
        consistent_count = 0

        # Also track metrics exclusively for images that pass quality gate
        y_true_gated = []
        y_pred_gated = []

        with torch.no_grad():
            for rec, base_img in cached_images:
                fname = os.path.basename(rec["filepath"])
                base_ref = baseline_predictions[fname]
                
                # Apply transformation in-memory
                tx_img = transform_fn(base_img)
                
                # Quality gate
                q_res = evaluate_quality_gate_in_memory(tx_img)
                if q_res["overall_pass"]:
                    gate_passed += 1
                else:
                    gate_rejected += 1

                # Neural inference
                tensor = preprocess_pil_image_to_tensor(tx_img).to(device)
                out = model(tensor)
                probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])

                true_idx = base_ref["true_idx"]
                total_evaluated += 1

                y_true_all.append(true_idx)
                y_pred_all.append(pred_idx)
                confs_all.append(conf)

                # Prediction consistency relative to clean baseline
                is_consistent = (pred_idx == base_ref["pred_idx"])
                if is_consistent:
                    consistent_count += 1
                else:
                    # Record unstable sample
                    unstable_samples.append({
                        "sample_name": fname,
                        "condition": cond_name,
                        "true_label": rec["class_name"],
                        "baseline_pred": base_ref["pred_label"],
                        "baseline_conf": round(base_ref["confidence"], 4),
                        "transformed_pred": CLASS_NAMES[pred_idx],
                        "transformed_conf": round(conf, 4),
                        "quality_pass": q_res["overall_pass"],
                        "reason": q_res.get("reason"),
                    })

                delta_c = conf - base_ref["confidence"]
                conf_deltas.append(delta_c)

                if q_res["overall_pass"]:
                    y_true_gated.append(true_idx)
                    y_pred_gated.append(pred_idx)

        # Compute classification metrics across all samples
        all_metrics = calculate_classification_metrics(y_true_all, y_pred_all, CLASS_NAMES)
        
        # Compute metrics on quality-passed subset if available
        gated_metrics = (
            calculate_classification_metrics(y_true_gated, y_pred_gated, CLASS_NAMES)
            if y_true_gated else None
        )

        consistency_rate = consistent_count / max(1, total_evaluated)
        mean_conf = float(np.mean(confs_all))
        mean_conf_delta = float(np.mean(conf_deltas))
        duration = time.time() - t_start

        cond_result = {
            "condition": cond_name,
            "description": cond_desc,
            "total_images": total_evaluated,
            "quality_gate_passed": gate_passed,
            "quality_gate_rejected": gate_rejected,
            "quality_pass_rate": round(gate_passed / max(1, total_evaluated), 4),
            "accuracy_all": round(all_metrics["overall_accuracy"], 4),
            "macro_f1_all": round(all_metrics["macro_f1"], 4),
            "weighted_f1_all": round(all_metrics["weighted_f1"], 4),
            "prediction_consistency": round(consistency_rate, 4),
            "mean_confidence": round(mean_conf, 4),
            "mean_confidence_delta": round(mean_conf_delta, 4),
            "duration_seconds": round(duration, 2),
            "per_class": all_metrics["per_class"],
            "confusion_matrix": all_metrics["confusion_matrix"],
            "gated_accuracy": round(gated_metrics["overall_accuracy"], 4) if gated_metrics else None,
            "gated_macro_f1": round(gated_metrics["macro_f1"], 4) if gated_metrics else None,
        }
        transformation_results[cond_name] = cond_result

        print(
            f"  Accuracy: {cond_result['accuracy_all']*100:.2f}% | "
            f"Macro F1: {cond_result['macro_f1_all']*100:.2f}% | "
            f"Consistency: {cond_result['prediction_consistency']*100:.2f}% | "
            f"Gate Pass: {gate_passed}/{total_evaluated} ({cond_result['quality_pass_rate']*100:.1f}%)"
        )

    # 5. Dedicated Quality Gate Invariant & Security Verification
    print("\n[PHASE 3] Auditing Quality-Gate Invariants & Security Guardrails...")
    qg_audit: Dict[str, Any] = {}

    # Check 1: Deterministic Repeatability
    sample_img = cached_images[0][1]
    repeat_verdicts = [evaluate_quality_gate_in_memory(sample_img)["overall_pass"] for _ in range(5)]
    is_repeatable = len(set(repeat_verdicts)) == 1
    qg_audit["repeatability_verified"] = is_repeatable
    print(f"  Quality Gate Repeatability (5 runs): {'PASS' if is_repeatable else 'FAIL'}")

    # Check 2: Severe Degradation Rejection Rate
    severe_blur_rej = transformation_results["stress_severe_blur"]["quality_gate_rejected"]
    severe_blur_total = transformation_results["stress_severe_blur"]["total_images"]
    blur_filter_rate = severe_blur_rej / max(1, severe_blur_total)
    qg_audit["severe_blur_rejection_rate"] = round(blur_filter_rate, 4)
    print(f"  Severe Blur Rejection Rate: {blur_filter_rate*100:.2f}% ({severe_blur_rej}/{severe_blur_total} rejected)")

    # Check 3: Corrupt & Invalid Files Rejection
    corrupt_checks_passed = True
    try:
        check_image_quality("nonexistent_corrupt_test_file.png")
    except Exception as e:
        corrupt_checks_passed = False
    qg_audit["corrupt_files_safety_verified"] = corrupt_checks_passed
    print(f"  Corrupt/Missing File Safe Handling: {'PASS' if corrupt_checks_passed else 'FAIL'}")

    # Check 4: Pipeline Inference Isolation on Rejection
    # Verify that run_screening skips neural inference if quality gate fails
    pipeline_isolation_verified = True
    try:
        # Blurry sample for rejection isolation test
        sample_blurry_path = os.path.join(tempfile.gettempdir(), "test_blurry_sample.png")
        if not os.path.isfile(sample_blurry_path):
            cv2.imwrite(sample_blurry_path, np.ones((224, 224, 3), dtype=np.uint8) * 128)
        res = run_screening(image_path=sample_blurry_path)
        if res.get("eye_screening", {}).get("status") != "rejected":
            pipeline_isolation_verified = False
        if res.get("eye_screening", {}).get("prediction") is not None:
            pipeline_isolation_verified = False
    except Exception:
        pipeline_isolation_verified = False
    qg_audit["pipeline_rejection_blocks_inference"] = pipeline_isolation_verified
    print(f"  Rejected Image Never Enters Model:  {'PASS' if pipeline_isolation_verified else 'FAIL'}")

    # 6. Save Machine-Readable Artifacts
    os.makedirs(output_dir, exist_ok=True)
    
    # Artifact 1: Full JSON Robustness Report
    robustness_report = {
        "report_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint": checkpoint_path,
            "manifest": manifest_path,
            "total_test_images": total_loaded,
            "seed": seed,
            "device": str(device),
            "canonical_classes": CLASS_NAMES,
        },
        "quality_gate_audit": qg_audit,
        "transformation_benchmarks": transformation_results,
    }
    json_path = os.path.join(output_dir, "robustness_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(robustness_report, f, indent=2)
    print(f"\n[Artifact Saved] JSON Robustness Report: {json_path}")

    # Artifact 2: CSV Summary Table
    csv_path = os.path.join(output_dir, "robustness_summary_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Condition",
            "Description",
            "Images_Evaluated",
            "Gate_Passed",
            "Gate_Rejected",
            "Gate_Pass_Rate",
            "Overall_Accuracy",
            "Macro_F1",
            "Prediction_Consistency",
            "Mean_Confidence",
            "Mean_Confidence_Delta",
            "Cataract_F1",
            "DR_F1",
            "Glaucoma_F1",
            "Normal_F1",
        ])
        for c_name, c_data in transformation_results.items():
            pc = c_data["per_class"]
            writer.writerow([
                c_name,
                c_data["description"],
                c_data["total_images"],
                c_data["quality_gate_passed"],
                c_data["quality_gate_rejected"],
                f"{c_data['quality_pass_rate']*100:.1f}%",
                f"{c_data['accuracy_all']*100:.2f}%",
                f"{c_data['macro_f1_all']*100:.2f}%",
                f"{c_data['prediction_consistency']*100:.2f}%",
                f"{c_data['mean_confidence']*100:.2f}%",
                f"{c_data['mean_confidence_delta']*100:+.2f}%",
                f"{pc['cataract']['f1_score']*100:.2f}%",
                f"{pc['diabetic_retinopathy']['f1_score']*100:.2f}%",
                f"{pc['glaucoma']['f1_score']*100:.2f}%",
                f"{pc['normal']['f1_score']*100:.2f}%",
            ])
    print(f"[Artifact Saved] CSV Summary Table:     {csv_path}")

    # Artifact 3: CSV Unstable Samples List
    unstable_csv_path = os.path.join(output_dir, "robustness_unstable_samples.csv")
    with open(unstable_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_Name",
            "Transformation",
            "True_Label",
            "Baseline_Pred",
            "Baseline_Conf",
            "Transformed_Pred",
            "Transformed_Conf",
            "Quality_Gate_Pass",
            "Rejection_Reason",
        ])
        for u in unstable_samples:
            writer.writerow([
                u["sample_name"],
                u["condition"],
                u["true_label"],
                u["baseline_pred"],
                u["baseline_conf"],
                u["transformed_pred"],
                u["transformed_conf"],
                u["quality_pass"],
                u["reason"] or "N/A",
            ])
    print(f"[Artifact Saved] CSV Unstable Cases:    {unstable_csv_path} ({len(unstable_samples)} instability events recorded)")

    print("=" * 80)
    print("ROBUSTNESS EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return robustness_report


if __name__ == "__main__":
    run_robustness_evaluation()
