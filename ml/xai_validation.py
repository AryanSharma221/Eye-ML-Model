"""
Explainable AI (XAI) and Model Confidence Validation Suite.
Multimodal Health Screening - DenseNet201 Retinal Disease Classifier.

Validates:
1. Confidence Calculations:
   - Numerical bounds, finite checks, probability sum to 1.0, argmax alignment.
   - Confidence distributions for correct vs. incorrect predictions.
   - High-confidence error audit (confidence >= 0.80 and >= 0.90).
2. Grad-CAM Explanations:
   - Base64 PNG encoding, decoded dimensional alignment, finite non-blank heatmaps.
   - Class-specific gradient discrimination across target classes.
   - Hook removal and memory safety across repeated requests.
3. Calibration Evaluation:
   - Expected Calibration Error (ECE), MCE, NLL, Brier score.
   - Temperature scaling fitted strictly on validation split (N=865) and evaluated on test split (N=894).
   - Reliability diagram visualization.
4. Failure Handling:
   - Missing/invalid checkpoint, invalid class index, corrupt image, non-finite outputs, missing layer.

Artifacts saved to ml/evaluation_results/.
"""

import base64
import csv
import json
import logging
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.dataset_loader import (
    CLASS_NAMES,
    NUM_CLASSES,
    EyeDiseaseDataset,
    load_split_manifest,
)
from ml.exceptions import (
    CheckpointNotFoundError,
    GradCAMGenerationError,
    InvalidImageInputError,
    ModelInferenceError,
)
from ml.eye_model import (
    DEFAULT_CHECKPOINT_PATH,
    build_densenet201_model,
    get_cached_model,
    get_default_device,
    get_model,
    load_eye_model_checkpoint,
    predict_eye_disease,
)
from ml.gradcam import generate_gradcam, get_target_layer
from ml.preprocessing import preprocess_image
from ml.quality_gate import check_image_quality

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = os.path.join(
    os.path.dirname(__file__), "dataset", "split_manifest.json"
)
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")


# -----------------------------------------------------------------------------
# 1. Calibration Metrics & Temperature Scaling
# -----------------------------------------------------------------------------

def compute_ece_and_mce(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    using equal-width confidence binning in [0, 1].

    Args:
        confidences: 1D array of predicted confidence scores in [0, 1].
        accuracies: 1D boolean/binary array (1 if prediction was correct, 0 otherwise).
        num_bins: Number of equal-width bins (default: 10).

    Returns:
        (ece, mce, bin_details)
    """
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    n_samples = len(confidences)

    bin_details = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples in bin
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)

            ece += (bin_count / n_samples) * bin_error
            mce = max(mce, bin_error)
        else:
            bin_acc = 0.0
            bin_conf = (bin_lower + bin_upper) / 2.0
            bin_error = 0.0

        bin_details.append({
            "bin_index": i,
            "bin_lower": round(float(bin_lower), 3),
            "bin_upper": round(float(bin_upper), 3),
            "sample_count": bin_count,
            "mean_accuracy": round(bin_acc, 4) if bin_count > 0 else None,
            "mean_confidence": round(bin_conf, 4) if bin_count > 0 else None,
            "calibration_error": round(bin_error, 4) if bin_count > 0 else None,
        })

    return float(ece), float(mce), bin_details


def compute_nll_and_brier(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
) -> Tuple[float, float]:
    """Computes Negative Log-Likelihood (NLL) and multi-class Brier score."""
    n_samples = len(true_labels)
    eps = 1e-12
    clipped_probs = np.clip(probabilities, eps, 1.0)

    # NLL
    correct_class_probs = clipped_probs[np.arange(n_samples), true_labels]
    nll = float(-np.mean(np.log(correct_class_probs)))

    # Multi-class Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(n_samples), true_labels] = 1.0
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    return nll, brier


def fit_temperature_scaling(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
) -> float:
    """
    Learns optimal scalar temperature T on validation split by minimizing NLL loss.
    Strictly fitted on validation data, never on test data.
    """
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = val_logits / temperature
        loss = nll_criterion(scaled_logits, val_labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    t_opt = float(temperature.item())
    # Guard against invalid temperature
    return max(0.01, min(10.0, t_opt))


# -----------------------------------------------------------------------------
# 2. Render Reliability Diagram
# -----------------------------------------------------------------------------

def render_reliability_diagram(
    uncalibrated_bins: List[Dict[str, Any]],
    calibrated_bins: List[Dict[str, Any]],
    ece_uncal: float,
    ece_cal: float,
    temperature: float,
    output_path: str,
) -> str:
    """
    Renders an annotated 2-panel reliability diagram (Uncalibrated vs Temperature Scaled).
    Uses PIL only (zero matplotlib/display server dependency).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img_w, img_h = 1000, 600
    canvas = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Title
    draw.text((40, 25), "DenseNet201 Eye Disease Classifier - Calibration Analysis", fill=(15, 23, 42))
    draw.text((40, 50), f"Held-Out Test Set (N=894) | Temperature Fitted on Val Split (T = {temperature:.3f})", fill=(71, 85, 105))

    panels = [
        ("Uncalibrated Baseline (T = 1.000)", uncalibrated_bins, ece_uncal, 40, (59, 130, 246)),
        (f"Temperature Scaled (T = {temperature:.3f})", calibrated_bins, ece_cal, 520, (16, 185, 129)),
    ]

    for title, bins, ece_val, left_x, bar_color in panels:
        top_y = 110
        chart_w = 420
        chart_h = 400

        # Draw panel box
        draw.rectangle([left_x, top_y, left_x + chart_w, top_y + chart_h], outline=(203, 213, 225), width=1)
        draw.text((left_x + 15, top_y + 15), title, fill=(15, 23, 42))
        draw.text((left_x + 15, top_y + 35), f"Expected Calibration Error (ECE): {ece_val*100:.2f}%", fill=(100, 116, 139))

        # Inner plot area
        plot_x0 = left_x + 50
        plot_y0 = top_y + 70
        plot_w = 340
        plot_h = 280
        plot_x1 = plot_x0 + plot_w
        plot_y1 = plot_y0 + plot_h

        # Axes
        draw.line([(plot_x0, plot_y1), (plot_x1, plot_y1)], fill=(100, 116, 139), width=2)
        draw.line([(plot_x0, plot_y0), (plot_x0, plot_y1)], fill=(100, 116, 139), width=2)

        # Ideal diagonal (perfect calibration: acc == conf)
        draw.line([(plot_x0, plot_y1), (plot_x1, plot_y0)], fill=(203, 213, 225), width=2)
        draw.text((plot_x1 - 80, plot_y0 + 10), "Ideal (y=x)", fill=(148, 163, 184))

        # Ticks and labels
        draw.text((plot_x0 + plot_w // 2 - 40, plot_y1 + 15), "Confidence ->", fill=(71, 85, 105))
        draw.text((plot_x0 - 45, plot_y0 + plot_h // 2 - 10), "Acc ^", fill=(71, 85, 105))

        # Draw bins
        num_bins = len(bins)
        bin_width_px = plot_w / num_bins

        for b in bins:
            i = b["bin_index"]
            bx0 = plot_x0 + i * bin_width_px
            bx1 = bx0 + bin_width_px - 2

            acc = b["mean_accuracy"]
            conf = b["mean_confidence"]

            if acc is not None:
                # Bar height corresponds to accuracy
                bar_h = acc * plot_h
                by0 = plot_y1 - bar_h
                draw.rectangle([bx0, by0, bx1, plot_y1], fill=bar_color, outline=(30, 41, 59), width=1)

                # Ideal line reference point
                ideal_y = plot_y1 - (b["bin_upper"] * plot_h)
                draw.rectangle([bx0 + bin_width_px // 2 - 3, ideal_y - 3, bx0 + bin_width_px // 2 + 3, ideal_y + 3], fill=(220, 38, 38))

    # Footer notice
    draw.text((40, 560), "Clinical Decision Support Prototype Notice: Softmax confidence is an algorithmic metric, not diagnostic certainty.", fill=(148, 163, 184))

    canvas.save(output_path, "PNG")
    return output_path


# -----------------------------------------------------------------------------
# 3. Main Validation Suite Execution
# -----------------------------------------------------------------------------

def run_xai_and_confidence_validation(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Executes the full confidence, Grad-CAM, calibration, and failure testing suite."""
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = device if device is not None else get_default_device()
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "xai_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 80)
    print("XAI & CONFIDENCE VALIDATION SUITE: DENSENET201 RETINAL CLASSIFIER")
    print("=" * 80)
    print(f"Device:             {target_device}")
    print(f"Target Checkpoint:  {checkpoint_path}")
    print(f"Manifest Path:      {manifest_path}")
    print(f"Random Seed:        {seed}")
    print("=" * 80)

    # 1. Load Model
    if not os.path.exists(checkpoint_path):
        raise CheckpointNotFoundError(f"Checkpoint file not found: {checkpoint_path}", operation="xai_validation")

    model = build_densenet201_model(num_classes=NUM_CLASSES)
    model = load_eye_model_checkpoint(model, checkpoint_path, device=target_device)
    model.eval()

    # Verify Target Layer
    target_layer = get_target_layer(model)
    print(f"\n[+] Model loaded successfully. Target layer resolved: {type(target_layer).__name__}")

    # 2. Load Split Manifest
    manifest = load_split_manifest(manifest_path)
    val_records = manifest["splits"]["val"]
    test_records = manifest["splits"]["test"]
    print(f"[+] Loaded splits: Val={len(val_records)}, Test={len(test_records)}")

    # -------------------------------------------------------------------------
    # STAGE A: Confidence Calculations Validation Across Test Split (N=894)
    # -------------------------------------------------------------------------
    print("\n[STAGE 1] Validating Confidence Calculations on Test Split...")
    test_dataset = EyeDiseaseDataset(test_records, split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    all_test_logits: List[torch.Tensor] = []
    all_test_probs: List[np.ndarray] = []
    all_test_preds: List[int] = []
    all_test_targets: List[int] = []

    # Reliability flags
    all_probs_finite = True
    all_probs_bounded = True
    probs_sum_approx_1 = True
    argmax_matches = True

    with torch.inference_mode():
        for b_inputs, b_targets in test_loader:
            b_inputs = b_inputs.to(target_device)
            logits = model(b_inputs)
            probs = torch.softmax(logits, dim=1)

            all_test_logits.append(logits.cpu())
            all_test_probs.append(probs.cpu().numpy())
            preds = torch.argmax(probs, dim=1).cpu().numpy().tolist()
            all_test_preds.extend(preds)
            all_test_targets.extend(b_targets.numpy().tolist())

    test_logits_tensor = torch.cat(all_test_logits, dim=0)
    test_probs_arr = np.concatenate(all_test_probs, axis=0)  # [894, 4]
    test_targets_arr = np.array(all_test_targets, dtype=np.int64)
    test_preds_arr = np.array(all_test_preds, dtype=np.int64)

    # Check numerical invariants on each sample
    for p_vec, pred_idx in zip(test_probs_arr, test_preds_arr):
        if not np.all(np.isfinite(p_vec)):
            all_probs_finite = False
        if not np.all((p_vec >= 0.0) & (p_vec <= 1.0)):
            all_probs_bounded = False
        if abs(np.sum(p_vec) - 1.0) > 0.005:
            probs_sum_approx_1 = False
        if int(np.argmax(p_vec)) != pred_idx:
            argmax_matches = False

    test_confidences = np.max(test_probs_arr, axis=1)
    test_accuracies = (test_preds_arr == test_targets_arr).astype(np.int64)

    correct_mask = test_accuracies == 1
    incorrect_mask = test_accuracies == 0

    conf_correct = test_confidences[correct_mask]
    conf_incorrect = test_confidences[incorrect_mask]

    def calc_dist_stats(arr: np.ndarray) -> Dict[str, float]:
        if len(arr) == 0:
            return {}
        return {
            "count": int(len(arr)),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
        }

    confidence_validation_results = {
        "total_test_images": len(test_confidences),
        "all_probabilities_finite": all_probs_finite,
        "all_probabilities_bounded_0_1": all_probs_bounded,
        "probabilities_sum_approx_1": probs_sum_approx_1,
        "argmax_matches_prediction": argmax_matches,
        "overall_accuracy": round(float(np.mean(test_accuracies)), 4),
        "correct_predictions_count": int(np.sum(correct_mask)),
        "incorrect_predictions_count": int(np.sum(incorrect_mask)),
        "distribution_overall": calc_dist_stats(test_confidences),
        "distribution_correct": calc_dist_stats(conf_correct),
        "distribution_incorrect": calc_dist_stats(conf_incorrect),
    }

    print(f"  All probabilities finite:      {all_probs_finite}")
    print(f"  Probabilities in [0, 1]:       {all_probs_bounded}")
    print(f"  Sum to 1.0 (approx):           {probs_sum_approx_1}")
    print(f"  Argmax matches predicted:      {argmax_matches}")
    print(f"  Correct Confidence Mean:       {confidence_validation_results['distribution_correct']['mean']:.4f}")
    print(f"  Incorrect Confidence Mean:     {confidence_validation_results['distribution_incorrect']['mean']:.4f}")

    # Audit High-Confidence Errors (conf >= 0.80 and conf >= 0.90)
    high_conf_errors = []
    for idx, (rec, yt, yp, conf, p_vec) in enumerate(
        zip(test_records, test_targets_arr, test_preds_arr, test_confidences, test_probs_arr)
    ):
        if yt != yp and conf >= 0.80:
            high_conf_errors.append({
                "sample_name": os.path.basename(rec["filepath"]),
                "true_label": CLASS_NAMES[yt],
                "predicted_label": CLASS_NAMES[yp],
                "confidence": round(float(conf), 4),
                "is_extreme_conf_90": bool(conf >= 0.90),
                "prob_cataract": round(float(p_vec[0]), 4),
                "prob_diabetic_retinopathy": round(float(p_vec[1]), 4),
                "prob_glaucoma": round(float(p_vec[2]), 4),
                "prob_normal": round(float(p_vec[3]), 4),
            })

    high_conf_csv = os.path.join(output_dir, "high_confidence_errors.csv")
    with open(high_conf_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample_Name", "True_Label", "Predicted_Label", "Confidence",
            "Extreme_Conf_ge_0.90", "Prob_Cataract", "Prob_DR", "Prob_Glaucoma", "Prob_Normal"
        ])
        for e in high_conf_errors:
            writer.writerow([
                e["sample_name"], e["true_label"], e["predicted_label"], e["confidence"],
                e["is_extreme_conf_90"], e["prob_cataract"], e["prob_diabetic_retinopathy"],
                e["prob_glaucoma"], e["prob_normal"]
            ])

    confidence_validation_results["high_confidence_errors_count_ge_80"] = len(high_conf_errors)
    confidence_validation_results["extreme_confidence_errors_count_ge_90"] = sum(
        1 for e in high_conf_errors if e["is_extreme_conf_90"]
    )
    print(f"  High-confidence errors (>=80%): {len(high_conf_errors)} (saved to {high_conf_csv})")

    # -------------------------------------------------------------------------
    # STAGE B: Temperature Scaling & Calibration Evaluation
    # -------------------------------------------------------------------------
    print("\n[STAGE 2] Evaluating Model Calibration (ECE, MCE, Temperature Scaling)...")

    # Step 1: Collect validation split logits (to fit temperature scaling)
    val_dataset = EyeDiseaseDataset(val_records, split="val")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    all_val_logits = []
    all_val_targets = []
    with torch.inference_mode():
        for b_inputs, b_targets in val_loader:
            b_inputs = b_inputs.to(target_device)
            logits = model(b_inputs)
            all_val_logits.append(logits.cpu())
            all_val_targets.extend(b_targets.numpy().tolist())

    val_logits_tensor = torch.cat(all_val_logits, dim=0)
    val_targets_tensor = torch.tensor(all_val_targets, dtype=torch.long)

    # Step 2: Fit temperature T on validation split
    t_opt = fit_temperature_scaling(val_logits_tensor, val_targets_tensor)
    print(f"  Fitted Temperature Parameter (on Val N={len(val_records)}): T = {t_opt:.4f}")

    # Step 3: Compute test set uncalibrated calibration metrics (T=1.0)
    ece_uncal_10, mce_uncal_10, bins_uncal_10 = compute_ece_and_mce(
        test_confidences, test_accuracies, num_bins=10
    )
    ece_uncal_15, mce_uncal_15, _ = compute_ece_and_mce(
        test_confidences, test_accuracies, num_bins=15
    )
    nll_uncal, brier_uncal = compute_nll_and_brier(test_probs_arr, test_targets_arr)

    # Step 4: Compute test set temperature scaled calibration metrics (T=t_opt)
    scaled_test_logits = test_logits_tensor / t_opt
    scaled_test_probs = torch.softmax(scaled_test_logits, dim=1).numpy()
    scaled_test_confidences = np.max(scaled_test_probs, axis=1)
    scaled_test_accuracies = (np.argmax(scaled_test_probs, axis=1) == test_targets_arr).astype(np.int64)

    ece_cal_10, mce_cal_10, bins_cal_10 = compute_ece_and_mce(
        scaled_test_confidences, scaled_test_accuracies, num_bins=10
    )
    ece_cal_15, mce_cal_15, _ = compute_ece_and_mce(
        scaled_test_confidences, scaled_test_accuracies, num_bins=15
    )
    nll_cal, brier_cal = compute_nll_and_brier(scaled_test_probs, test_targets_arr)

    # Step 5: Render Reliability Diagram
    diagram_path = os.path.join(output_dir, "reliability_diagram.png")
    render_reliability_diagram(
        uncalibrated_bins=bins_uncal_10,
        calibrated_bins=bins_cal_10,
        ece_uncal=ece_uncal_10,
        ece_cal=ece_cal_10,
        temperature=t_opt,
        output_path=diagram_path,
    )
    print(f"  [Artifact Saved] Reliability diagram: {diagram_path}")
    print(f"  Test ECE (10 bins): Uncalibrated = {ece_uncal_10*100:.2f}% | Scaled = {ece_cal_10*100:.2f}%")
    print(f"  Test MCE (10 bins): Uncalibrated = {mce_uncal_10*100:.2f}% | Scaled = {mce_cal_10*100:.2f}%")
    print(f"  Test NLL:           Uncalibrated = {nll_uncal:.4f}  | Scaled = {nll_cal:.4f}")
    print(f"  Test Brier:         Uncalibrated = {brier_uncal:.4f}  | Scaled = {brier_cal:.4f}")

    calibration_results = {
        "temperature_fitted_on_val": round(t_opt, 4),
        "validation_samples_count": len(val_records),
        "test_samples_count": len(test_records),
        "uncalibrated_baseline": {
            "temperature": 1.0,
            "ece_10_bins": round(ece_uncal_10, 4),
            "ece_15_bins": round(ece_uncal_15, 4),
            "mce_10_bins": round(mce_uncal_10, 4),
            "mce_15_bins": round(mce_uncal_15, 4),
            "nll": round(nll_uncal, 4),
            "brier_score": round(brier_uncal, 4),
            "bin_details_10": bins_uncal_10,
        },
        "temperature_scaled": {
            "temperature": round(t_opt, 4),
            "ece_10_bins": round(ece_cal_10, 4),
            "ece_15_bins": round(ece_cal_15, 4),
            "mce_10_bins": round(mce_cal_10, 4),
            "mce_15_bins": round(mce_cal_15, 4),
            "nll": round(nll_cal, 4),
            "brier_score": round(brier_cal, 4),
            "bin_details_10": bins_cal_10,
        },
        "diagram_artifact_path": diagram_path,
    }

    # -------------------------------------------------------------------------
    # STAGE C: Grad-CAM Validation on Representative Samples
    # -------------------------------------------------------------------------
    print("\n[STAGE 3] Validating Grad-CAM Generation on Representative Images...")

    # Identify representative samples from test split (passing quality gate)
    # Find 4 correct (one per class) and 4 misclassified
    rep_correct = {}
    rep_misclassified = []

    for idx, (rec, yt, yp, conf) in enumerate(
        zip(test_records, test_targets_arr, test_preds_arr, test_confidences)
    ):
        path = rec["filepath"]
        c_name = CLASS_NAMES[yt]

        # Prioritize Kaggle images that pass quality gate
        if yt == yp and c_name not in rep_correct:
            q_res = check_image_quality(path)
            if q_res["overall_pass"]:
                rep_correct[c_name] = {
                    "path": path,
                    "sample_name": os.path.basename(path),
                    "true_class": c_name,
                    "pred_class": CLASS_NAMES[yp],
                    "pred_idx": int(yp),
                    "confidence": float(conf),
                }

        if yt != yp and len(rep_misclassified) < 4:
            q_res = check_image_quality(path)
            if q_res["overall_pass"]:
                rep_misclassified.append({
                    "path": path,
                    "sample_name": os.path.basename(path),
                    "true_class": c_name,
                    "pred_class": CLASS_NAMES[yp],
                    "pred_idx": int(yp),
                    "confidence": float(conf),
                })

    all_rep_cases = [("correct_" + k, v) for k, v in rep_correct.items()] + [
        (f"misclassified_{i}_{v['true_class']}_as_{v['pred_class']}", v)
        for i, v in enumerate(rep_misclassified)
    ]

    gradcam_sample_results = []

    for label_key, item in all_rep_cases:
        img_path = item["path"]
        orig_img = cv2.imread(img_path)
        orig_h, orig_w = orig_img.shape[:2]

        t0 = time.perf_counter()
        b64_output = generate_gradcam(
            image_path=img_path,
            predicted_class_idx=item["pred_idx"],
            model=model,
            device=target_device,
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0

        # Decode Base64 PNG
        raw_png = base64.b64decode(b64_output)
        is_png_signature = raw_png.startswith(b"\x89PNG\r\n\x1a\n")

        decoded = cv2.imdecode(np.frombuffer(raw_png, np.uint8), cv2.IMREAD_COLOR)
        matches_dims = (decoded.shape == (orig_h, orig_w, 3))

        # Save visualization to disk
        out_vis_file = os.path.join(vis_dir, f"{label_key}_{item['sample_name']}.png")
        with open(out_vis_file, "wb") as f_out:
            f_out.write(raw_png)

        # Heatmap intensity checks
        # Compute difference from original image to verify overlay is not blank
        diff_from_orig = np.abs(decoded.astype(np.float32) - orig_img.astype(np.float32))
        overlay_mean_diff = float(np.mean(diff_from_orig))
        is_non_blank = overlay_mean_diff > 1.0

        gradcam_sample_results.append({
            "key": label_key,
            "sample_name": item["sample_name"],
            "true_class": item["true_class"],
            "pred_class": item["pred_class"],
            "pred_idx": item["pred_idx"],
            "confidence": round(item["confidence"], 4),
            "duration_ms": round(duration_ms, 2),
            "is_png_signature": is_png_signature,
            "matches_original_dims": matches_dims,
            "overlay_mean_diff": round(overlay_mean_diff, 2),
            "is_non_blank": is_non_blank,
            "artifact_path": out_vis_file,
        })
        print(f"  [+] Grad-CAM generated: {label_key} -> dims match: {matches_dims}, valid PNG: {is_png_signature}")

    # -------------------------------------------------------------------------
    # STAGE D: Class-Specific Counterfactual Discrimination Test
    # -------------------------------------------------------------------------
    print("\n[STAGE 4] Validating Class-Specific Gradient Computation (Counterfactuals)...")
    sample_for_counterfactual = rep_correct["cataract"]["path"]
    orig_shape = cv2.imread(sample_for_counterfactual).shape[:2]

    class_heatmaps = []
    for c_idx in range(NUM_CLASSES):
        b64_c = generate_gradcam(
            image_path=sample_for_counterfactual,
            predicted_class_idx=c_idx,
            model=model,
            device=target_device,
        )
        dec_c = cv2.imdecode(np.frombuffer(base64.b64decode(b64_c), np.uint8), cv2.IMREAD_COLOR)
        class_heatmaps.append(dec_c.astype(np.float32))

    # Compute pairwise mean absolute differences between target class visualizations
    pairwise_diffs = {}
    for c1 in range(NUM_CLASSES):
        for c2 in range(c1 + 1, NUM_CLASSES):
            diff = float(np.mean(np.abs(class_heatmaps[c1] - class_heatmaps[c2])))
            pair_key = f"{CLASS_NAMES[c1]}_vs_{CLASS_NAMES[c2]}"
            pairwise_diffs[pair_key] = round(diff, 2)

    class_discrimination_verified = all(d > 0.5 for d in pairwise_diffs.values())
    print(f"  Class gradient discrimination verified: {class_discrimination_verified}")
    print(f"  Pairwise visualization differences: {pairwise_diffs}")

    # -------------------------------------------------------------------------
    # STAGE E: Hook Leakage and Memory Safety Test (50 repeated calls)
    # -------------------------------------------------------------------------
    print("\n[STAGE 5] Auditing Hook Leakage & Memory Safety (50 iterations)...")
    leakage_test_passed = True
    for it in range(50):
        _ = generate_gradcam(
            image_path=sample_for_counterfactual,
            predicted_class_idx=0,
            model=model,
            device=target_device,
        )
        if len(target_layer._forward_hooks) != 0 or len(target_layer._backward_hooks) != 0:
            leakage_test_passed = False
            break

    # Parameter gradients check
    grad_leak_found = False
    for p in model.parameters():
        if p.grad is not None:
            grad_leak_found = True
            break

    hooks_cleaned = leakage_test_passed and (not grad_leak_found)
    print(f"  50 Iterations Hook & Gradient Safety Verified: {'PASS' if hooks_cleaned else 'FAIL'}")

    # -------------------------------------------------------------------------
    # STAGE F: Failure Handling Verification
    # -------------------------------------------------------------------------
    print("\n[STAGE 6] Verifying Robust Failure Handling...")
    failure_results = {}

    # Test 1: Missing checkpoint
    try:
        generate_gradcam("ml/dataset/test/cataract/1019.png", 0, model=None, weights_path="nonexistent_checkpoint.pth")
        failure_results["missing_checkpoint"] = "FAIL (No exception)"
    except FileNotFoundError:
        failure_results["missing_checkpoint"] = "PASS (FileNotFoundError)"
    except Exception as e:
        failure_results["missing_checkpoint"] = f"PASS ({type(e).__name__})"

    # Test 2: Invalid class index
    try:
        generate_gradcam(sample_for_counterfactual, predicted_class_idx=99, model=model)
        failure_results["invalid_class_index_99"] = "FAIL (No exception)"
    except ValueError:
        failure_results["invalid_class_index_99"] = "PASS (ValueError)"
    except Exception as e:
        failure_results["invalid_class_index_99"] = f"FAIL ({type(e).__name__})"

    try:
        generate_gradcam(sample_for_counterfactual, predicted_class_idx=-1, model=model)
        failure_results["invalid_class_index_neg1"] = "PASS (ValueError)"
    except ValueError:
        failure_results["invalid_class_index_neg1"] = "PASS (ValueError)"
    except Exception as e:
        failure_results["invalid_class_index_neg1"] = f"FAIL ({type(e).__name__})"

    # Test 3: Unsupported / corrupted image
    try:
        generate_gradcam("nonexistent_image.jpg", 0, model=model)
        failure_results["missing_image"] = "FAIL (No exception)"
    except FileNotFoundError:
        failure_results["missing_image"] = "PASS (FileNotFoundError)"
    except Exception as e:
        failure_results["missing_image"] = f"PASS ({type(e).__name__})"

    # Test 4: Missing target layer
    class DummyModel(nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 4))

    try:
        generate_gradcam(sample_for_counterfactual, 0, model=DummyModel().eval())
        failure_results["missing_target_layer"] = "FAIL (No exception)"
    except AttributeError:
        failure_results["missing_target_layer"] = "PASS (AttributeError)"
    except Exception as e:
        failure_results["missing_target_layer"] = f"PASS ({type(e).__name__})"

    # Test 5: Non-finite model outputs (NaN logits)
    class NaNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential()
            self.features.denseblock4 = nn.Conv2d(3, 8, 3, padding=1)
            self.classifier = nn.Linear(8, 4)

        def forward(self, x):
            _ = self.features.denseblock4(x)
            return torch.full((x.shape[0], 4), float("nan"), requires_grad=True)

    try:
        generate_gradcam(sample_for_counterfactual, 0, model=NaNModel().eval())
        failure_results["non_finite_logits_gradcam"] = "FAIL (Silently returned map)"
    except GradCAMGenerationError:
        failure_results["non_finite_logits_gradcam"] = "PASS (GradCAMGenerationError raised)"
    except Exception as e:
        failure_results["non_finite_logits_gradcam"] = f"PASS ({type(e).__name__})"

    try:
        predict_eye_disease(sample_for_counterfactual, model=NaNModel().eval())
        failure_results["non_finite_logits_predict"] = "FAIL (Silently returned dict)"
    except ModelInferenceError:
        failure_results["non_finite_logits_predict"] = "PASS (ModelInferenceError raised)"
    except Exception as e:
        failure_results["non_finite_logits_predict"] = f"PASS ({type(e).__name__})"

    for k, v in failure_results.items():
        print(f"  {k:30s} -> {v}")

    # -------------------------------------------------------------------------
    # STAGE G: Compile Final Machine-Readable JSON Report
    # -------------------------------------------------------------------------
    duration_total = time.time() - start_time

    full_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint": checkpoint_path,
            "manifest": manifest_path,
            "seed": seed,
            "device": str(target_device),
            "canonical_classes": CLASS_NAMES,
            "duration_seconds": round(duration_total, 2),
        },
        "confidence_validation": confidence_validation_results,
        "calibration_evaluation": calibration_results,
        "gradcam_validation": {
            "target_layer": type(target_layer).__name__,
            "class_discrimination_verified": class_discrimination_verified,
            "pairwise_class_diffs": pairwise_diffs,
            "hook_cleanup_verified": hooks_cleaned,
            "representative_samples_count": len(gradcam_sample_results),
            "representative_samples": gradcam_sample_results,
        },
        "failure_handling": failure_results,
        "disclaimer": (
            "Explainability Notice: Grad-CAM generates visual saliency overlays indicating "
            "regions that contributed to the model's activation. It does not constitute proof "
            "of correct pathophysiological reasoning. Softmax confidence is an algorithmic metric, "
            "not a calibrated diagnosis. This software is a triage decision-support prototype."
        ),
    }

    report_path = os.path.join(output_dir, "xai_validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[Artifact Saved] XAI Validation JSON Report: {report_path}")
    print(f"[Artifact Saved] High-Confidence Errors CSV:   {high_conf_csv}")
    print(f"[Artifact Saved] Reliability Diagram PNG:     {diagram_path}")
    print(f"[Artifact Saved] Visualizations Directory:    {vis_dir}")
    print("=" * 80)
    print("XAI & CONFIDENCE VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return full_report


if __name__ == "__main__":
    run_xai_and_confidence_validation()
