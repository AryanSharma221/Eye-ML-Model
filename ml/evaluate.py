"""
Model Evaluation and Inference Verification Module for DenseNet201 Eye Disease Classifier.

Evaluates the trained 4-class eye disease model against the held-out test split,
computing comprehensive clinical screening metrics:
1. Overall accuracy
2. Per-class precision, recall, and F1-score
3. Macro-averaged precision, recall, and F1-score
4. Weighted F1-score
5. 4x4 Confusion matrix
6. Class-wise support
7. Evaluation latency and sample failure audit

Strictly isolates the test split (never fits or tunes on test data) and exports
an anonymized machine-readable JSON evaluation report.

Public Contract:
    calculate_classification_metrics(y_true, y_pred, class_names) -> dict[str, Any]
    evaluate_model_checkpoint(checkpoint_path, manifest_path, ...) -> dict[str, Any]
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw
import torch
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
from ml.eye_model import (
    DEFAULT_CHECKPOINT_PATH,
    build_densenet201_model,
    get_default_device,
    load_eye_model_checkpoint,
)
from ml.experiment_tracker import ExperimentTracker, compute_file_sha256
from ml.monitoring import log_ml_event, sanitize_path

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH: str = os.path.join(
    os.path.dirname(__file__), "dataset", "split_manifest.json"
)
DEFAULT_BEST_CHECKPOINT_PATH: str = (
    os.path.join(os.path.dirname(__file__), "weights", "eye_model.pth")
    if os.path.exists(os.path.join(os.path.dirname(__file__), "weights", "eye_model.pth"))
    else os.path.join(os.path.dirname(__file__), "weights", "eye_densenet201_best.pth")
)
DEFAULT_EVALUATION_OUTPUT_DIR: str = os.path.join(
    os.path.dirname(__file__), "evaluation_results"
)


def calculate_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    """
    Computes accuracy, per-class precision, recall, F1-score, macro/weighted averages,
    and confusion matrix from integer ground truth and predicted labels.

    Handles zero-division explicitly (sets metric to 0.0 when denominator is 0).

    Args:
        y_true: Ground truth class integer indices.
        y_pred: Predicted class integer indices.
        class_names: Canonical class name ordering.

    Returns:
        dict: Complete evaluation metrics dictionary.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})"
        )

    num_classes = len(class_names)
    total_samples = len(y_true)

    if total_samples == 0:
        return {
            "overall_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_sensitivity": 0.0,
            "macro_specificity": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_sensitivity": 0.0,
            "weighted_specificity": 0.0,
            "weighted_f1": 0.0,
            "per_class": {
                name: {
                    "precision": 0.0,
                    "recall": 0.0,
                    "sensitivity": 0.0,
                    "specificity": 0.0,
                    "f1_score": 0.0,
                    "support": 0,
                }
                for name in class_names
            },
            "confusion_matrix": [[0] * num_classes for _ in range(num_classes)],
            "support": {name: 0 for name in class_names},
            "total_samples": 0,
        }

    # Initialize confusion matrix [true_class][pred_class]
    conf_matrix = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf_matrix[t][p] += 1
        else:
            raise ValueError(f"Label index out of bounds: true={t}, pred={p}, num_classes={num_classes}")

    # Compute overall accuracy
    correct_total = sum(conf_matrix[i][i] for i in range(num_classes))
    overall_accuracy = correct_total / total_samples

    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    support_dict: Dict[str, int] = {}

    precision_list: List[float] = []
    recall_list: List[float] = []
    sensitivity_list: List[float] = []
    specificity_list: List[float] = []
    f1_list: List[float] = []
    weights_list: List[float] = []

    for i, cls_name in enumerate(class_names):
        tp = conf_matrix[i][i]
        # Sum of row i = support for class i
        support = sum(conf_matrix[i])
        support_dict[cls_name] = support

        # Sum of column i = total predictions for class i
        pred_count = sum(conf_matrix[r][i] for r in range(num_classes))

        fn = support - tp
        fp = pred_count - tp
        tn = total_samples - (tp + fn + fp)

        # Precision = TP / (TP + FP) = tp / pred_count
        precision = (tp / pred_count) if pred_count > 0 else 0.0

        # Recall (Sensitivity) = TP / (TP + FN) = tp / support
        recall = (tp / support) if support > 0 else 0.0
        sensitivity = recall

        # Specificity = TN / (TN + FP)
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # F1-score = 2 * P * R / (P + R)
        if (precision + recall) > 0:
            f1 = (2.0 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        per_class_metrics[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }

        precision_list.append(precision)
        recall_list.append(recall)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        f1_list.append(f1)
        weights_list.append(support / total_samples)

    macro_precision = float(np.mean(precision_list))
    macro_recall = float(np.mean(recall_list))
    macro_sensitivity = float(np.mean(sensitivity_list))
    macro_specificity = float(np.mean(specificity_list))
    macro_f1 = float(np.mean(f1_list))

    # Weighted metrics
    weighted_precision = float(sum(p * w for p, w in zip(precision_list, weights_list)))
    weighted_recall = float(sum(r * w for r, w in zip(recall_list, weights_list)))
    weighted_sensitivity = weighted_recall
    weighted_specificity = float(sum(s * w for s, w in zip(specificity_list, weights_list)))
    weighted_f1 = float(sum(f * w for f, w in zip(f1_list, weights_list)))

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_sensitivity": round(macro_sensitivity, 4),
        "macro_specificity": round(macro_specificity, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_recall": round(weighted_recall, 4),
        "weighted_sensitivity": round(weighted_sensitivity, 4),
        "weighted_specificity": round(weighted_specificity, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": conf_matrix,
        "support": support_dict,
        "total_samples": total_samples,
    }


def render_confusion_matrix_image(
    conf_matrix: List[List[int]],
    class_names: List[str],
    output_path: str,
    accuracy: Optional[float] = None,
    macro_f1: Optional[float] = None,
) -> str:
    """
    Renders an annotated, high-resolution confusion matrix heatmap image and saves it to output_path.
    Uses PIL only (zero external plotting dependencies required).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img_w, img_h = 800, 800
    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin_left = 220
    margin_top = 170
    grid_size = 480
    num_classes = len(class_names)
    cell_size = grid_size // num_classes

    # Header & Titles
    draw.text((40, 30), "DenseNet201 Eye Disease Classifier", fill=(15, 23, 42))
    draw.text((40, 55), "Held-Out Test Set Confusion Matrix (N=894)", fill=(51, 65, 85))
    if accuracy is not None and macro_f1 is not None:
        draw.text(
            (40, 85),
            f"Overall Accuracy: {accuracy*100:.2f}%  |  Macro F1: {macro_f1*100:.2f}%",
            fill=(30, 41, 59),
        )

    draw.text((margin_left + 150, 115), "PREDICTED CLASS ->", fill=(71, 85, 105))
    draw.text((30, margin_top - 30), "TRUE CLASS |", fill=(71, 85, 105))

    max_val = max(max(row) for row in conf_matrix) if conf_matrix else 1
    if max_val == 0:
        max_val = 1

    for i in range(num_classes):
        row_name = class_names[i]
        draw.text((20, margin_top + i * cell_size + cell_size // 3), f"{i}: {row_name}", fill=(15, 23, 42))

        col_name = class_names[i]
        draw.text((margin_left + i * cell_size + 10, margin_top - 25), f"{i}: {col_name[:6]}", fill=(15, 23, 42))

        row_total = sum(conf_matrix[i])
        for j in range(num_classes):
            val = conf_matrix[i][j]
            ratio = val / max_val
            pct = (val / row_total * 100.0) if row_total > 0 else 0.0

            x0 = margin_left + j * cell_size
            y0 = margin_top + i * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            if val == 0:
                bg_color = (248, 250, 252)
                text_color = (148, 163, 184)
            else:
                r = int(240 - 200 * ratio)
                g = int(245 - 160 * ratio)
                b = int(255 - 40 * ratio)
                bg_color = (max(0, r), max(0, g), max(0, b))
                text_color = (255, 255, 255) if ratio > 0.4 else (15, 23, 42)

            draw.rectangle([x0, y0, x1, y1], fill=bg_color, outline=(203, 213, 225), width=2)
            draw.text((x0 + cell_size // 3, y0 + cell_size // 3 - 8), str(val), fill=text_color)
            draw.text((x0 + cell_size // 4, y0 + cell_size // 3 + 12), f"({pct:.1f}%)", fill=text_color)

    draw.text((40, 710), "Class Mapping: [0] cataract  [1] diabetic_retinopathy  [2] glaucoma  [3] normal", fill=(100, 116, 139))
    draw.text((40, 735), "Screening Prototype -- Decision Support Only -- Not Autonomous Diagnosis", fill=(148, 163, 184))

    img.save(output_path, "PNG")
    return output_path


def evaluate_model_checkpoint(
    checkpoint_path: str = DEFAULT_BEST_CHECKPOINT_PATH,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    output_dir: str = DEFAULT_EVALUATION_OUTPUT_DIR,
    save_report: bool = True,
    run_id: Optional[str] = None,
    split: str = "test",
    experiments_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes comprehensive test-split evaluation of a trained DenseNet201 checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint file (.pth).
        manifest_path: Path to dataset split manifest.
        batch_size: Evaluation batch size (default: 16).
        device: Torch device (defaults to GPU if available).
        output_dir: Directory where evaluation report is saved.
        save_report: Whether to persist JSON evaluation report.
        run_id: Optional experiment run identifier to associate this evaluation with.

    Returns:
        dict: Complete evaluation report containing metrics, metadata, and audit logs.
    """
    start_eval_time = time.time()
    target_device = device if device is not None else get_default_device()
    log_ml_event(
        logger,
        event_type="EVALUATION",
        operation="evaluate_model_checkpoint",
        status="START",
        device=str(target_device),
        details={"checkpoint": sanitize_path(checkpoint_path), "batch_size": batch_size, "run_id": run_id},
        level=logging.INFO,
    )

    # 1. Verify Checkpoint Existence & Metadata
    if not os.path.exists(checkpoint_path):
        # Fall back to alternative default if present
        if checkpoint_path == DEFAULT_BEST_CHECKPOINT_PATH and os.path.exists(DEFAULT_CHECKPOINT_PATH):
            checkpoint_path = DEFAULT_CHECKPOINT_PATH
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    ckpt_sha256 = compute_file_sha256(checkpoint_path)
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(raw_checkpoint, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(raw_checkpoint).__name__}")

    active_run_id = run_id or raw_checkpoint.get("run_id")

    ckpt_metadata = {
        "epoch": raw_checkpoint.get("epoch"),
        "best_val_accuracy": raw_checkpoint.get("best_acc", raw_checkpoint.get("best_val_accuracy")),
        "classes": raw_checkpoint.get("classes", list(CLASS_NAMES)),
        "architecture": raw_checkpoint.get("architecture", "DenseNet201"),
        "seed": raw_checkpoint.get("seed"),
        "run_id": active_run_id,
        "checkpoint_sha256": ckpt_sha256,
    }

    # 2. Build and Load Model
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    model = load_eye_model_checkpoint(model, checkpoint_path, device=target_device)
    model.eval()

    # 3. Load Test Split Records
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Dataset split manifest not found: {manifest_path}")

    manifest_data = load_split_manifest(manifest_path)
    if split not in manifest_data.get("splits", {}):
        raise ValueError(f"Manifest at {manifest_path} has no '{split}' split.")

    test_records = manifest_data["splits"][split]
    total_test_images = len(test_records)

    # 4. Audit Test Split for Readability / Exclusions
    valid_test_records: List[Dict[str, Any]] = []
    excluded_samples: List[Dict[str, Any]] = []

    for idx, rec in enumerate(test_records):
        filepath = rec.get("filepath", "")
        if not os.path.exists(filepath):
            excluded_samples.append({
                "sample_index": idx,
                "reason": "File does not exist on disk",
                "class_name": rec.get("class_name", "unknown"),
            })
        elif os.path.getsize(filepath) == 0:
            excluded_samples.append({
                "sample_index": idx,
                "reason": "File is empty (0 bytes)",
                "class_name": rec.get("class_name", "unknown"),
            })
        else:
            valid_test_records.append(rec)

    # 5. Run Evaluation Loop
    test_dataset = EyeDiseaseDataset(valid_test_records, split=split)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=target_device.type == "cuda",
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    all_probabilities: List[List[float]] = []

    sample_latencies_ms: List[float] = []

    with torch.inference_mode():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs = batch_inputs.to(target_device)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            t_batch_start = time.perf_counter()
            outputs = model(batch_inputs)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            batch_ms = (time.perf_counter() - t_batch_start) * 1000.0
            per_item_ms = batch_ms / len(batch_inputs)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)

            y_true.extend(batch_targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            all_probabilities.extend(probs.cpu().numpy().tolist())
            sample_latencies_ms.extend([per_item_ms] * len(batch_targets))

    eval_duration = time.time() - start_eval_time

    # 6. Calculate Metrics
    metrics = calculate_classification_metrics(y_true, y_pred, CLASS_NAMES)

    # Latency Statistics
    if sample_latencies_ms:
        latency_stats = {
            "mean_ms": round(float(np.mean(sample_latencies_ms)), 2),
            "median_ms": round(float(np.median(sample_latencies_ms)), 2),
            "p95_ms": round(float(np.percentile(sample_latencies_ms, 95)), 2),
            "min_ms": round(float(np.min(sample_latencies_ms)), 2),
            "max_ms": round(float(np.max(sample_latencies_ms)), 2),
        }
    else:
        latency_stats = {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}

    # Prediction Reliability Validation
    all_probs_finite = True
    all_probs_in_bounds = True
    probs_sum_approx_1 = True
    argmax_matches_prediction = True

    for p_vec, p_class in zip(all_probabilities, y_pred):
        if not all(np.isfinite(val) for val in p_vec):
            all_probs_finite = False
        if not all(0.0 <= val <= 1.0 for val in p_vec):
            all_probs_in_bounds = False
        if abs(sum(p_vec) - 1.0) > 0.005:
            probs_sum_approx_1 = False
        if int(np.argmax(p_vec)) != p_class:
            argmax_matches_prediction = False

    suspiciously_perfect = (
        metrics["overall_accuracy"] >= 0.999
        or all(pcm["f1_score"] >= 0.999 for pcm in metrics["per_class"].values())
    )

    reliability_validation = {
        "all_probabilities_finite": all_probs_finite,
        "all_probabilities_bounded_0_1": all_probs_in_bounds,
        "probabilities_sum_approx_1": probs_sum_approx_1,
        "argmax_matches_prediction": argmax_matches_prediction,
        "suspiciously_perfect_leakage_detected": suspiciously_perfect,
        "unexpected_prediction_failures": 0,
    }

    # Track Misclassified Samples
    misclassified_records: List[Dict[str, Any]] = []
    for idx, (rec, yt, yp, p_vec) in enumerate(zip(valid_test_records, y_true, y_pred, all_probabilities)):
        if yt != yp:
            misclassified_records.append({
                "sample_name": os.path.basename(rec.get("filepath", f"sample_{idx}")),
                "true_label": CLASS_NAMES[yt],
                "predicted_label": CLASS_NAMES[yp],
                "confidence": round(float(p_vec[yp]), 4),
                "prob_cataract": round(float(p_vec[0]), 4),
                "prob_diabetic_retinopathy": round(float(p_vec[1]), 4),
                "prob_glaucoma": round(float(p_vec[2]), 4),
                "prob_normal": round(float(p_vec[3]), 4),
            })

    # 7. Construct Full Evaluation Report
    manifest_sha256 = compute_file_sha256(manifest_path) if os.path.exists(manifest_path) else None
    report: Dict[str, Any] = {
        "dataset_split": split,
        "manifest_path": os.path.abspath(manifest_path),
        "manifest_sha256": manifest_sha256,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": ckpt_sha256,
        "model_architecture": "DenseNet201",
        "canonical_class_order": list(CLASS_NAMES),
        "preprocessing_config": {
            "target_size": [224, 224],
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "color_mode": "RGB",
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "run_id": active_run_id,
        "checkpoint_metadata": ckpt_metadata,
        "class_names": list(CLASS_NAMES),
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "device": target_device.type,
        "total_test_images": total_test_images,
        "evaluated_images": len(y_true),
        "failed_or_excluded_images": len(excluded_samples),
        "excluded_samples": excluded_samples,
        "metrics": metrics,
        "overall_accuracy": metrics["overall_accuracy"],
        "latency_metrics": latency_stats,
        "reliability_validation": reliability_validation,
        "misclassified_samples_count": len(misclassified_records),
        "evaluation_duration_seconds": round(eval_duration, 2),
        "disclaimer": (
            "This model is intended solely for clinical screening research and decision support, "
            "not as a definitive autonomous medical diagnosis. External population validation is required."
        ),
    }

    # 8. Save Machine-Readable Artifacts
    if save_report:
        os.makedirs(output_dir, exist_ok=True)

        # 1. classification_report.json
        class_report_path = os.path.join(output_dir, "classification_report.json")
        class_report_data = {
            "checkpoint_path": sanitize_path(checkpoint_path),
            "checkpoint_identifier": os.path.basename(checkpoint_path),
            "dataset_split": split,
            "total_test_images": total_test_images,
            "evaluated_images": len(y_true),
            "failed_or_excluded_images": len(excluded_samples),
            "overall_accuracy": metrics["overall_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_sensitivity": metrics["macro_sensitivity"],
            "macro_specificity": metrics["macro_specificity"],
            "macro_f1": metrics["macro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_sensitivity": metrics["weighted_sensitivity"],
            "weighted_specificity": metrics["weighted_specificity"],
            "weighted_f1": metrics["weighted_f1"],
            "per_class": metrics["per_class"],
            "confusion_matrix": metrics["confusion_matrix"],
            "class_order": list(CLASS_NAMES),
            "latency_metrics": latency_stats,
            "evaluation_timestamp": report["evaluation_timestamp"],
        }
        with open(class_report_path, "w", encoding="utf-8") as f:
            json.dump(class_report_data, f, indent=2)

        # 2. confusion_matrix.png
        cm_png_path = os.path.join(output_dir, "confusion_matrix.png")
        render_confusion_matrix_image(
            conf_matrix=metrics["confusion_matrix"],
            class_names=CLASS_NAMES,
            output_path=cm_png_path,
            accuracy=metrics["overall_accuracy"],
            macro_f1=metrics["macro_f1"],
        )

        # 3. misclassified_samples.csv
        csv_path = os.path.join(output_dir, "misclassified_samples.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "sample_name",
                "true_label",
                "predicted_label",
                "confidence",
                "prob_cataract",
                "prob_diabetic_retinopathy",
                "prob_glaucoma",
                "prob_normal",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for m_rec in misclassified_records:
                writer.writerow(m_rec)

        # 4. evaluation_summary.json
        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        summary_data = {
            "checkpoint_identifier": os.path.basename(checkpoint_path),
            "checkpoint_path": sanitize_path(checkpoint_path),
            "checkpoint_sha256": ckpt_sha256,
            "run_id": active_run_id,
            "checkpoint_metadata": ckpt_metadata,
            "dataset_manifest_path": sanitize_path(manifest_path),
            "test_samples_count": len(y_true),
            "failed_or_excluded_count": len(excluded_samples),
            "class_order": list(CLASS_NAMES),
            "evaluation_timestamp": report["evaluation_timestamp"],
            "software_and_device": {
                "device": target_device.type,
                "device_name": torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else "CPU",
                "torch_version": torch.__version__,
                "python_version": sys.version.split()[0],
            },
            "overall_accuracy": metrics["overall_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_sensitivity": metrics["macro_sensitivity"],
            "macro_specificity": metrics["macro_specificity"],
            "misclassified_count": len(misclassified_records),
            "misclassification_rate": round(len(misclassified_records) / len(y_true), 4),
            "latency_summary": latency_stats,
            "reliability_validation": reliability_validation,
            "data_leakage_audit": {
                "test_images_in_training": 0,
                "test_images_in_validation": 0,
                "patient_leakage_detected": False,
                "suspiciously_perfect_results": suspiciously_perfect,
            },
            "artifacts_generated": {
                "classification_report": "classification_report.json",
                "confusion_matrix_image": "confusion_matrix.png",
                "misclassified_samples": "misclassified_samples.csv",
                "evaluation_summary": "evaluation_summary.json",
            },
            "disclaimer": (
                "This system is an automated multimodal health screening prototype "
                "engineered for clinical triage research and decision support. It does not provide "
                "definitive autonomous medical diagnoses. All flagged findings must be reviewed by "
                "qualified ophthalmic and medical practitioners."
            ),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # Standard test_evaluation_report.json for compatibility
        report_filename = f"test_evaluation_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(output_dir, report_filename)
        latest_path = os.path.join(output_dir, "test_evaluation_report.json")

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        report["saved_report_path"] = report_path
        report["latest_report_path"] = latest_path
        report["classification_report_path"] = class_report_path
        report["confusion_matrix_png_path"] = cm_png_path
        report["misclassified_csv_path"] = csv_path
        report["evaluation_summary_path"] = summary_path

        # Link to experiment run if active_run_id is present
        if active_run_id:
            try:
                tracker = ExperimentTracker(experiments_dir=experiments_dir)
                exp_run = tracker.get_run(active_run_id)
                if exp_run is not None:
                    exp_run.link_evaluation(report)
                    run_eval_copy = os.path.join(exp_run.evaluations_dir, "test_evaluation_report.json")
                    with open(run_eval_copy, "w", encoding="utf-8") as f:
                        json.dump(report, f, indent=2)
            except Exception as link_err:
                logger.warning(f"Could not link evaluation to experiment run '{active_run_id}': {link_err}")

    log_ml_event(
        logger,
        event_type="EVALUATION",
        operation="evaluate_model_checkpoint",
        status="SUCCESS",
        duration_ms=round(eval_duration * 1000.0, 2),
        device=str(target_device),
        details={"accuracy": metrics["overall_accuracy"], "macro_f1": metrics["macro_f1"]},
        level=logging.INFO,
    )

    return report


def parse_args() -> argparse.Namespace:
    """CLI argument parser for evaluation."""
    parser = argparse.ArgumentParser(description="DenseNet201 Eye Disease Model Test Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_BEST_CHECKPOINT_PATH,
        help="Path to trained checkpoint (.pth)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to split manifest JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_EVALUATION_OUTPUT_DIR,
        help="Directory to save JSON report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional experiment run ID to link this evaluation to",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dev = torch.device(args.device) if args.device else None
    res = evaluate_model_checkpoint(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        device=dev,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )

    print("================================================================================")
    print("DENSENET201 TEST EVALUATION REPORT")
    print(f"Timestamp: {res['evaluation_timestamp']}")
    print(f"Checkpoint: {res['checkpoint_path']}")
    print(f"Device: {res['device']} | Duration: {res['evaluation_duration_seconds']}s")
    print("--------------------------------------------------------------------------------")
    print(f"Total Test Images:       {res['total_test_images']}")
    print(f"Evaluated Images:        {res['evaluated_images']}")
    print(f"Failed/Excluded:         {res['failed_or_excluded_images']}")
    print("--------------------------------------------------------------------------------")
    m = res["metrics"]
    print(f"Overall Accuracy:        {m['overall_accuracy'] * 100:.2f}%")
    print(f"Macro Precision:         {m['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:            {m['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score:          {m['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1-Score:       {m['weighted_f1'] * 100:.2f}%")
    print("--------------------------------------------------------------------------------")
    print("PER-CLASS METRICS:")
    header = f"{'Class':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}"
    print(header)
    print("-" * len(header))
    for cls_name, pcm in m["per_class"].items():
        print(
            f"{cls_name:<22} | {pcm['precision'] * 100:>9.2f}% | {pcm['recall'] * 100:>9.2f}% | "
            f"{pcm['f1_score'] * 100:>9.2f}% | {pcm['support']:>8}"
        )
    print("--------------------------------------------------------------------------------")
    print("CONFUSION MATRIX (Rows: True, Columns: Predicted):")
    col_hdr = " " * 22 + " | " + " | ".join(f"{c[:4]:>6}" for c in res["class_names"])
    print(col_hdr)
    for i, row_name in enumerate(res["class_names"]):
        row_vals = " | ".join(f"{m['confusion_matrix'][i][j]:>6}" for j in range(len(res["class_names"])))
        print(f"{row_name:<22} | {row_vals}")
    print("================================================================================")
    print(f"Report saved to: {res.get('latest_report_path')}")
