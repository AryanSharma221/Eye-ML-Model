"""
Audit Reconciliation Script for CPU vs CUDA and Repeatability Verification.

Evaluates all 894 held-out test samples under:
1. CUDA Pass 1
2. CUDA Pass 2
3. CPU Pass

Computes exact:
- Sample count (verified 894)
- Prediction agreement count & percentage
- Maximum absolute probability difference
- Mean absolute probability difference
- Bit-for-bit array equality

Saves verified results to ml/evaluation_results/reconciled_reproducibility_execution.json.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.config import CLASS_NAMES, NUM_CLASSES, get_default_checkpoint_path
from ml.dataset_loader import EyeDiseaseDataset, load_split_manifest
from ml.experiment_tracker import compute_file_sha256
from ml.eye_model import build_densenet201_model, load_eye_model_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reconcile_reproducibility")


def evaluate_split(
    checkpoint_path: str,
    manifest_path: str,
    device_name: str,
    batch_size: int = 32,
) -> Tuple[List[int], np.ndarray, List[int], float]:
    """
    Evaluates the full test split on the specified device.
    Returns: (predictions, probabilities_array, true_labels, elapsed_seconds)
    """
    device = torch.device(device_name)
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    model = load_eye_model_checkpoint(model, checkpoint_path, device=device)
    model.eval()

    manifest_data = load_split_manifest(manifest_path)
    test_records = manifest_data["splits"]["test"]

    dataset = EyeDiseaseDataset(test_records, split="test")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    all_targets: List[int] = []

    t0 = time.perf_counter()
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())
            all_targets.extend(targets.numpy().tolist())

            if device.type == "cuda":
                torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - t0
    probs_matrix = np.vstack(all_probs)
    return all_preds, probs_matrix, all_targets, elapsed


def main():
    checkpoint_path = get_default_checkpoint_path()
    manifest_path = os.path.join(_SCRIPT_DIR, "dataset", "split_manifest.json")
    output_path = os.path.join(_SCRIPT_DIR, "evaluation_results", "reconciled_reproducibility_execution.json")

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Manifest: {manifest_path}")

    ckpt_sha = compute_file_sha256(checkpoint_path)
    manifest_sha = compute_file_sha256(manifest_path)

    # 1. CUDA Run 1
    logger.info("Starting CUDA Pass 1 on all test samples...")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    preds_cuda_1, probs_cuda_1, targets, t_cuda_1 = evaluate_split(
        checkpoint_path, manifest_path, "cuda", batch_size=32
    )
    n_samples = len(preds_cuda_1)
    logger.info(f"CUDA Pass 1 completed in {t_cuda_1:.2f}s on {n_samples} samples.")

    # 2. CUDA Run 2
    logger.info("Starting CUDA Pass 2 on all test samples...")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    preds_cuda_2, probs_cuda_2, _, t_cuda_2 = evaluate_split(
        checkpoint_path, manifest_path, "cuda", batch_size=32
    )
    logger.info(f"CUDA Pass 2 completed in {t_cuda_2:.2f}s on {len(preds_cuda_2)} samples.")

    # 3. CPU Run
    logger.info("Starting CPU Pass on all test samples...")
    torch.manual_seed(42)
    preds_cpu, probs_cpu, _, t_cpu = evaluate_split(
        checkpoint_path, manifest_path, "cpu", batch_size=32
    )
    logger.info(f"CPU Pass completed in {t_cpu:.2f}s on {len(preds_cpu)} samples.")

    # Comparisons
    # A. CUDA Repeatability (Pass 1 vs Pass 2)
    diff_cuda_cuda = np.abs(probs_cuda_1 - probs_cuda_2)
    max_diff_cuda_cuda = float(np.max(diff_cuda_cuda))
    mean_diff_cuda_cuda = float(np.mean(diff_cuda_cuda))
    agreement_cuda_cuda = int(np.sum(np.array(preds_cuda_1) == np.array(preds_cuda_2)))
    pct_cuda_cuda = (agreement_cuda_cuda / n_samples) * 100.0
    bit_identical_cuda = bool(np.array_equal(probs_cuda_1, probs_cuda_2))

    # B. CPU vs CUDA Parity (Pass 1 vs CPU)
    diff_cpu_cuda = np.abs(probs_cuda_1 - probs_cpu)
    max_diff_cpu_cuda = float(np.max(diff_cpu_cuda))
    mean_diff_cpu_cuda = float(np.mean(diff_cpu_cuda))
    agreement_cpu_cuda = int(np.sum(np.array(preds_cuda_1) == np.array(preds_cpu)))
    pct_cpu_cuda = (agreement_cpu_cuda / n_samples) * 100.0
    bit_identical_cpu = bool(np.array_equal(probs_cuda_1, probs_cpu))

    # Accuracy check
    acc_cuda = float(np.mean(np.array(preds_cuda_1) == np.array(targets)))
    acc_cpu = float(np.mean(np.array(preds_cpu) == np.array(targets)))

    result = {
        "metadata": {
            "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint_sha256": ckpt_sha,
            "manifest_sha256": manifest_sha,
            "total_test_samples": n_samples,
            "target_classes": list(CLASS_NAMES),
        },
        "timings_seconds": {
            "cuda_pass_1": round(t_cuda_1, 2),
            "cuda_pass_2": round(t_cuda_2, 2),
            "cpu_pass": round(t_cpu, 2),
        },
        "accuracy": {
            "cuda_pass_1": round(acc_cuda, 4),
            "cuda_pass_2": round(float(np.mean(np.array(preds_cuda_2) == np.array(targets))), 4),
            "cpu_pass": round(acc_cpu, 4),
        },
        "repeatability_cuda_vs_cuda": {
            "sample_count": n_samples,
            "prediction_agreement_count": agreement_cuda_cuda,
            "prediction_agreement_pct": round(pct_cuda_cuda, 4),
            "bit_for_bit_identical": bit_identical_cuda,
            "max_absolute_probability_difference": max_diff_cuda_cuda,
            "mean_absolute_probability_difference": mean_diff_cuda_cuda,
        },
        "cross_device_cpu_vs_cuda": {
            "sample_count": n_samples,
            "prediction_agreement_count": agreement_cpu_cuda,
            "prediction_agreement_pct": round(pct_cpu_cuda, 4),
            "bit_for_bit_identical": bit_identical_cpu,
            "max_absolute_probability_difference": max_diff_cpu_cuda,
            "mean_absolute_probability_difference": mean_diff_cpu_cuda,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results successfully saved to {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
