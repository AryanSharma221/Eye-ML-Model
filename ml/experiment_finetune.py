"""
Controlled Model Improvement Experiment Pipeline (Part 15).

Investigates and addresses the evidence-based weakness identified in baseline evaluation:
- Glaucoma has the lowest F1 (76.66%) and precision/recall (76.47% / 76.85%), with 27 cases misclassified as Normal.
- Cataract has low recall (79.05%), with 20 cases misclassified as Glaucoma.

Controlled Intervention:
1. Difficulty-compensated class weighting: [cataract: 1.15, DR: 0.90, glaucoma: 1.25, normal: 0.90] (mean-normalized).
2. Label smoothing (epsilon = 0.05) to reduce overconfident classification on ambiguous optic disc boundaries.
3. Fine-tuning learning rate: lr = 1e-4 (5x lower than initial training) using Adam with weight decay 1e-4.
4. Model selection: Strictly determined by validation set performance (865 images).
5. Held-out test evaluation: Evaluated on test split (894 images) only after model selection is frozen.
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ml.dataset_loader import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    NUM_CLASSES,
    EyeDiseaseDataset,
)
from ml.eye_model import (
    DEFAULT_CHECKPOINT_PATH,
    build_densenet201_model,
    clear_model_cache,
    get_cached_model,
    get_default_device,
    load_eye_model_checkpoint,
)
from ml.train_eye import (
    DEFAULT_MANIFEST_PATH,
    evaluate_dataset,
    save_atomic_checkpoint,
    set_seed,
)


def compute_macro_f1_from_cm(cm: List[List[int]]) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Computes precision, recall, F1 per class and macro F1 from confusion matrix."""
    per_class = {}
    num_classes = len(CLASS_NAMES)
    f1s = []

    for i, c_name in enumerate(CLASS_NAMES):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(num_classes) if r != i)
        fn = sum(cm[i][c] for c in range(num_classes) if c != i)
        support = sum(cm[i])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[c_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }
        f1s.append(f1)

    macro_f1 = round(float(np.mean(f1s)), 4) if f1s else 0.0
    return macro_f1, per_class


def run_controlled_improvement_experiment(
    baseline_checkpoint_path: str = os.path.join("ml", "weights", "baseline_eye_model_backup.pth"),
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    output_checkpoint_path: str = os.path.join("ml", "weights", "eye_model_experiment.pth"),
    production_checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 1e-4,
    label_smoothing: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes a controlled fine-tuning experiment to improve Glaucoma and Cataract recognition.
    """
    set_seed(seed)
    device = get_default_device()
    print("=" * 80)
    print(" CONTROLLED MODEL IMPROVEMENT EXPERIMENT (PART 15)")
    print("=" * 80)
    print(f"Device:              {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Baseline Checkpoint: {baseline_checkpoint_path}")
    print(f"Manifest Path:       {manifest_path}")
    print(f"Fine-Tuning Epochs:  {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print(f"Label Smoothing:     {label_smoothing} | Seed: {seed}")
    print("-" * 80)

    # 1. Dataloaders
    train_ds = EyeDiseaseDataset(manifest_path, split="train")
    val_ds = EyeDiseaseDataset(manifest_path, split="val")
    test_ds = EyeDiseaseDataset(manifest_path, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 2. Evaluate Baseline Model on Validation Set
    baseline_model = build_densenet201_model(num_classes=NUM_CLASSES)
    baseline_model = load_eye_model_checkpoint(baseline_model, baseline_checkpoint_path, device=device)
    baseline_criterion = nn.CrossEntropyLoss()

    print("Evaluating baseline model on Validation Split (865 images)...")
    base_val_eval = evaluate_dataset(baseline_model, val_loader, baseline_criterion, device, CLASS_NAMES)
    base_val_macro_f1, base_val_per_class = compute_macro_f1_from_cm(base_val_eval["confusion_matrix"])

    baseline_metrics = {
        "val_loss": round(base_val_eval["loss"], 4),
        "val_accuracy": round(base_val_eval["accuracy"], 4),
        "val_macro_f1": base_val_macro_f1,
        "val_per_class": base_val_per_class,
        "val_confusion_matrix": base_val_eval["confusion_matrix"],
    }
    print(f"Baseline Validation Acc:      {baseline_metrics['val_accuracy'] * 100:.2f}%")
    print(f"Baseline Validation Macro F1: {baseline_metrics['val_macro_f1'] * 100:.2f}%")
    print(f"Baseline Per-Class F1 (Val):  { {c: v['f1_score'] for c, v in base_val_per_class.items()} }")

    # 3. Setup Fine-Tuning Model with Difficulty-Compensated Loss
    # Classes: ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
    # Target difficulty: boost Glaucoma (1.25) and Cataract (1.15), lower DR (0.90) and Normal (0.90)
    raw_weights = torch.tensor([1.15, 0.90, 1.25, 0.90], dtype=torch.float32)
    normalized_weights = (raw_weights / raw_weights.mean()).to(device)
    print(f"\nDifficulty-Compensated Weights: { {c: round(w.item(), 3) for c, w in zip(CLASS_NAMES, normalized_weights)} }")

    criterion = nn.CrossEntropyLoss(weight=normalized_weights, label_smoothing=label_smoothing)

    # Clone weights from baseline model for fine-tuning
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    model.load_state_dict(baseline_model.state_dict())
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 4. Fine-Tuning Execution
    best_candidate_acc = baseline_metrics["val_accuracy"]
    best_candidate_macro_f1 = baseline_metrics["val_macro_f1"]
    best_candidate_state = None
    experiment_history: List[Dict[str, Any]] = []

    for ep in range(epochs):
        ep_start = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == targets).item()
            total_train += inputs.size(0)

        ep_train_loss = train_loss / total_train
        ep_train_acc = train_correct / total_train

        # Validation Step
        val_eval = evaluate_dataset(model, val_loader, criterion, device, CLASS_NAMES)
        val_macro_f1, val_per_class = compute_macro_f1_from_cm(val_eval["confusion_matrix"])
        val_acc = val_eval["accuracy"]
        val_loss = val_eval["loss"]

        ep_time = time.time() - ep_start
        print(
            f"Epoch {ep + 1}/{epochs} ({ep_time:.1f}s) | "
            f"Train Loss: {ep_train_loss:.4f} Acc: {ep_train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}% | "
            f"Val Macro F1: {val_macro_f1 * 100:.2f}%"
        )

        ep_record = {
            "epoch": ep + 1,
            "train_loss": round(ep_train_loss, 4),
            "train_acc": round(ep_train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": val_macro_f1,
            "val_per_class": val_per_class,
        }
        experiment_history.append(ep_record)

        # Track best model strictly on validation data
        if val_acc >= best_candidate_acc or (val_acc >= 0.875 and val_macro_f1 > best_candidate_macro_f1):
            best_candidate_acc = val_acc
            best_candidate_macro_f1 = val_macro_f1
            best_candidate_state = {
                "state_dict": model.state_dict(),
                "epoch": ep + 5,
                "best_acc": val_acc,
                "best_val_accuracy": val_acc,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "classes": list(CLASS_NAMES),
                "architecture": "DenseNet201",
                "seed": seed,
            }

    # 5. Model Selection Decision
    improved_on_validation = False
    if best_candidate_state is not None and (
        best_candidate_acc > baseline_metrics["val_accuracy"]
        or (best_candidate_acc >= 0.880 and best_candidate_macro_f1 > baseline_metrics["val_macro_f1"])
    ):
        improved_on_validation = True
        print(f"\n[DECISION] Fine-tuned model improved on validation data! (Acc: {best_candidate_acc*100:.2f}%, F1: {best_candidate_macro_f1*100:.2f}%)")
        save_atomic_checkpoint(best_candidate_state, output_checkpoint_path)
        # Promote to production checkpoint
        save_atomic_checkpoint(best_candidate_state, production_checkpoint_path)
        selected_model_path = production_checkpoint_path
    else:
        print(f"\n[DECISION] Baseline model remains champion on validation data (Baseline: {baseline_metrics['val_accuracy']*100:.2f}%). Preserving baseline checkpoint.")
        selected_model_path = baseline_checkpoint_path

    # 6. Final Evaluation on Held-Out Test Set (894 images)
    clear_model_cache()
    final_eval_model = build_densenet201_model(num_classes=NUM_CLASSES)
    final_eval_model = load_eye_model_checkpoint(final_eval_model, selected_model_path, device=device)
    final_eval_model.eval()

    test_criterion = nn.CrossEntropyLoss()
    test_eval = evaluate_dataset(final_eval_model, test_loader, test_criterion, device, CLASS_NAMES)
    test_macro_f1, test_per_class = compute_macro_f1_from_cm(test_eval["confusion_matrix"])

    test_metrics = {
        "test_loss": round(test_eval["loss"], 4),
        "test_accuracy": round(test_eval["accuracy"], 4),
        "test_macro_f1": test_macro_f1,
        "test_per_class": test_per_class,
        "test_confusion_matrix": test_eval["confusion_matrix"],
    }

    print("\n--- FINAL HELD-OUT TEST EVALUATION (894 IMAGES) ---")
    print(f"Overall Test Accuracy:        {test_metrics['test_accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:               {test_metrics['test_macro_f1'] * 100:.2f}%")
    for c_name, m_dict in test_per_class.items():
        print(f"  {c_name:<21}: Precision={m_dict['precision']*100:5.2f}% | Recall={m_dict['recall']*100:5.2f}% | F1={m_dict['f1_score']*100:5.2f}%")

    report: Dict[str, Any] = {
        "experiment_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": str(device),
            "seed": seed,
            "fine_tuning_epochs": epochs,
            "learning_rate": lr,
            "label_smoothing": label_smoothing,
            "difficulty_weights": {c: round(w.item(), 3) for c, w in zip(CLASS_NAMES, normalized_weights)},
            "baseline_checkpoint": baseline_checkpoint_path,
            "selected_checkpoint": selected_model_path,
        },
        "baseline_validation_metrics": baseline_metrics,
        "experiment_training_history": experiment_history,
        "model_selection": {
            "improved_on_validation": improved_on_validation,
            "selected_model": "experiment_candidate" if improved_on_validation else "baseline_champion",
            "validation_decision_criterion": "Validation Accuracy and Macro F1",
        },
        "final_held_out_test_metrics": test_metrics,
    }

    report_path = os.path.join("ml", "evaluation_results", "experiment_improvement_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Experiment report persisted to: {report_path}")
    print("=" * 80)

    return report


if __name__ == "__main__":
    run_controlled_improvement_experiment()
