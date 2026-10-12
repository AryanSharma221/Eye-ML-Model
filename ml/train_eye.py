"""
DenseNet201 Eye Disease Classifier Training Pipeline.

Trains the 4-class retinal disease model (Cataract, Diabetic Retinopathy, Glaucoma, Normal)
using the validated patient-stratified dataset split manifest and torchvision DenseNet201 backbone.

Public Contract:
    TrainConfig: dataclass configuration
    compute_class_weights(class_counts, class_names) -> torch.Tensor
    get_training_dataloaders(manifest_path, batch_size, ...) -> dict[str, DataLoader]
    build_training_model(...) -> nn.Module
    evaluate_dataset(model, dataloader, criterion, device, class_names) -> dict[str, Any]
    train_eye_model(config: Optional[TrainConfig] = None) -> dict[str, Any]
"""

import argparse
from dataclasses import asdict, dataclass, field
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone

# Ensure project root is on sys.path when invoked directly as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from ml.dataset_loader import (
    CLASS_NAMES,
    NUM_CLASSES,
    EyeDiseaseDataset,
)
from ml.eye_model import (
    DEFAULT_CHECKPOINT_PATH,
    DENSENET201_IN_FEATURES,
    get_default_device,
)
from ml.experiment_tracker import (
    ExperimentRun,
    ExperimentTracker,
    verify_checkpoint_integrity,
)
from ml.monitoring import log_ml_event, sanitize_path

logger = logging.getLogger(__name__)

# Standard local cache location for PyTorch pretrained weights
LOCAL_TORCH_HUB_WEIGHTS: str = os.path.join(
    os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "densenet201-c1103571.pth"
)
DEFAULT_MANIFEST_PATH: str = os.path.join(
    os.path.dirname(__file__), "dataset", "split_manifest.json"
)
DEFAULT_BEST_CHECKPOINT_PATH: str = os.path.join(
    os.path.dirname(__file__), "weights", "eye_densenet201_best.pth"
)
DEFAULT_RECOVERY_CHECKPOINT_PATH: str = os.path.join(
    os.path.dirname(__file__), "weights", "recovery_checkpoint.pth"
)


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_atomic_checkpoint(checkpoint_dict: Dict[str, Any], target_path: str) -> None:
    """Saves checkpoint safely to a temporary file before renaming to prevent corruption."""
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)
    tmp_path = f"{target_path}.tmp"
    torch.save(checkpoint_dict, tmp_path)
    try:
        os.replace(tmp_path, target_path)
    except OSError:
        if os.path.exists(target_path):
            os.remove(target_path)
        os.replace(tmp_path, target_path)


def compute_class_weights(
    class_counts: Dict[str, int],
    class_names: List[str] = CLASS_NAMES,
) -> torch.Tensor:
    """
    Computes inverse class frequency weights normalized so mean weight is 1.0.

    Formula:
        weight[c] = total_samples / (num_classes * count[c])

    Args:
        class_counts: Dictionary mapping class name to image count.
        class_names: Canonical class order.

    Returns:
        torch.Tensor of shape (num_classes,) with float32 class weights.
    """
    total = sum(class_counts.values())
    num_classes = len(class_names)
    weights = []
    for cls in class_names:
        count = class_counts.get(cls, 0)
        if count > 0:
            w = total / (num_classes * count)
        else:
            w = 1.0
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor


@dataclass
class TrainConfig:
    """Configuration dataclass for DenseNet201 eye disease training."""

    manifest_path: str = DEFAULT_MANIFEST_PATH
    output_checkpoint_path: str = DEFAULT_CHECKPOINT_PATH
    best_checkpoint_path: str = DEFAULT_BEST_CHECKPOINT_PATH
    recovery_checkpoint_path: str = DEFAULT_RECOVERY_CHECKPOINT_PATH
    num_classes: int = NUM_CLASSES
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))
    batch_size: int = 8  # Optimized for 8GB VRAM (RTX 4060 Laptop GPU)
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    use_class_weights: bool = True
    num_workers: int = 0  # 0 is safe on Windows to prevent multiprocessing spawn deadlocks
    device: Optional[torch.device] = None
    use_local_pretrained: bool = True
    pretrained_weights_path: Optional[str] = None
    resume: bool = False
    resume_from: Optional[str] = None
    seed: int = 42
    patience: int = 3
    eval_test_after_train: bool = True

    # Lightweight Experiment Tracking & Checkpoint Versioning
    track_experiment: bool = True
    experiment_name: str = "eye_densenet201_screening"
    run_id: Optional[str] = None
    experiments_dir: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["densenet201", "retinal_screening"])
    description: str = ""
    promote_to_production: bool = False


def get_training_dataloaders(
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Constructs PyTorch DataLoaders for train, val, and test splits from a split manifest.

    Args:
        manifest_path: Path to the JSON split manifest.
        batch_size: Number of images per mini-batch (default: 8).
        num_workers: Number of subprocess workers for data loading (default: 0).

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Split manifest file not found: {manifest_path}")

    train_ds = EyeDiseaseDataset(manifest_path, split="train")
    val_ds = EyeDiseaseDataset(manifest_path, split="val")
    test_ds = EyeDiseaseDataset(manifest_path, split="test")

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }


def build_training_model(
    num_classes: int = NUM_CLASSES,
    use_local_pretrained: bool = True,
    pretrained_weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Constructs DenseNet201 model for training, loading verified local ImageNet
    pretrained weights when available, without initiating network downloads.

    Args:
        num_classes: Number of output disease classes (default: 4).
        use_local_pretrained: Whether to load local ImageNet backbone weights if present.
        pretrained_weights_path: Explicit path to local pretrained weights file.
        device: Target device.

    Returns:
        nn.Module configured with 4-class output head.
    """
    target_device = device if device is not None else get_default_device()

    local_weights = pretrained_weights_path or LOCAL_TORCH_HUB_WEIGHTS
    has_local_weights = use_local_pretrained and os.path.exists(local_weights)

    if has_local_weights:
        # Load local ImageNet backbone without network access
        model = models.densenet201(weights=None)
        state_dict = torch.load(local_weights, map_location="cpu", weights_only=True)
        # Remap legacy keys (norm.1 -> norm1, conv.1 -> conv1) if present
        remapped_state_dict = {}
        for k, v in state_dict.items():
            new_k = re.sub(r"norm\.([12])", r"norm\1", k)
            new_k = re.sub(r"conv\.([12])", r"conv\1", new_k)
            remapped_state_dict[new_k] = v
        model.load_state_dict(remapped_state_dict, strict=True)
    else:
        # Instantiate without pretrained weights
        model = models.densenet201(weights=None)

    in_features = model.classifier.in_features
    if in_features != DENSENET201_IN_FEATURES:
        raise ValueError(f"Unexpected in_features: {in_features}, expected {DENSENET201_IN_FEATURES}.")

    # Replace classifier head with 4-class projection layer
    model.classifier = nn.Linear(in_features, num_classes)
    model = model.to(target_device)
    return model


def evaluate_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    """
    Evaluates a trained model across an entire dataset split, computing
    loss, overall accuracy, per-class accuracy, and confusion matrix.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    num_classes = len(class_names)
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, preds = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            correct += torch.sum(preds == targets).item()

            for t, p in zip(targets.view(-1), preds.view(-1)):
                t_idx = t.item()
                p_idx = p.item()
                confusion_matrix[t_idx][p_idx] += 1
                per_class_total[t_idx] += 1
                if t_idx == p_idx:
                    per_class_correct[t_idx] += 1

    avg_loss = total_loss / max(1, total_samples)
    accuracy = correct / max(1, total_samples)
    per_class_acc = {}
    for i, name in enumerate(class_names):
        tot = per_class_total[i]
        per_class_acc[name] = (per_class_correct[i] / tot) if tot > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion_matrix,
        "total_samples": total_samples,
    }


def train_eye_model(config: Optional[TrainConfig] = None) -> Dict[str, Any]:
    """
    Executes training loop across epochs, tracking train/val metrics,
    saving periodic recovery checkpoints and the best model checkpoint.

    Args:
        config: TrainConfig dataclass. Uses defaults if None.

    Returns:
        dict: Training history, best validation metrics, checkpoint paths, and test evaluation.
    """
    cfg = config if config is not None else TrainConfig()
    set_seed(cfg.seed)
    target_device = cfg.device if cfg.device is not None else get_default_device()

    dataloaders = get_training_dataloaders(
        manifest_path=cfg.manifest_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = build_training_model(
        num_classes=cfg.num_classes,
        use_local_pretrained=cfg.use_local_pretrained,
        pretrained_weights_path=cfg.pretrained_weights_path,
        device=target_device,
    )

    # Compute class weights from training distribution if requested
    if cfg.use_class_weights and hasattr(dataloaders["train"].dataset, "get_class_counts"):
        class_counts = dataloaders["train"].dataset.get_class_counts()
        weights = compute_class_weights(class_counts, cfg.class_names).to(target_device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=cfg.patience,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    start_epoch = 0
    best_acc = 0.0
    best_weights = None

    # Lightweight Experiment Tracking Setup
    exp_run: Optional[ExperimentRun] = None
    if cfg.track_experiment:
        tracker = ExperimentTracker(experiments_dir=cfg.experiments_dir)
        exp_run = tracker.create_run(
            experiment_name=cfg.experiment_name,
            run_id=cfg.run_id,
            description=cfg.description,
            tags=cfg.tags,
        )
        exp_run.start(
            config_dict=asdict(cfg),
            manifest_path=cfg.manifest_path,
            device=target_device,
            seed=cfg.seed,
        )

    # Resume from recovery checkpoint if requested and file exists
    resume_target = cfg.resume_from or cfg.recovery_checkpoint_path
    if cfg.resume and os.path.exists(resume_target):
        verification = verify_checkpoint_integrity(resume_target, num_classes=cfg.num_classes)
        if not verification.get("valid", False):
            raise ValueError(
                f"Cannot resume from checkpoint {resume_target}: integrity check failed: {verification.get('error')}"
            )

        recovery_ckpt = torch.load(
            resume_target, map_location=target_device, weights_only=True
        )
        if "state_dict" in recovery_ckpt and recovery_ckpt["state_dict"]:
            model.load_state_dict(recovery_ckpt["state_dict"])
        if recovery_ckpt.get("optimizer_state_dict"):
            try:
                optimizer.load_state_dict(recovery_ckpt["optimizer_state_dict"])
            except Exception as opt_err:
                logger.warning("Could not restore optimizer state from checkpoint: %s", opt_err)
        if recovery_ckpt.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(recovery_ckpt["scheduler_state_dict"])
            except Exception as sched_err:
                logger.warning("Could not restore scheduler state from checkpoint: %s", sched_err)
        start_epoch = recovery_ckpt.get("epoch", 0)
        best_acc = recovery_ckpt.get("best_acc", recovery_ckpt.get("best_val_accuracy", 0.0))
        if "history" in recovery_ckpt and isinstance(recovery_ckpt["history"], dict):
            history = recovery_ckpt["history"]
        print(
            f"[RESUME] Resumed from verified recovery checkpoint {resume_target} at epoch {start_epoch} "
            f"(best val acc: {best_acc * 100:.2f}%)"
        )
        if exp_run:
            exp_run.metadata["resumed_from"] = {
                "checkpoint_path": sanitize_path(resume_target),
                "sha256": verification["sha256"],
                "resumed_epoch": start_epoch,
            }
            exp_run._save_metadata_atomic()

    dataset_sizes = {
        "train": len(dataloaders["train"].dataset),
        "val": len(dataloaders["val"].dataset),
    }

    train_start_time = time.time()
    device_name = torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else "CPU"
    print(f"================================================================================")
    print(f"Starting DenseNet201 Training | Device: {device_name} | Seed: {cfg.seed}")
    print(f"Dataset: Train={dataset_sizes['train']} | Val={dataset_sizes['val']} | Batch Size={cfg.batch_size}")
    print(f"Epochs: {start_epoch + 1} to {cfg.num_epochs} | Initial LR: {cfg.learning_rate}")
    if exp_run:
        print(f"Experiment Run ID: {exp_run.run_id} | Artifacts Dir: {exp_run.run_dir}")
    print(f"================================================================================")
    log_ml_event(
        logger,
        event_type="TRAINING",
        operation="train_eye_model",
        status="START",
        device=str(target_device),
        details={"epochs": cfg.num_epochs, "batch_size": cfg.batch_size, "lr": cfg.learning_rate, "seed": cfg.seed},
    )

    try:
        for epoch in range(start_epoch, cfg.num_epochs):
            epoch_start_time = time.time()

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                running_samples = 0
                batch_count = 0
                total_batches = len(dataloaders[phase])

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(target_device)
                    labels = labels.to(target_device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()
                    running_samples += inputs.size(0)
                    batch_count += 1

                    # Periodic progress display for training batches
                    if phase == "train" and (batch_count % 100 == 0 or batch_count == total_batches):
                        cur_loss = running_loss / max(1, running_samples)
                        cur_acc = running_corrects / max(1, running_samples)
                        print(
                            f"  [Epoch {epoch + 1}/{cfg.num_epochs} | Train Batch {batch_count}/{total_batches}] "
                            f"Batch Loss: {loss.item():.4f} | Running Loss: {cur_loss:.4f} | Running Acc: {cur_acc * 100:.2f}%",
                            flush=True,
                        )

                epoch_loss = running_loss / max(1, dataset_sizes[phase])
                epoch_acc = running_corrects / max(1, dataset_sizes[phase])

                history[f"{phase}_loss"].append(epoch_loss)
                history[f"{phase}_acc"].append(epoch_acc)

                if phase == "val":
                    scheduler.step(epoch_acc)
                    current_lr = optimizer.param_groups[0]["lr"]
                    history["lr"].append(current_lr)

                    if epoch_acc > best_acc or best_weights is None:
                        if epoch_acc > best_acc:
                            best_acc = epoch_acc
                        best_weights = {
                            "state_dict": model.state_dict(),
                            "classes": cfg.class_names,
                            "best_acc": best_acc,
                            "best_val_accuracy": best_acc,
                            "epoch": epoch + 1,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "architecture": "DenseNet201",
                            "history": history,
                            "seed": cfg.seed,
                        }
                        if exp_run:
                            exp_run.save_run_checkpoint(
                                model=model,
                                epoch=epoch + 1,
                                is_best=True,
                                best_val_acc=best_acc,
                                val_loss=epoch_loss,
                                optimizer=optimizer,
                                scheduler=scheduler,
                            )

                        # Safe checkpoint versioning:
                        # Only overwrite production weights when explicit promotion is requested.
                        # Preserve explicit custom targets (e.g. temporary fixtures in tests).
                        if cfg.promote_to_production or cfg.output_checkpoint_path != DEFAULT_CHECKPOINT_PATH:
                            save_atomic_checkpoint(best_weights, cfg.output_checkpoint_path)
                        if cfg.promote_to_production or (cfg.best_checkpoint_path and cfg.best_checkpoint_path != DEFAULT_BEST_CHECKPOINT_PATH):
                            save_atomic_checkpoint(best_weights, cfg.best_checkpoint_path)

                        log_ml_event(
                            logger,
                            event_type="CHECKPOINT_SAVE",
                            operation="train_eye_model",
                            status="SUCCESS",
                            details={
                                "checkpoint": sanitize_path(cfg.output_checkpoint_path),
                                "epoch": epoch + 1,
                                "best_acc": best_acc,
                                "promoted_to_production": cfg.promote_to_production,
                            },
                        )

            epoch_duration = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]["lr"]
            vram_mb = (
                torch.cuda.memory_allocated(target_device) / (1024**2)
                if target_device.type == "cuda"
                else 0.0
            )

            print(
                f"Epoch {epoch + 1:2d}/{cfg.num_epochs} | "
                f"Train Loss: {history['train_loss'][-1]:.4f} Acc: {history['train_acc'][-1] * 100:5.2f}% | "
                f"Val Loss: {history['val_loss'][-1]:.4f} Acc: {history['val_acc'][-1] * 100:5.2f}% | "
                f"Best Val: {best_acc * 100:5.2f}% | LR: {current_lr:.6f} | "
                f"Time: {epoch_duration:5.1f}s | VRAM: {vram_mb:5.1f} MB",
                flush=True,
            )

            # Save recovery checkpoint after every epoch
            recovery_payload = {
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "classes": cfg.class_names,
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "best_val_accuracy": best_acc,
                "history": history,
                "seed": cfg.seed,
                "architecture": "DenseNet201",
            }

            if exp_run:
                exp_run.log_epoch(
                    epoch=epoch + 1,
                    train_loss=history["train_loss"][-1],
                    train_acc=history["train_acc"][-1],
                    val_loss=history["val_loss"][-1],
                    val_acc=history["val_acc"][-1],
                    lr=current_lr,
                    duration_seconds=epoch_duration,
                )
                exp_run.save_run_checkpoint(
                    model=model,
                    epoch=epoch + 1,
                    is_recovery=True,
                    best_val_acc=best_acc,
                    val_loss=history["val_loss"][-1],
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

            if cfg.recovery_checkpoint_path != DEFAULT_RECOVERY_CHECKPOINT_PATH or cfg.promote_to_production or not cfg.track_experiment:
                save_atomic_checkpoint(recovery_payload, cfg.recovery_checkpoint_path)

        # Save final checkpoint in run
        if exp_run:
            exp_run.save_run_checkpoint(
                model=model,
                epoch=cfg.num_epochs,
                is_final=True,
                best_val_acc=best_acc,
                val_loss=history["val_loss"][-1] if history["val_loss"] else None,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    except KeyboardInterrupt as ki:
        print("\n[INTERRUPTED] Training run interrupted by user signal.")
        if exp_run:
            exp_run.interrupt("Training interrupted by KeyboardInterrupt")
        raise ki
    except Exception as exc:
        print(f"\n[FAILED] Training run failed with exception: {exc}")
        if exp_run:
            exp_run.fail(str(exc), exception=exc)
        raise exc

    total_training_time = time.time() - train_start_time
    print(f"================================================================================")
    print(f"Training Complete in {total_training_time / 60:.2f} minutes | Best Val Acc: {best_acc * 100:.2f}%")
    print(f"================================================================================")
    log_ml_event(
        logger,
        event_type="TRAINING",
        operation="train_eye_model",
        status="SUCCESS",
        duration_ms=round(total_training_time * 1000.0, 2),
        details={"best_val_acc": best_acc, "total_epochs": cfg.num_epochs},
    )

    # Test set evaluation
    test_results: Optional[Dict[str, Any]] = None
    if cfg.eval_test_after_train and "test" in dataloaders:
        print("Evaluating best checkpoint on Test Split...")
        eval_model = build_training_model(
            num_classes=cfg.num_classes,
            use_local_pretrained=False,
            device=target_device,
        )
        # Load best weights if saved, otherwise current model
        if best_weights is not None:
            eval_model.load_state_dict(best_weights["state_dict"])
        else:
            eval_model.load_state_dict(model.state_dict())

        test_results = evaluate_dataset(
            model=eval_model,
            dataloader=dataloaders["test"],
            criterion=criterion,
            device=target_device,
            class_names=cfg.class_names,
        )
        print(f"Test Set Evaluation Results (N={test_results['total_samples']}):")
        print(f"  Test Loss:     {test_results['loss']:.4f}")
        print(f"  Test Accuracy: {test_results['accuracy'] * 100:.2f}%")
        print("  Per-class Accuracy:")
        for cls_name, acc in test_results["per_class_accuracy"].items():
            print(f"    - {cls_name:20s}: {acc * 100:.2f}%")
        print(f"================================================================================")

        if exp_run:
            best_ckpt_meta = exp_run.metadata.get("checkpoints", {}).get("best_checkpoint") or {}
            best_ckpt_sha = best_ckpt_meta.get("sha256")
            exp_run.link_evaluation(
                {
                    "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "dataset_split": "test",
                    "evaluated_images": test_results["total_samples"],
                    "checkpoint_sha256": best_ckpt_sha,
                    "metrics": {
                        "overall_accuracy": test_results["accuracy"],
                    },
                    "saved_report_path": "",
                }
            )

    if exp_run:
        exp_run.complete()

    # Determine resolved checkpoint paths for return
    saved_output_path = None
    if best_weights is not None:
        if cfg.promote_to_production or cfg.output_checkpoint_path != DEFAULT_CHECKPOINT_PATH:
            saved_output_path = cfg.output_checkpoint_path
        elif exp_run and exp_run.metadata.get("checkpoints", {}).get("best_checkpoint"):
            saved_output_path = exp_run.metadata["checkpoints"]["best_checkpoint"]["path"]
        else:
            saved_output_path = cfg.output_checkpoint_path

    saved_best_path = None
    if best_weights is not None:
        if cfg.promote_to_production or (cfg.best_checkpoint_path and cfg.best_checkpoint_path != DEFAULT_BEST_CHECKPOINT_PATH):
            saved_best_path = cfg.best_checkpoint_path
        elif exp_run and exp_run.metadata.get("checkpoints", {}).get("best_checkpoint"):
            saved_best_path = exp_run.metadata["checkpoints"]["best_checkpoint"]["path"]
        else:
            saved_best_path = cfg.best_checkpoint_path

    saved_recovery_path = cfg.recovery_checkpoint_path
    if (not cfg.promote_to_production) and cfg.recovery_checkpoint_path == DEFAULT_RECOVERY_CHECKPOINT_PATH and exp_run:
        if exp_run.metadata.get("checkpoints", {}).get("recovery_checkpoint"):
            saved_recovery_path = exp_run.metadata["checkpoints"]["recovery_checkpoint"]["path"]

    return {
        "best_val_accuracy": best_acc,
        "history": history,
        "checkpoint_saved": saved_output_path,
        "best_checkpoint_path": saved_best_path,
        "recovery_checkpoint_path": saved_recovery_path,
        "run_id": exp_run.run_id if exp_run else None,
        "run_dir": exp_run.run_dir if exp_run else None,
        "metadata_path": exp_run.metadata_path if exp_run else None,
        "total_training_time_seconds": total_training_time,
        "test_results": test_results,
    }


def parse_args() -> argparse.Namespace:
    """CLI argument parser for training invocation."""
    parser = argparse.ArgumentParser(description="DenseNet201 Eye Disease Classifier Training")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST_PATH, help="Path to split manifest")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate (default: 0.0005)")
    parser.add_argument("--resume", action="store_true", help="Resume from recovery checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Explicit checkpoint path or run ID to resume from")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weighting")
    parser.add_argument("--patience", type=int, default=3, help="Plateau LR scheduler patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--track", action="store_true", default=True, help="Enable experiment tracking (default: True)")
    parser.add_argument("--no-track", dest="track", action="store_false", help="Disable experiment tracking")
    parser.add_argument("--experiment-name", type=str, default="eye_densenet201_screening", help="Experiment name")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run identifier")
    parser.add_argument("--experiments-dir", type=str, default=None, help="Custom experiments directory")
    parser.add_argument("--description", type=str, default="", help="Experiment description")
    parser.add_argument("--promote", action="store_true", default=False, help="Promote best model to production weights")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig(
        manifest_path=args.manifest,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume=args.resume,
        resume_from=args.resume_from,
        use_class_weights=not args.no_class_weights,
        patience=args.patience,
        seed=args.seed,
        track_experiment=args.track,
        experiment_name=args.experiment_name,
        run_id=args.run_id,
        experiments_dir=args.experiments_dir,
        description=args.description,
        promote_to_production=args.promote,
    )
    results = train_eye_model(config)
    print(f"Job completed successfully. Best Val Acc: {results['best_val_accuracy'] * 100:.2f}%")

