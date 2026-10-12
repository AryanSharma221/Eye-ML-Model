"""
Lightweight, Reproducible Experiment Tracking & Checkpoint Versioning Engine.

Provides self-contained, machine-readable tracking for Multimodal Health Screening models:
1. Unique Run Directories: Isolates all artifacts (metadata, checkpoints, evaluations, logs).
2. Cryptographic Checkpoint Versioning: SHA-256 checksums, pre-use verification, atomic writes.
3. Provenance & Reproducibility: Captures exact dataset manifest hash, random seeds, hyperparams,
   hardware device, dependency versions, and determinism settings.
4. Safe Checkpoint Promotion: Strictly protects existing production checkpoints from accidental overwriting.
5. Recoverability: Structured periodic recovery checkpoints, interruption recording, and resume verification.
6. Run-to-Evaluation Linkage: Couples held-out test evaluations directly to their training run and checkpoint hash.
7. Local-First Architecture: Zero external server or cloud dependencies (pure JSON + filesystem).
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import torch
import torch.nn as nn

from ml.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    PROJECT_ROOT,
    get_default_checkpoint_path,
    get_default_experiments_dir,
    get_ml_config,
    resolve_device,
)
from ml.monitoring import log_ml_event, sanitize_path

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Cryptographic Hashing and Verification
# =============================================================================

def compute_file_sha256(filepath: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Computes SHA-256 cryptographic hash of a file on disk.

    Args:
        filepath: Absolute or relative path to target file.
        chunk_size: Read buffer size in bytes (default: 1 MB).

    Returns:
        Hexadecimal SHA-256 digest string.

    Raises:
        FileNotFoundError: If target file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found for hash calculation: {filepath}")

    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checkpoint_integrity(
    checkpoint_path: str,
    num_classes: int = NUM_CLASSES,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Verifies that a checkpoint is physically intact, secure, and loadable by DenseNet201.

    Validation steps:
    1. Computes SHA-256 hash.
    2. Loads checkpoint using strict PyTorch weights_only=True protocol.
    3. Confirms payload dictionary structure and presence of 'state_dict'.
    4. Validates classifier weight dimension (num_classes, 1920) and bias dimension (num_classes,).
    5. Checks class order alignment when class metadata is present.
    6. Executes a single zero-tensor forward pass to prove numerical viability (no NaNs/Infs).

    Args:
        checkpoint_path: Path to checkpoint file on disk.
        num_classes: Expected number of classes (default: 4).
        device: Device to use for test forward pass (default: CPU).

    Returns:
        dict: Integrity verification report with 'valid', 'sha256', metadata, or 'error'.
    """
    if not os.path.isfile(checkpoint_path):
        return {
            "valid": False,
            "error": f"Checkpoint file does not exist: {checkpoint_path}",
            "sha256": None,
        }

    target_device = device or torch.device("cpu")
    file_sha256 = compute_file_sha256(checkpoint_path)
    file_size_bytes = os.path.getsize(checkpoint_path)

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as err:
        return {
            "valid": False,
            "error": f"PyTorch safe loader failed: {err}",
            "sha256": file_sha256,
            "file_size_bytes": file_size_bytes,
        }

    if not isinstance(payload, dict):
        return {
            "valid": False,
            "error": f"Checkpoint payload is {type(payload).__name__}, expected dict",
            "sha256": file_sha256,
        }

    state_dict = payload.get("state_dict")
    if state_dict is None or not isinstance(state_dict, dict):
        return {
            "valid": False,
            "error": "Checkpoint payload missing 'state_dict' dictionary",
            "sha256": file_sha256,
        }

    # Head dimension checks
    weight_key = "classifier.weight"
    bias_key = "classifier.bias"
    if weight_key not in state_dict:
        return {
            "valid": False,
            "error": f"State dict missing classifier weight '{weight_key}'",
            "sha256": file_sha256,
        }

    weight_shape = tuple(state_dict[weight_key].shape)
    expected_weight_shape = (num_classes, 1920)
    if weight_shape != expected_weight_shape:
        return {
            "valid": False,
            "error": f"Classifier weight shape {weight_shape} != expected {expected_weight_shape}",
            "sha256": file_sha256,
        }

    if bias_key in state_dict:
        bias_shape = tuple(state_dict[bias_key].shape)
        if bias_shape != (num_classes,):
            return {
                "valid": False,
                "error": f"Classifier bias shape {bias_shape} != expected ({num_classes},)",
                "sha256": file_sha256,
            }

    # Class order check if recorded
    classes_recorded = payload.get("classes")
    if classes_recorded and list(classes_recorded) != list(CLASS_NAMES):
        return {
            "valid": False,
            "error": f"Class order mismatch in checkpoint: {classes_recorded} != {CLASS_NAMES}",
            "sha256": file_sha256,
        }

    # Dry-run numerical forward pass check
    try:
        from ml.eye_model import build_densenet201_model
        test_model = build_densenet201_model(num_classes=num_classes)
        test_model.load_state_dict(state_dict, strict=True)
        test_model.eval()
        test_model = test_model.to(target_device)

        dummy_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=target_device)
        with torch.inference_mode():
            dummy_out = test_model(dummy_input)

        if not torch.isfinite(dummy_out).all():
            return {
                "valid": False,
                "error": "Forward pass produced non-finite logits (NaN or Inf)",
                "sha256": file_sha256,
            }
    except Exception as forward_err:
        return {
            "valid": False,
            "error": f"Forward pass test failed: {forward_err}",
            "sha256": file_sha256,
        }

    return {
        "valid": True,
        "sha256": file_sha256,
        "file_size_bytes": file_size_bytes,
        "epoch": payload.get("epoch"),
        "best_val_accuracy": payload.get("best_acc", payload.get("best_val_accuracy")),
        "architecture": payload.get("architecture", "DenseNet201"),
        "classes": list(classes_recorded) if classes_recorded else list(CLASS_NAMES),
        "has_optimizer": "optimizer_state_dict" in payload,
        "has_scheduler": "scheduler_state_dict" in payload,
    }


# =============================================================================
# 2. System and Dataset Metadata Collection
# =============================================================================

def collect_system_environment() -> Dict[str, Any]:
    """
    Collects exact software versions, hardware specifications, and determinism settings.
    Excludes sensitive personal information and environment secrets.
    """
    git_commit = "unavailable"
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            git_commit = res.stdout.strip()
    except Exception:
        pass

    cuda_info: Dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }

    if torch.cuda.is_available():
        try:
            cuda_info["vram_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2
            )
        except Exception:
            pass

    import torchvision
    import cv2
    import PIL
    import numpy as np

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python_version": sys.version.split()[0],
        "git_commit": git_commit,
        "dependencies": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
            "numpy": np.__version__,
        },
        "cuda": cuda_info,
    }


def collect_dataset_metadata(manifest_path: str) -> Dict[str, Any]:
    """
    Parses split manifest, computes cryptographic checksum, and extracts class/split distributions.
    """
    if not os.path.exists(manifest_path):
        return {
            "manifest_path": sanitize_path(manifest_path),
            "manifest_sha256": None,
            "error": "Manifest file not found",
        }

    manifest_sha = compute_file_sha256(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    splits = data.get("splits", {})

    counts_by_split: Dict[str, int] = {}
    for split_name, recs in splits.items():
        counts_by_split[split_name] = len(recs)

    class_dist: Dict[str, Dict[str, int]] = meta.get("class_distribution", {})
    if not class_dist and splits:
        # Tally on the fly if not precomputed in metadata header
        class_dist = {}
        for split_name, recs in splits.items():
            split_tally: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
            for r in recs:
                cname = r.get("class_name")
                if cname in split_tally:
                    split_tally[cname] += 1
            class_dist[split_name] = split_tally

    return {
        "manifest_path": sanitize_path(manifest_path),
        "manifest_sha256": manifest_sha,
        "split_manifest_sha256": manifest_sha,
        "total_images": meta.get("total_images", sum(counts_by_split.values())),
        "split_counts": counts_by_split,
        "class_distribution": class_dist,
        "canonical_classes": list(CLASS_NAMES),
        "num_classes": NUM_CLASSES,
    }


# =============================================================================
# 3. Experiment Run Life-Cycle Manager
# =============================================================================

class ExperimentRun:
    """
    Manages the lifecycle, metadata, checkpoints, and evaluations for a single training run.
    """

    def __init__(
        self,
        run_id: str,
        experiments_root: str,
        experiment_name: str = "eye_densenet201_screening",
        description: str = "",
        tags: Optional[List[str]] = None,
    ):
        self.run_id = run_id
        self.experiment_name = experiment_name
        self.description = description
        self.tags = tags or []
        self.experiments_root = os.path.abspath(experiments_root)

        # Isolated directory structure for this run
        self.run_dir = os.path.join(self.experiments_root, "runs", self.run_id)
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.evaluations_dir = os.path.join(self.run_dir, "evaluations")
        self.logs_dir = os.path.join(self.run_dir, "logs")
        self.metadata_path = os.path.join(self.run_dir, "run_metadata.json")

        self.start_time_epoch = time.time()
        self.metadata: Dict[str, Any] = {}
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Creates the run's isolated subdirectories."""
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.evaluations_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def _save_metadata_atomic(self) -> None:
        """Atomically flushes the in-memory metadata dictionary to disk."""
        self._ensure_dirs()
        tmp_path = f"{self.metadata_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        try:
            os.replace(tmp_path, self.metadata_path)
        except OSError:
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            os.replace(tmp_path, self.metadata_path)

    def start(
        self,
        config_dict: Dict[str, Any],
        manifest_path: str,
        device: Union[str, torch.device],
        seed: int = 42,
    ) -> "ExperimentRun":
        """
        Initializes the training run on disk and writes initial provenance metadata.
        """
        self._ensure_dirs()
        self.start_time_epoch = time.time()
        start_utc = datetime.now(timezone.utc).isoformat()

        # Sanitize device representation
        dev_str = str(device)

        # Collect environment and dataset metadata
        env_meta = collect_system_environment()
        ds_meta = collect_dataset_metadata(manifest_path)

        self.metadata = {
            "schema_version": "1.0.0",
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "description": self.description,
            "tags": self.tags,
            "status": "RUNNING",
            "timestamps": {
                "start_utc": start_utc,
                "completion_utc": None,
                "total_duration_seconds": None,
            },
            "reproducibility": {
                "seed": seed,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "git_commit": env_meta.get("git_commit"),
            },
            "device": {
                "configured": dev_str,
                "cuda_device_name": env_meta["cuda"].get("current_device_name"),
            },
            "dataset": ds_meta,
            "model": {
                "architecture": "DenseNet201",
                "in_features": 1920,
                "num_classes": NUM_CLASSES,
                "class_names": list(CLASS_NAMES),
                "pretrained_backbone": config_dict.get("use_local_pretrained", True),
                "total_parameters": 18100612,
            },
            "hyperparameters": {
                "batch_size": config_dict.get("batch_size"),
                "num_epochs": config_dict.get("num_epochs"),
                "learning_rate": config_dict.get("learning_rate"),
                "weight_decay": config_dict.get("weight_decay"),
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau",
                "scheduler_factor": 0.5,
                "patience": config_dict.get("patience", 3),
                "loss_criterion": "CrossEntropyLoss",
                "use_class_weights": config_dict.get("use_class_weights", True),
                "checkpoint_selection_metric": "max_val_accuracy",
            },
            "preprocessing": {
                "target_resolution": [224, 224],
                "color_mode": "RGB",
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "data_augmentation": "RandomHorizontalFlip(p=0.5), RandomRotation(degrees=15)",
            },
            "environment": env_meta,
            "epoch_history": [],
            "best_metrics": {
                "best_epoch": None,
                "best_val_accuracy": None,
                "best_val_loss": None,
            },
            "checkpoints": {
                "best_checkpoint": None,
                "final_checkpoint": None,
                "recovery_checkpoint": None,
            },
            "evaluations": [],
            "failure_reason": None,
        }

        self._save_metadata_atomic()
        log_ml_event(
            logger,
            event_type="EXPERIMENT_RUN",
            operation="start",
            status="INITIALIZED",
            request_id=self.run_id,
            details={"run_id": self.run_id, "experiment": self.experiment_name},
        )
        return self

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        duration_seconds: float,
    ) -> None:
        """
        Logs a single completed epoch's metrics to the run metadata.
        """
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "train_acc": round(float(train_acc), 4),
            "val_loss": round(float(val_loss), 4),
            "val_acc": round(float(val_acc), 4),
            "learning_rate": float(lr),
            "epoch_duration_seconds": round(float(duration_seconds), 2),
        }
        self.metadata["epoch_history"].append(epoch_record)
        self._save_metadata_atomic()

    def save_run_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        is_best: bool = False,
        is_recovery: bool = False,
        is_final: bool = False,
        best_val_acc: Optional[float] = None,
        val_loss: Optional[float] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> str:
        """
        Safely versions and saves a checkpoint inside this run's isolated directory.

        Guarantees:
        - Writes atomically via temporary file.
        - Calculates SHA-256 hash immediately after write.
        - Verifies integrity (weights_only readability and DenseNet201 compatibility).
        - Embeds verified hash and epoch metadata in run_metadata.json.
        - Never touches production or other runs' checkpoints.

        Returns:
            The absolute path of the verified saved checkpoint.
        """
        self._ensure_dirs()

        if is_best:
            filename = "best_model.pth"
        elif is_recovery:
            filename = "recovery_checkpoint.pth"
        elif is_final:
            filename = "final_model.pth"
        else:
            filename = f"checkpoint_epoch_{epoch:03d}.pth"

        target_path = os.path.join(self.checkpoints_dir, filename)
        tmp_path = f"{target_path}.tmp"

        checkpoint_payload: Dict[str, Any] = {
            "state_dict": model.state_dict(),
            "classes": list(CLASS_NAMES),
            "epoch": epoch,
            "best_acc": best_val_acc if best_val_acc is not None else 0.0,
            "best_val_accuracy": best_val_acc if best_val_acc is not None else 0.0,
            "architecture": "DenseNet201",
            "run_id": self.run_id,
            "seed": self.metadata.get("reproducibility", {}).get("seed", 42),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

        # Resumable checkpoints require optimizer & scheduler states
        if is_recovery or is_final:
            if optimizer is not None:
                checkpoint_payload["optimizer_state_dict"] = optimizer.state_dict()
            if scheduler is not None and hasattr(scheduler, "state_dict"):
                checkpoint_payload["scheduler_state_dict"] = scheduler.state_dict()
            checkpoint_payload["history"] = self.metadata.get("epoch_history", [])

        # 1. Atomic write
        torch.save(checkpoint_payload, tmp_path)
        try:
            os.replace(tmp_path, target_path)
        except OSError:
            if os.path.exists(target_path):
                os.remove(target_path)
            os.replace(tmp_path, target_path)

        # 2. Cryptographic hash calculation & verification
        verification = verify_checkpoint_integrity(target_path, num_classes=NUM_CLASSES)
        if not verification.get("valid", False):
            raise RuntimeError(
                f"Saved checkpoint failed integrity validation: {verification.get('error')}"
            )

        ckpt_record = {
            "path": sanitize_path(target_path),
            "filename": filename,
            "sha256": verification["sha256"],
            "file_size_bytes": verification["file_size_bytes"],
            "epoch": epoch,
            "best_val_accuracy": best_val_acc,
            "val_loss": val_loss,
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "verified": True,
        }

        if is_best:
            self.metadata["checkpoints"]["best_checkpoint"] = ckpt_record
            self.metadata["best_metrics"]["best_epoch"] = epoch
            self.metadata["best_metrics"]["best_val_accuracy"] = best_val_acc
            self.metadata["best_metrics"]["best_val_loss"] = val_loss
        elif is_recovery:
            self.metadata["checkpoints"]["recovery_checkpoint"] = ckpt_record
        elif is_final:
            self.metadata["checkpoints"]["final_checkpoint"] = ckpt_record

        self._save_metadata_atomic()
        return target_path

    def complete(self) -> None:
        """Marks the run as successfully completed and calculates total elapsed time."""
        end_utc = datetime.now(timezone.utc).isoformat()
        duration = round(time.time() - self.start_time_epoch, 2)

        self.metadata["status"] = "COMPLETED"
        self.metadata["timestamps"]["completion_utc"] = end_utc
        self.metadata["timestamps"]["total_duration_seconds"] = duration
        self._save_metadata_atomic()

        log_ml_event(
            logger,
            event_type="EXPERIMENT_RUN",
            operation="complete",
            status="SUCCESS",
            request_id=self.run_id,
            duration_ms=duration * 1000.0,
            details={"run_id": self.run_id, "duration_s": duration},
        )

    def fail(self, error_message: str, exception: Optional[Exception] = None) -> None:
        """Marks the run as failed with structured error diagnostic."""
        end_utc = datetime.now(timezone.utc).isoformat()
        duration = round(time.time() - self.start_time_epoch, 2)

        self.metadata["status"] = "FAILED"
        self.metadata["failure_reason"] = error_message
        self.metadata["timestamps"]["completion_utc"] = end_utc
        self.metadata["timestamps"]["total_duration_seconds"] = duration
        self._save_metadata_atomic()

        log_ml_event(
            logger,
            event_type="EXPERIMENT_RUN",
            operation="fail",
            status="FAILURE",
            request_id=self.run_id,
            duration_ms=duration * 1000.0,
            details={"run_id": self.run_id, "error": error_message},
            level=logging.ERROR,
        )

    def interrupt(self, reason: str = "Training interrupted by user or external signal") -> None:
        """Marks the run as interrupted, ensuring recovery checkpoint is documented."""
        end_utc = datetime.now(timezone.utc).isoformat()
        duration = round(time.time() - self.start_time_epoch, 2)

        self.metadata["status"] = "INTERRUPTED"
        self.metadata["failure_reason"] = reason
        self.metadata["timestamps"]["completion_utc"] = end_utc
        self.metadata["timestamps"]["total_duration_seconds"] = duration
        self._save_metadata_atomic()

        log_ml_event(
            logger,
            event_type="EXPERIMENT_RUN",
            operation="interrupt",
            status="WARNING",
            request_id=self.run_id,
            details={"run_id": self.run_id, "reason": reason},
            level=logging.WARNING,
        )

    def link_evaluation(self, eval_report: Dict[str, Any]) -> None:
        """
        Attaches a held-out test set evaluation report to this run's record.
        """
        eval_link = {
            "evaluation_timestamp": eval_report.get("evaluation_timestamp"),
            "dataset_split": eval_report.get("dataset_split", "test"),
            "evaluated_images": eval_report.get("evaluated_images"),
            "checkpoint_sha256": eval_report.get("checkpoint_sha256"),
            "overall_accuracy": eval_report.get("metrics", {}).get("overall_accuracy"),
            "macro_f1": eval_report.get("metrics", {}).get("macro_f1"),
            "weighted_f1": eval_report.get("metrics", {}).get("weighted_f1"),
            "report_path": sanitize_path(eval_report.get("saved_report_path", "")),
        }
        self.metadata["evaluations"].append(eval_link)
        self._save_metadata_atomic()


# =============================================================================
# 4. Global Tracker and Experiment Registry
# =============================================================================

class ExperimentTracker:
    """
    Central experiment tracking registry for creating, discovering, and summarizing runs.
    """

    def __init__(
        self,
        experiments_root: Optional[str] = None,
        experiments_dir: Optional[str] = None,
    ):
        target_root = experiments_dir or experiments_root or get_default_experiments_dir()
        self.experiments_root = os.path.abspath(target_root)
        self.runs_root = os.path.join(self.experiments_root, "runs")
        os.makedirs(self.runs_root, exist_ok=True)

    def create_run(
        self,
        experiment_name: str = "eye_densenet201_screening",
        run_id: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> ExperimentRun:
        """
        Allocates and initializes a new unique experiment run.
        """
        if run_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:8]
            run_id = f"run_{ts}_{short_uuid}"

        return ExperimentRun(
            run_id=run_id,
            experiments_root=self.experiments_root,
            experiment_name=experiment_name,
            description=description,
            tags=tags,
        )

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Loads an existing run by run_id."""
        run_meta = os.path.join(self.runs_root, run_id, "run_metadata.json")
        if not os.path.isfile(run_meta):
            return None

        run = ExperimentRun(run_id=run_id, experiments_root=self.experiments_root)
        try:
            with open(run_meta, "r", encoding="utf-8") as f:
                run.metadata = json.load(f)
            return run
        except Exception:
            return None

    def list_runs(self) -> List[Dict[str, Any]]:
        """
        Discovers and summarizes all tracked runs in chronological order.
        """
        if not os.path.exists(self.runs_root):
            return []

        runs_summary: List[Dict[str, Any]] = []
        for entry in os.listdir(self.runs_root):
            meta_path = os.path.join(self.runs_root, entry, "run_metadata.json")
            if not os.path.isfile(meta_path):
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                # Extract best checkpoint summary
                best_ckpt = meta.get("checkpoints", {}).get("best_checkpoint")
                best_sha = best_ckpt.get("sha256") if best_ckpt else None
                best_acc = meta.get("best_metrics", {}).get("best_val_accuracy")

                # Extract latest evaluation summary
                evals = meta.get("evaluations", [])
                latest_eval = evals[-1] if evals else {}
                test_acc = latest_eval.get("overall_accuracy")
                test_f1 = latest_eval.get("macro_f1")

                runs_summary.append({
                    "run_id": meta.get("run_id", entry),
                    "experiment_name": meta.get("experiment_name", "unknown"),
                    "status": meta.get("status", "UNKNOWN"),
                    "start_utc": meta.get("timestamps", {}).get("start_utc", "N/A"),
                    "epochs_completed": len(meta.get("epoch_history", [])),
                    "best_val_accuracy": best_acc,
                    "best_checkpoint_sha256": best_sha[:10] if best_sha else None,
                    "test_accuracy": test_acc,
                    "test_macro_f1": test_f1,
                    "has_recovery_checkpoint": meta.get("checkpoints", {}).get("recovery_checkpoint") is not None,
                })
            except Exception as err:
                runs_summary.append({
                    "run_id": entry,
                    "status": "CORRUPT_METADATA",
                    "error": str(err),
                })

        # Sort newest first
        runs_summary.sort(key=lambda x: x.get("start_utc", ""), reverse=True)
        return runs_summary

    def audit_legacy_checkpoints(self, weights_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Audits legacy baseline checkpoints via the module auditor."""
        return audit_legacy_checkpoints(weights_dir=weights_dir)


# =============================================================================
# 5. Legacy Production Checkpoint Auditor
# =============================================================================

def audit_legacy_checkpoints(weights_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Audits existing baseline checkpoints in ml/weights, computing SHA-256 hashes and
    verifying structure without modifying or overwriting any files.
    """
    w_dir = weights_dir or os.path.join(os.path.dirname(__file__), "weights")
    if not os.path.isdir(w_dir):
        return []

    results = []
    for f in sorted(os.listdir(w_dir)):
        if not f.endswith(".pth"):
            continue
        full_path = os.path.join(w_dir, f)
        integrity = verify_checkpoint_integrity(full_path)
        results.append({
            "filename": f,
            "path": sanitize_path(full_path),
            "size_mb": round(os.path.getsize(full_path) / (1024 * 1024), 2),
            "file_size_bytes": os.path.getsize(full_path),
            "sha256": integrity.get("sha256"),
            "valid": integrity.get("valid", False),
            "integrity": integrity,
            "epoch": integrity.get("epoch"),
            "best_val_accuracy": integrity.get("best_val_accuracy"),
            "has_optimizer": integrity.get("has_optimizer", False),
            "is_resumable": integrity.get("has_optimizer", False) and integrity.get("has_scheduler", False),
            "error": integrity.get("error"),
        })
    return results
