"""
Tests for DenseNet201 Training Pipeline Configuration and Execution (ml/train_eye.py).

Verifies:
1. Training configuration default values and hyperparameters.
2. Dataloader construction for train, val, and test splits.
3. DenseNet201 model creation with 4-class classifier head.
4. Class weights calculation and normalization.
5. Evaluation function across dataset splits.
6. End-to-end dry-run training loop execution and checkpoint validation via load_eye_model_checkpoint.
7. Seamless resumption from periodic recovery checkpoint.
"""

import os
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES, save_split_manifest
from ml.eye_model import build_densenet201_model, load_eye_model_checkpoint
from ml.train_eye import (
    TrainConfig,
    build_training_model,
    compute_class_weights,
    evaluate_dataset,
    get_training_dataloaders,
    save_atomic_checkpoint,
    train_eye_model,
)


@pytest.fixture
def mini_manifest_and_images(tmp_path) -> str:
    """Creates a tiny synthetic dataset manifest for rapid dry-run training tests."""
    dataset_dir = str(tmp_path / "mini_data")
    os.makedirs(dataset_dir, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        for i, cls_name in enumerate(CLASS_NAMES):
            img_path = os.path.join(dataset_dir, f"{split}_{cls_name}_{i}.png")
            arr = np.full((224, 224, 3), 100 + i * 20, dtype=np.uint8)
            cv2.circle(arr, (112, 112), 40, (200, 150, 100), -1)
            cv2.imwrite(img_path, arr)

            splits[split].append(
                {
                    "filepath": img_path,
                    "filename": os.path.basename(img_path),
                    "class_name": cls_name,
                    "class_index": i,
                    "patient_id": f"p_{split}_{i}",
                    "split": split,
                }
            )

    manifest_path = str(tmp_path / "mini_manifest.json")
    save_split_manifest(splits, manifest_path)
    return manifest_path


class TestTrainConfigurationAndDataloaders:
    """Verifies training configuration and DataLoader construction."""

    def test_train_config_defaults(self):
        cfg = TrainConfig()
        assert cfg.num_classes == 4
        assert cfg.class_names == CLASS_NAMES
        assert cfg.batch_size == 8
        assert cfg.learning_rate == 0.001
        assert "eye_model.pth" in cfg.output_checkpoint_path
        assert "eye_densenet201_best.pth" in cfg.best_checkpoint_path
        assert "recovery_checkpoint.pth" in cfg.recovery_checkpoint_path
        assert cfg.use_class_weights is True

    def test_get_training_dataloaders(self, mini_manifest_and_images):
        loaders = get_training_dataloaders(
            manifest_path=mini_manifest_and_images,
            batch_size=2,
            num_workers=0,
        )

        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders

        batch_x, batch_y = next(iter(loaders["train"]))
        assert batch_x.shape == (2, 3, 224, 224)
        assert batch_y.shape == (2,)


class TestTrainingUtilities:
    """Verifies loss weights, model building, and evaluation utilities."""

    def test_compute_class_weights(self):
        # 100 cataract, 200 DR, 50 glaucoma, 400 normal
        counts = {
            "cataract": 100,
            "diabetic_retinopathy": 200,
            "glaucoma": 50,
            "normal": 400,
        }
        weights = compute_class_weights(counts, CLASS_NAMES)
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (4,)
        # Glaucoma has lowest count -> highest weight
        assert weights[2] > weights[0]
        # Normal has highest count -> lowest weight
        assert weights[3] < weights[1]
        # Mean normalized to ~1.0
        assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-4)

    def test_build_training_model_head_and_device(self):
        cpu_device = torch.device("cpu")
        model = build_training_model(
            num_classes=4,
            use_local_pretrained=False,
            device=cpu_device,
        )

        assert isinstance(model.classifier, nn.Linear)
        assert model.classifier.in_features == 1920
        assert model.classifier.out_features == 4
        assert next(model.parameters()).device == cpu_device

    def test_evaluate_dataset(self, mini_manifest_and_images):
        loaders = get_training_dataloaders(
            manifest_path=mini_manifest_and_images,
            batch_size=2,
            num_workers=0,
        )
        cpu_device = torch.device("cpu")
        model = build_training_model(num_classes=4, use_local_pretrained=False, device=cpu_device)
        criterion = nn.CrossEntropyLoss()

        eval_res = evaluate_dataset(
            model=model,
            dataloader=loaders["test"],
            criterion=criterion,
            device=cpu_device,
            class_names=CLASS_NAMES,
        )

        assert "loss" in eval_res
        assert "accuracy" in eval_res
        assert "per_class_accuracy" in eval_res
        assert "confusion_matrix" in eval_res
        assert eval_res["total_samples"] == 4
        assert len(eval_res["per_class_accuracy"]) == 4


@pytest.mark.slow
@pytest.mark.integration
class TestTrainingLoopExecution:
    """Verifies that the training loop runs, computes losses, and saves checkpoints."""

    def test_dry_run_training_epoch(self, mini_manifest_and_images, tmp_path):
        output_ckpt = str(tmp_path / "saved_test_model.pth")
        best_ckpt = str(tmp_path / "saved_best_model.pth")
        recovery_ckpt = str(tmp_path / "saved_recovery.pth")

        config = TrainConfig(
            manifest_path=mini_manifest_and_images,
            output_checkpoint_path=output_ckpt,
            best_checkpoint_path=best_ckpt,
            recovery_checkpoint_path=recovery_ckpt,
            batch_size=2,
            num_epochs=1,
            num_workers=0,
            use_local_pretrained=False,
            device=torch.device("cpu"),
            eval_test_after_train=True,
        )

        result = train_eye_model(config)

        assert "best_val_accuracy" in result
        assert "history" in result
        assert len(result["history"]["train_loss"]) == 1
        assert len(result["history"]["val_loss"]) == 1
        assert os.path.exists(output_ckpt)
        assert os.path.exists(recovery_ckpt)
        assert result["test_results"] is not None

        # Verify checkpoint content and strict loading with load_eye_model_checkpoint
        ckpt_data = torch.load(output_ckpt, map_location="cpu", weights_only=True)
        assert "state_dict" in ckpt_data
        assert "classes" in ckpt_data
        assert ckpt_data["classes"] == CLASS_NAMES
        assert ckpt_data["state_dict"]["classifier.weight"].shape == torch.Size([4, 1920])

        fresh_model = build_densenet201_model(num_classes=4)
        loaded_model = load_eye_model_checkpoint(fresh_model, output_ckpt, device=torch.device("cpu"))
        assert loaded_model is not None

    def test_resume_from_recovery_checkpoint(self, mini_manifest_and_images, tmp_path):
        recovery_ckpt = str(tmp_path / "mock_recovery.pth")
        output_ckpt = str(tmp_path / "resumed_output.pth")

        # Create a mock recovery checkpoint indicating epoch 2 completed
        model = build_training_model(num_classes=4, use_local_pretrained=False, device=torch.device("cpu"))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        recovery_payload = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "classes": CLASS_NAMES,
            "epoch": 2,
            "best_acc": 0.5,
            "best_val_accuracy": 0.5,
            "history": {
                "train_loss": [1.0, 0.8],
                "train_acc": [0.25, 0.5],
                "val_loss": [1.1, 0.9],
                "val_acc": [0.25, 0.5],
                "lr": [0.001, 0.001],
            },
            "seed": 42,
            "architecture": "DenseNet201",
        }
        save_atomic_checkpoint(recovery_payload, recovery_ckpt)

        # Run with resume=True and total epochs = 3 (should run only epoch 3)
        config = TrainConfig(
            manifest_path=mini_manifest_and_images,
            output_checkpoint_path=output_ckpt,
            recovery_checkpoint_path=recovery_ckpt,
            batch_size=2,
            num_epochs=3,
            resume=True,
            use_local_pretrained=False,
            device=torch.device("cpu"),
        )

        res = train_eye_model(config)
        assert len(res["history"]["train_loss"]) == 3  # 2 from history + 1 newly trained
