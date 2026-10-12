"""
Tests for Model Evaluation Module (ml/evaluate.py).

Verifies:
1. Metric calculation functions (precision, recall, F1, macro/weighted, confusion matrix) with known inputs.
2. Zero-division and empty-input handling.
3. Dimension and canonical class order enforcement.
4. Test-split isolation and evaluation pipeline execution with synthetic data.
5. JSON report generation and schema compliance.
6. Checkpoint validation and error handling on invalid files.
7. End-to-end inference output contract compatibility.
"""

import json
import os
import cv2
import numpy as np
import pytest
import torch

from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES, save_split_manifest
from ml.evaluate import (
    calculate_classification_metrics,
    evaluate_model_checkpoint,
)
from ml.eye_model import (
    build_densenet201_model,
    predict_eye_disease,
    screen_eye_image,
)
from ml.train_eye import save_atomic_checkpoint


@pytest.fixture
def known_classification_labels():
    """Returns fixed ground truth and predictions for deterministic metric tests."""
    # Class order: 0: cataract, 1: diabetic_retinopathy, 2: glaucoma, 3: normal
    # 4 classes, 12 samples
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    y_pred = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]
    return y_true, y_pred


@pytest.fixture
def eval_synthetic_manifest_and_checkpoint(tmp_path):
    """Creates a tiny synthetic test dataset and valid checkpoint for evaluation tests."""
    dataset_dir = str(tmp_path / "eval_data")
    os.makedirs(dataset_dir, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        for i, cls_name in enumerate(CLASS_NAMES):
            img_path = os.path.join(dataset_dir, f"{split}_{cls_name}_{i}.png")
            arr = np.full((224, 224, 3), 120 + i * 15, dtype=np.uint8)
            cv2.circle(arr, (112, 112), 45, (180, 140, 90), -1)
            cv2.imwrite(img_path, arr)

            splits[split].append({
                "filepath": img_path,
                "filename": os.path.basename(img_path),
                "class_name": cls_name,
                "class_index": i,
                "patient_id": f"p_{split}_{i}",
                "split": split,
            })

    manifest_path = str(tmp_path / "eval_manifest.json")
    save_split_manifest(splits, manifest_path)

    # Create a small valid checkpoint
    ckpt_path = str(tmp_path / "mock_eval_model.pth")
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    payload = {
        "state_dict": model.state_dict(),
        "classes": list(CLASS_NAMES),
        "epoch": 3,
        "best_acc": 0.85,
        "best_val_accuracy": 0.85,
        "architecture": "DenseNet201",
        "seed": 42,
    }
    save_atomic_checkpoint(payload, ckpt_path)

    return manifest_path, ckpt_path


@pytest.mark.eval
class TestMetricCalculations:
    """Verifies precision, recall, F1, and confusion matrix arithmetic."""

    def test_metrics_deterministic_values(self, known_classification_labels):
        y_true, y_pred = known_classification_labels
        res = calculate_classification_metrics(y_true, y_pred, CLASS_NAMES)

        # 8 correct out of 12 = 0.6667
        assert res["overall_accuracy"] == round(8 / 12, 4)
        assert res["total_samples"] == 12

        # Verify confusion matrix dimensions [4 x 4]
        cm = res["confusion_matrix"]
        assert len(cm) == 4
        assert all(len(row) == 4 for row in cm)

        # Check support: each class had 3 samples
        for name in CLASS_NAMES:
            assert res["support"][name] == 3

        # Cataract (0): TP=2, FP=1 (sample 11 predicted 0), FN=1 (sample 2 predicted 1)
        # Precision = 2 / (2 + 1) = 0.6667, Recall = 2 / 3 = 0.6667
        cat_metrics = res["per_class"]["cataract"]
        assert cat_metrics["precision"] == round(2 / 3, 4)
        assert cat_metrics["recall"] == round(2 / 3, 4)
        assert cat_metrics["f1_score"] == round(2 / 3, 4)

        # Macro and weighted F1 should be finite and within [0, 1]
        assert 0.0 <= res["macro_f1"] <= 1.0
        assert 0.0 <= res["weighted_f1"] <= 1.0

    def test_empty_input_handling(self):
        res = calculate_classification_metrics([], [], CLASS_NAMES)
        assert res["total_samples"] == 0
        assert res["overall_accuracy"] == 0.0
        assert res["macro_f1"] == 0.0

    def test_zero_division_resilience(self):
        # A class that is never predicted or has 0 support
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 0, 0]  # Only class 0 predicted; classes 1, 2, 3 have 0 precision
        res = calculate_classification_metrics(y_true, y_pred, CLASS_NAMES)

        # Glaucoma (class 2) has 0 support and 0 predictions -> precision and recall 0.0 without crash
        glauc = res["per_class"]["glaucoma"]
        assert glauc["precision"] == 0.0
        assert glauc["recall"] == 0.0
        assert glauc["f1_score"] == 0.0

    def test_mismatched_lengths_raises_error(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            calculate_classification_metrics([0, 1], [0], CLASS_NAMES)

    def test_out_of_bounds_class_index_raises_error(self):
        with pytest.raises(ValueError, match="out of bounds"):
            calculate_classification_metrics([0, 5], [0, 1], CLASS_NAMES)


@pytest.mark.eval
class TestEvaluationExecution:
    """Verifies end-to-end evaluation pipeline with checkpoints and manifests."""

    def test_evaluate_model_checkpoint_execution(
        self, eval_synthetic_manifest_and_checkpoint, tmp_path
    ):
        manifest_path, ckpt_path = eval_synthetic_manifest_and_checkpoint
        out_dir = str(tmp_path / "eval_out")

        report = evaluate_model_checkpoint(
            checkpoint_path=ckpt_path,
            manifest_path=manifest_path,
            batch_size=2,
            device=torch.device("cpu"),
            output_dir=out_dir,
            save_report=True,
        )

        assert report["dataset_split"] == "test"
        assert report["total_test_images"] == 4
        assert report["evaluated_images"] == 4
        assert report["failed_or_excluded_images"] == 0
        assert "metrics" in report
        assert "confusion_matrix" in report["metrics"]
        assert "overall_accuracy" in report["metrics"]
        assert os.path.exists(report["saved_report_path"])
        assert os.path.exists(report["latest_report_path"])

        # Check JSON deserialization integrity
        with open(report["latest_report_path"], "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["checkpoint_metadata"]["epoch"] == 3
        assert data["class_names"] == CLASS_NAMES

    def test_missing_checkpoint_raises_error(self, eval_synthetic_manifest_and_checkpoint):
        manifest_path, _ = eval_synthetic_manifest_and_checkpoint
        with pytest.raises(FileNotFoundError):
            evaluate_model_checkpoint(
                checkpoint_path="nonexistent_checkpoint.pth",
                manifest_path=manifest_path,
                device=torch.device("cpu"),
                save_report=False,
            )

    def test_missing_manifest_raises_error(self, eval_synthetic_manifest_and_checkpoint):
        _, ckpt_path = eval_synthetic_manifest_and_checkpoint
        with pytest.raises(FileNotFoundError):
            evaluate_model_checkpoint(
                checkpoint_path=ckpt_path,
                manifest_path="nonexistent_manifest.json",
                device=torch.device("cpu"),
                save_report=False,
            )


class TestInferenceOutputContract:
    """Verifies inference contracts against evaluation requirements."""

    def test_inference_contract_keys_and_probabilities(
        self, eval_synthetic_manifest_and_checkpoint
    ):
        manifest_path, ckpt_path = eval_synthetic_manifest_and_checkpoint
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        sample_img = manifest["splits"]["test"][0]["filepath"]

        res = predict_eye_disease(
            sample_img,
            weights_path=ckpt_path,
            validate_quality=False,
            device=torch.device("cpu"),
        )

        # Check required public contract keys
        required_keys = ["prediction", "class_index", "confidence", "risk_level", "all_probabilities"]
        for k in required_keys:
            assert k in res, f"Missing required key: {k}"

        assert res["prediction"] in CLASS_NAMES
        assert 0 <= res["class_index"] < NUM_CLASSES
        assert 0.0 <= res["confidence"] <= 1.0
        assert res["risk_level"] in ["LOW", "MODERATE", "HIGH"]

        # Check all_probabilities structure and sum
        probs = res["all_probabilities"]
        assert len(probs) == NUM_CLASSES
        for c in CLASS_NAMES:
            assert c in probs
            assert 0.0 <= probs[c] <= 1.0

        total_p = sum(probs.values())
        assert abs(total_p - 1.0) < 1e-4

        # Confidence matches predicted class probability
        assert abs(res["confidence"] - probs[res["prediction"]]) < 1e-4

        # Ensure response is JSON-serializable
        json_str = json.dumps(res)
        assert json_str is not None
