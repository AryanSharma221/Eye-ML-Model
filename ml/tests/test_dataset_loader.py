"""
Tests for Dataset Discovery, Patient Stratified Splitting, and EyeDiseaseDataset (ml/dataset_loader.py).

Verifies:
1. Discovery of supported image formats (.jpg, .jpeg, .png, .bmp).
2. Label folder mapping to canonical 4 classes (cataract, diabetic_retinopathy, glaucoma, normal).
3. Detection and exclusion of unreadable, 0-byte, and conflicting duplicate images.
4. Patient grouping logic (Kaggle left/right eyes, RFMiD identifiers).
5. Reproducible stratified splitting (70/15/15 default ratio) with fixed seed.
6. Strict patient isolation (zero patient leakage across train, val, and test splits).
7. Zero record omission across splits.
8. Manifest JSON serialization and deserialization.
9. PyTorch EyeDiseaseDataset compatibility and DataLoader mini-batch generation.
"""

import json
import os
import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ml.dataset_loader import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    NUM_CLASSES,
    EyeDiseaseDataset,
    create_stratified_patient_split,
    discover_and_validate_dataset,
    load_split_manifest,
    map_folder_to_class,
    parse_patient_identifier,
    save_split_manifest,
)


@pytest.fixture
def synthetic_dataset_dir(tmp_path) -> str:
    """
    Creates a temporary synthetic dataset structure mimicking the project's classes and patient naming.
    """
    root = str(tmp_path / "mock_dataset")
    os.makedirs(root, exist_ok=True)

    # Create class folders
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(root, cls), exist_ok=True)

    def _create_dummy_image(path: str, color: tuple = (120, 120, 120)):
        img = np.full((100, 100, 3), color, dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, (200, 150, 100), -1)
        cv2.imwrite(path, img)

    # 1. Cataract: 4 patients (2 with left+right eyes, 2 single) -> 6 images
    _create_dummy_image(os.path.join(root, "cataract", "kaggle_101_left.jpg"), (100, 100, 100))
    _create_dummy_image(os.path.join(root, "cataract", "kaggle_101_right.jpg"), (105, 100, 100))
    _create_dummy_image(os.path.join(root, "cataract", "kaggle_102_left.jpeg"), (110, 100, 100))
    _create_dummy_image(os.path.join(root, "cataract", "kaggle_102_right.jpeg"), (115, 100, 100))
    _create_dummy_image(os.path.join(root, "cataract", "kaggle_103_left.png"), (120, 100, 100))
    _create_dummy_image(os.path.join(root, "cataract", "104.png"), (125, 100, 100))

    # 2. Diabetic Retinopathy: 4 patients -> 6 images
    _create_dummy_image(os.path.join(root, "diabetic_retinopathy", "kaggle_201_left.jpg"), (100, 120, 100))
    _create_dummy_image(os.path.join(root, "diabetic_retinopathy", "kaggle_201_right.jpg"), (100, 125, 100))
    _create_dummy_image(os.path.join(root, "diabetic_retinopathy", "kaggle_202_left.jpeg"), (100, 130, 100))
    _create_dummy_image(os.path.join(root, "diabetic_retinopathy", "kaggle_202_right.jpeg"), (100, 135, 100))
    _create_dummy_image(os.path.join(root, "diabetic_retinopathy", "203.png"), (100, 140, 100))
    _create_dummy_image(os.path.join(root, "diabetic_retinopathy", "204.png"), (100, 145, 100))

    # 3. Glaucoma: 4 patients -> 6 images
    _create_dummy_image(os.path.join(root, "glaucoma", "kaggle_301_left.jpg"), (100, 100, 120))
    _create_dummy_image(os.path.join(root, "glaucoma", "kaggle_301_right.jpg"), (100, 100, 125))
    _create_dummy_image(os.path.join(root, "glaucoma", "kaggle_302_left.jpeg"), (100, 100, 130))
    _create_dummy_image(os.path.join(root, "glaucoma", "kaggle_302_right.jpeg"), (100, 100, 135))
    _create_dummy_image(os.path.join(root, "glaucoma", "303.png"), (100, 100, 140))
    _create_dummy_image(os.path.join(root, "glaucoma", "304.png"), (100, 100, 145))

    # 4. Normal: 4 patients -> 6 images
    _create_dummy_image(os.path.join(root, "normal", "kaggle_401_left.jpg"), (120, 120, 120))
    _create_dummy_image(os.path.join(root, "normal", "kaggle_401_right.jpg"), (125, 125, 125))
    _create_dummy_image(os.path.join(root, "normal", "kaggle_402_left.jpeg"), (130, 130, 130))
    _create_dummy_image(os.path.join(root, "normal", "kaggle_402_right.jpeg"), (135, 135, 135))
    _create_dummy_image(os.path.join(root, "normal", "403.png"), (140, 140, 140))
    _create_dummy_image(os.path.join(root, "normal", "404.png"), (145, 145, 145))

    # Add an empty 0-byte file (should be flagged unreadable)
    with open(os.path.join(root, "cataract", "empty_corrupt.png"), "wb"):
        pass

    # Add a non-image text file (should be ignored)
    with open(os.path.join(root, "notes.txt"), "w") as f:
        f.write("ignore me")

    return root


class TestDatasetDiscoveryAndLabelMapping:
    """Verifies image discovery, format filtering, and label mapping."""

    def test_map_folder_to_class(self):
        assert map_folder_to_class("cataract") == "cataract"
        assert map_folder_to_class("Cataract") == "cataract"
        assert map_folder_to_class("diabetic_retinopathy") == "diabetic_retinopathy"
        assert map_folder_to_class("Diabetic Retinopathy") == "diabetic_retinopathy"
        assert map_folder_to_class("DR") == "diabetic_retinopathy"
        assert map_folder_to_class("glaucoma") == "glaucoma"
        assert map_folder_to_class("Glaucoma") == "glaucoma"
        assert map_folder_to_class("normal") == "normal"
        assert map_folder_to_class("Normal") == "normal"
        assert map_folder_to_class("unrelated_folder") is None

    def test_parse_patient_identifier(self):
        assert parse_patient_identifier("kaggle_1234_left.jpg") == "kaggle_1234"
        assert parse_patient_identifier("kaggle_1234_right.jpeg") == "kaggle_1234"
        assert parse_patient_identifier("55.png", parent_split="train") == "rfmid_train_55"
        assert parse_patient_identifier("55.png", parent_split="val") == "rfmid_val_55"

    def test_discover_dataset_finds_valid_and_excludes_empty(self, synthetic_dataset_dir):
        report = discover_and_validate_dataset(synthetic_dataset_dir, verify_image_integrity=True)

        assert report["total_images_found"] == 24
        assert report["usable_images_count"] == 24
        assert len(report["unreadable_files"]) == 1
        assert "empty_corrupt.png" in report["unreadable_files"][0]

        # Verify class counts
        for cls in CLASS_NAMES:
            assert report["class_counts"][cls] == 6


class TestDuplicateAndContradictionDetection:
    """Verifies that duplicate content is detected and cross-class contradictions are excluded."""

    def test_conflicting_duplicate_excluded(self, tmp_path):
        root = str(tmp_path / "conflict_dataset")
        for c in CLASS_NAMES:
            os.makedirs(os.path.join(root, c), exist_ok=True)

        # Create identical bytes in both cataract and glaucoma
        shared_bytes = np.full((50, 50, 3), 150, dtype=np.uint8)
        cv2.imwrite(os.path.join(root, "cataract", "dup1.png"), shared_bytes)
        cv2.imwrite(os.path.join(root, "glaucoma", "dup1.png"), shared_bytes)

        # Create one unique image in normal
        unique_bytes = np.full((50, 50, 3), 200, dtype=np.uint8)
        cv2.imwrite(os.path.join(root, "normal", "unique.png"), unique_bytes)

        report = discover_and_validate_dataset(root, verify_image_integrity=True)

        # 3 total files found, but the 2 conflicting files must be excluded from usable records
        assert report["total_images_found"] == 3
        assert report["usable_images_count"] == 1
        assert len(report["ambiguous_excluded_files"]) == 2
        assert len(report["records"]) == 1
        assert report["records"][0]["class_name"] == "normal"


class TestReproducibleSplitAndLeakagePrevention:
    """Verifies patient grouping, deterministic splitting, and zero data leakage."""

    def test_patient_isolation_zero_leakage(self, synthetic_dataset_dir):
        report = discover_and_validate_dataset(synthetic_dataset_dir, verify_image_integrity=False)
        splits = create_stratified_patient_split(
            report["records"],
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )

        train_pts = {r["patient_id"] for r in splits["train"]}
        val_pts = {r["patient_id"] for r in splits["val"]}
        test_pts = {r["patient_id"] for r in splits["test"]}

        # Strict patient isolation
        assert len(train_pts & val_pts) == 0, "Patient leakage between train and val!"
        assert len(train_pts & test_pts) == 0, "Patient leakage between train and test!"
        assert len(val_pts & test_pts) == 0, "Patient leakage between val and test!"

        # All records accounted for (no omissions)
        total_in_splits = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total_in_splits == len(report["records"])

    def test_reproducibility_with_seed(self, synthetic_dataset_dir):
        report = discover_and_validate_dataset(synthetic_dataset_dir, verify_image_integrity=False)

        splits_1 = create_stratified_patient_split(report["records"], random_seed=123)
        splits_2 = create_stratified_patient_split(report["records"], random_seed=123)

        assert [r["filepath"] for r in splits_1["train"]] == [r["filepath"] for r in splits_2["train"]]
        assert [r["filepath"] for r in splits_1["val"]] == [r["filepath"] for r in splits_2["val"]]
        assert [r["filepath"] for r in splits_1["test"]] == [r["filepath"] for r in splits_2["test"]]


class TestManifestPersistenceAndDataLoader:
    """Verifies manifest saving/loading and PyTorch dataset batch generation."""

    def test_save_and_load_manifest(self, synthetic_dataset_dir, tmp_path):
        report = discover_and_validate_dataset(synthetic_dataset_dir, verify_image_integrity=False)
        splits = create_stratified_patient_split(report["records"], random_seed=42)

        manifest_file = str(tmp_path / "manifest.json")
        saved_path = save_split_manifest(splits, manifest_file)
        assert os.path.exists(saved_path)

        loaded = load_split_manifest(saved_path)
        assert loaded["metadata"]["total_images"] == 24
        assert loaded["metadata"]["classes"] == CLASS_NAMES
        assert len(loaded["splits"]["train"]) == len(splits["train"])

    def test_eye_disease_dataset_and_dataloader(self, synthetic_dataset_dir, tmp_path):
        report = discover_and_validate_dataset(synthetic_dataset_dir, verify_image_integrity=False)
        splits = create_stratified_patient_split(report["records"], random_seed=42)
        manifest_file = str(tmp_path / "test_manifest.json")
        save_split_manifest(splits, manifest_file)

        # Instantiate dataset
        train_ds = EyeDiseaseDataset(manifest_file, split="train")
        assert len(train_ds) == len(splits["train"])

        # Fetch sample item
        tensor, label = train_ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert 0 <= label < NUM_CLASSES

        # DataLoader mini-batch test
        loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (4, 3, 224, 224)
        assert batch_y.shape == (4,)
        assert batch_x.dtype == torch.float32
