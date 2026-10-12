"""
Automated Unit Tests for Data Integrity, Leakage Prevention, and Reproducibility (Part 11).

Validates:
1. Exact file duplicates across splits.
2. Duplicate filenames with identical content.
3. Conflicting labels / diagnostic contradictions.
4. Missing split assignments and missing disk files.
5. Invalid class labels and index mappings.
6. Patient-level data leakage across splits.
7. Deterministic, stable manifest hash computation.
8. Pipeline, architecture, and class mapping consistency.
9. Preprocessing configuration and normalization consistency.
10. Perceptual image hashing (dHash) and Hamming distance metrics.
"""

import json
import os
import tempfile
from unittest.mock import patch
import cv2
import numpy as np
import pytest
import torch

from ml.data_integrity import (
    audit_dataset_integrity,
    compute_canonical_manifest_hash,
    compute_hamming_distance,
    compute_image_dhash,
    verify_pipeline_consistency,
)
from ml.dataset_loader import CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES
from ml.preprocessing import IMAGENET_MEAN, IMAGENET_STD


@pytest.fixture
def clean_manifest_fixture(tmp_path):
    """Creates a temporary, leakage-free dataset and split manifest fixture."""
    manifest_path = str(tmp_path / "test_split_manifest.json")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}

    # Generate 12 clean synthetic images (3 per class, partitioned 8 train, 2 val, 2 test)
    sample_idx = 0
    for split_name, count in [("train", 8), ("val", 2), ("test", 2)]:
        for _ in range(count):
            cls_name = CLASS_NAMES[sample_idx % NUM_CLASSES]
            cls_idx = CLASS_TO_IDX[cls_name]
            patient_id = f"patient_{split_name}_{sample_idx}"
            filename = f"img_{sample_idx}.png"

            img_path = str(dataset_dir / filename)
            canvas = np.full((100, 100, 3), 100 + sample_idx, dtype=np.uint8)
            cv2.circle(canvas, (50, 50), 30, (30, 40, 50), -1)
            cv2.imwrite(img_path, canvas)

            with open(img_path, "rb") as f:
                import hashlib
                sha = hashlib.sha256(f.read()).hexdigest()

            splits[split_name].append({
                "filepath": img_path,
                "filename": filename,
                "class_name": cls_name,
                "class_index": cls_idx,
                "patient_id": patient_id,
                "sha256": sha,
                "split": split_name,
            })
            sample_idx += 1

    manifest_content = {
        "metadata": {
            "classes": list(CLASS_NAMES),
            "num_classes": NUM_CLASSES,
            "total_images": sample_idx,
        },
        "splits": splits,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_content, f, indent=2)

    return {
        "manifest_path": manifest_path,
        "dataset_dir": str(dataset_dir),
        "manifest_data": manifest_content,
    }


class TestManifestHashAndReproducibility:
    """Verifies deterministic manifest hashing."""

    def test_canonical_manifest_hash_deterministic(self, clean_manifest_fixture):
        manifest_data = clean_manifest_fixture["manifest_data"]
        h1 = compute_canonical_manifest_hash(manifest_data)
        h2 = compute_canonical_manifest_hash(manifest_data)
        assert len(h1) == 64
        assert h1 == h2

    def test_canonical_manifest_hash_sensitive_to_split_change(self, clean_manifest_fixture):
        manifest_data = clean_manifest_fixture["manifest_data"]
        h1 = compute_canonical_manifest_hash(manifest_data)

        # Alter a single record's class or split assignment
        mutated = json.loads(json.dumps(manifest_data))
        mutated["splits"]["train"][0]["class_name"] = "glaucoma"
        mutated["splits"]["train"][0]["class_index"] = 2
        h2 = compute_canonical_manifest_hash(mutated)

        assert h1 != h2

    def test_canonical_manifest_hash_insensitive_to_dict_ordering(self, clean_manifest_fixture):
        manifest_data = clean_manifest_fixture["manifest_data"]
        h1 = compute_canonical_manifest_hash(manifest_data)

        # Reverse key ordering in metadata
        reversed_meta = dict(reversed(list(manifest_data.items())))
        h2 = compute_canonical_manifest_hash(reversed_meta)

        assert h1 == h2


class TestLeakageDetection:
    """Verifies detection of cross-split duplicate content, patient overlap, and file reuse."""

    def test_clean_manifest_passes_audit(self, clean_manifest_fixture):
        report = audit_dataset_integrity(
            manifest_path=clean_manifest_fixture["manifest_path"],
            dataset_dir=clean_manifest_fixture["dataset_dir"],
        )
        assert report["data_integrity_verdict"]["is_clean_and_leakage_free"] is True
        assert report["cross_split_hash_leakage_count"] == 0
        assert report["patient_leakage_count"] == 0
        assert report["label_contradictions_count"] == 0
        assert report["missing_disk_files_count"] == 0

    def test_detect_cross_split_hash_leakage(self, clean_manifest_fixture, tmp_path):
        manifest_data = clean_manifest_fixture["manifest_data"]
        # Intentionally inject train image's hash into test set
        leaked_manifest = json.loads(json.dumps(manifest_data))
        train_rec = leaked_manifest["splits"]["train"][0]

        # Duplicate the same image file in test split with same content hash
        leaked_manifest["splits"]["test"].append({
            "filepath": train_rec["filepath"],
            "filename": "leaked_test.png",
            "class_name": train_rec["class_name"],
            "class_index": train_rec["class_index"],
            "patient_id": "different_patient",
            "sha256": train_rec["sha256"],  # Identical hash!
            "split": "test",
        })

        leaked_path = str(tmp_path / "leaked_manifest.json")
        with open(leaked_path, "w", encoding="utf-8") as f:
            json.dump(leaked_manifest, f)

        report = audit_dataset_integrity(manifest_path=leaked_path)
        assert report["cross_split_hash_leakage_count"] == 1
        assert report["data_integrity_verdict"]["is_clean_and_leakage_free"] is False

    def test_detect_patient_level_leakage(self, clean_manifest_fixture, tmp_path):
        manifest_data = clean_manifest_fixture["manifest_data"]
        leaked_manifest = json.loads(json.dumps(manifest_data))

        # Re-use a patient ID from train set in validation set
        train_patient = leaked_manifest["splits"]["train"][0]["patient_id"]
        leaked_manifest["splits"]["val"][0]["patient_id"] = train_patient

        leaked_path = str(tmp_path / "patient_leak_manifest.json")
        with open(leaked_path, "w", encoding="utf-8") as f:
            json.dump(leaked_manifest, f)

        report = audit_dataset_integrity(manifest_path=leaked_path)
        assert report["patient_leakage_count"] == 1
        assert report["patient_leakage_cases"][0]["patient_id"] == train_patient
        assert report["data_integrity_verdict"]["is_clean_and_leakage_free"] is False

    def test_detect_filepath_leakage(self, clean_manifest_fixture, tmp_path):
        manifest_data = clean_manifest_fixture["manifest_data"]
        leaked_manifest = json.loads(json.dumps(manifest_data))

        # Train filepath also assigned to test
        train_path = leaked_manifest["splits"]["train"][0]["filepath"]
        leaked_manifest["splits"]["test"][0]["filepath"] = train_path

        leaked_path = str(tmp_path / "filepath_leak_manifest.json")
        with open(leaked_path, "w", encoding="utf-8") as f:
            json.dump(leaked_manifest, f)

        report = audit_dataset_integrity(manifest_path=leaked_path)
        assert report["filepath_leakage"]["train_test_overlap"] == 1
        assert report["data_integrity_verdict"]["is_clean_and_leakage_free"] is False


class TestIntegrityAndLabelValidation:
    """Verifies detection of label contradictions, invalid labels, and missing disk files."""

    def test_detect_conflicting_labels_on_identical_hashes(self, clean_manifest_fixture, tmp_path):
        manifest_data = clean_manifest_fixture["manifest_data"]
        conflicted_manifest = json.loads(json.dumps(manifest_data))

        # Same SHA256 marked as cataract in train and glaucoma in test
        train_rec = conflicted_manifest["splits"]["train"][0]
        conflicted_manifest["splits"]["test"].append({
            "filepath": str(tmp_path / "conflicted_image.png"),
            "filename": "conflicted_image.png",
            "class_name": "glaucoma",  # Contradicts train_rec class!
            "class_index": 2,
            "patient_id": "another_patient",
            "sha256": train_rec["sha256"],
            "split": "test",
        })

        conflicted_path = str(tmp_path / "conflicted_manifest.json")
        with open(conflicted_path, "w", encoding="utf-8") as f:
            json.dump(conflicted_manifest, f)

        report = audit_dataset_integrity(manifest_path=conflicted_path)
        assert report["label_contradictions_count"] == 1
        assert report["data_integrity_verdict"]["is_clean_and_leakage_free"] is False

    def test_detect_invalid_class_labels(self, clean_manifest_fixture, tmp_path):
        manifest_data = clean_manifest_fixture["manifest_data"]
        invalid_manifest = json.loads(json.dumps(manifest_data))

        invalid_manifest["splits"]["train"][0]["class_name"] = "maculopathy"  # Invalid!
        invalid_manifest["splits"]["train"][0]["class_index"] = 99

        invalid_path = str(tmp_path / "invalid_manifest.json")
        with open(invalid_path, "w", encoding="utf-8") as f:
            json.dump(invalid_manifest, f)

        report = audit_dataset_integrity(manifest_path=invalid_path)
        assert report["invalid_labels_count"] == 1
        assert report["data_integrity_verdict"]["is_clean_and_leakage_free"] is False

    def test_detect_missing_disk_files(self, clean_manifest_fixture, tmp_path):
        manifest_data = clean_manifest_fixture["manifest_data"]
        missing_manifest = json.loads(json.dumps(manifest_data))

        missing_manifest["splits"]["train"][0]["filepath"] = str(tmp_path / "does_not_exist.png")

        missing_path = str(tmp_path / "missing_manifest.json")
        with open(missing_path, "w", encoding="utf-8") as f:
            json.dump(missing_manifest, f)

        report = audit_dataset_integrity(manifest_path=missing_path)
        assert report["missing_disk_files_count"] == 1
        assert report["data_integrity_verdict"]["is_clean_and_leakage_free"] is False


class TestPerceptualHashing:
    """Verifies image perceptual difference hashing and Hamming distance."""

    def test_compute_image_dhash_synthetic(self, tmp_path):
        img_path = str(tmp_path / "test_dhash.png")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, (150, 150, 150), -1)
        cv2.imwrite(img_path, img)

        dh = compute_image_dhash(img_path)
        assert dh is not None
        assert len(dh) == 16  # 64 bits = 16 hex characters

    def test_compute_image_dhash_missing_file_returns_none(self, tmp_path):
        missing = str(tmp_path / "nonexistent.png")
        assert compute_image_dhash(missing) is None

    def test_hamming_distance_metrics(self):
        h1 = "0000000000000000"
        h2 = "0000000000000001"  # 1 bit difference
        h3 = "ffffffffffffffff"  # 64 bits difference

        assert compute_hamming_distance(h1, h1) == 0
        assert compute_hamming_distance(h1, h2) == 1
        assert compute_hamming_distance(h1, h3) == 64


class TestPipelineConsistencyAudit:
    """Verifies that preprocessing constants and class mappings are consistent."""

    def test_verify_pipeline_consistency_baseline(self):
        res = verify_pipeline_consistency()
        assert res["consistent"] is True
        assert res["canonical_classes"] == ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
        assert res["imagenet_mean"] == [0.485, 0.456, 0.406]
        assert res["imagenet_std"] == [0.229, 0.224, 0.225]

    def test_verify_pipeline_consistency_detects_class_tampering(self):
        with patch("ml.data_integrity.CLASS_NAMES", ["cataract", "glaucoma", "normal"]):
            res = verify_pipeline_consistency()
            assert res["consistent"] is False
            assert any("mismatch" in iss.lower() for iss in res["issues"])
