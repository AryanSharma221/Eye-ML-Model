"""
Dataset Discovery, Label Validation, Patient Grouping, and Split Management Module.

Provides utilities for:
1. Discovering retinal fundus images across standard formats (.jpg, .jpeg, .png, .bmp).
2. Validating labels against the canonical 4 classes:
   0: cataract, 1: diabetic_retinopathy, 2: glaucoma, 3: normal.
3. Detecting corrupt images, duplicate content hashes, and label contradictions.
4. Parsing patient/subject identifiers to prevent patient data leakage across splits.
5. Generating reproducible, patient-stratified train/val/test splits (70/15/15 ratio).
6. Persisting and loading deterministic JSON split manifests.
7. Providing a PyTorch-compatible EyeDiseaseDataset for training and evaluation.

Public Contracts:
    CLASS_NAMES: list[str]
    NUM_CLASSES: int
    CLASS_TO_IDX: dict[str, int]
    IDX_TO_CLASS: dict[int, str]
    discover_and_validate_dataset(dataset_dir: Optional[str]) -> dict
    create_stratified_patient_split(records: list[dict], ...) -> dict
    save_split_manifest(split_records: dict, manifest_path: str) -> str
    load_split_manifest(manifest_path: str) -> dict
    EyeDiseaseDataset(torch.utils.data.Dataset)
"""

import hashlib
import json
import os
import random
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from ml.preprocessing import get_image_transforms

# Canonical 4-class ordering matching eye_model.py
CLASS_NAMES: List[str] = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal",
]
NUM_CLASSES: int = len(CLASS_NAMES)
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Supported fundus image formats
SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}

# Default dataset discovery roots
DEFAULT_PROJECT_DATASET_DIR: str = os.path.join(
    os.path.dirname(__file__), "dataset"
)
EXTERNAL_SCRATCH_DATASET_DIR: str = os.environ.get(
    "EXTERNAL_SCRATCH_DATASET_DIR",
    os.path.normpath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "Custom_Eye_Disease_Model", "dataset"
        )
    ),
)


def resolve_dataset_directory(custom_path: Optional[str] = None) -> str:
    """
    Resolves the active dataset root directory, prioritizing:
    1. custom_path parameter (if provided and exists)
    2. ML_DATASET_DIR or EYE_DATASET_DIR environment variables
    3. Project local ml/dataset directory (if contains images)
    4. External verified scratch dataset directory (if exists)
    """
    if custom_path and os.path.exists(custom_path):
        return os.path.abspath(custom_path)

    env_dir = os.environ.get("ML_DATASET_DIR", os.environ.get("EYE_DATASET_DIR"))
    if env_dir and os.path.exists(env_dir):
        return os.path.abspath(env_dir)

    # Check project local ml/dataset (ensure it contains actual dataset images, >= 50)
    if os.path.exists(DEFAULT_PROJECT_DATASET_DIR):
        img_count = 0
        for _, _, files in os.walk(DEFAULT_PROJECT_DATASET_DIR):
            for f in files:
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                    img_count += 1
                    if img_count >= 50:
                        return os.path.abspath(DEFAULT_PROJECT_DATASET_DIR)

    # Check external scratch directory containing 5,300+ images (if available)
    if os.path.exists(EXTERNAL_SCRATCH_DATASET_DIR):
        return os.path.abspath(EXTERNAL_SCRATCH_DATASET_DIR)

    return os.path.abspath(DEFAULT_PROJECT_DATASET_DIR)


# Regex for Kaggle patient naming: e.g. kaggle_1234_left.jpg or kaggle_1234_right.jpg
KAGGLE_PATIENT_REGEX = re.compile(
    r"^kaggle_(\d+)_(left|right)\.(jpg|jpeg|png|bmp)$", re.IGNORECASE
)
# Regex for RFMiD image IDs: e.g. 101.png
RFMID_REGEX = re.compile(r"^(\d+)\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)


def parse_patient_identifier(filename: str, parent_split: str = "") -> str:
    """
    Extracts a patient/subject identifier from a retinal image filename.

    - Kaggle images: 'kaggle_1002_left.jpg' -> 'kaggle_1002'
    - RFMiD images: '101.png' in train split -> 'rfmid_train_101'
    - Other images: 'img_patientX_1.png' -> 'patientX' or fallback to filename
    """
    clean_name = os.path.basename(filename)

    km = KAGGLE_PATIENT_REGEX.match(clean_name)
    if km:
        return f"kaggle_{km.group(1)}"

    rm = RFMID_REGEX.match(clean_name)
    if rm:
        prefix = f"rfmid_{parent_split}_" if parent_split else "rfmid_"
        return f"{prefix}{rm.group(1)}"

    # Generic patient token extraction: e.g. patient_123, subject_05
    generic_match = re.search(r"(patient|subject)[_-]?([a-zA-Z0-9]+)", clean_name, re.IGNORECASE)
    if generic_match:
        return f"{generic_match.group(1).lower()}_{generic_match.group(2)}"

    # Default to filename stem if no pattern matched
    stem = os.path.splitext(clean_name)[0]
    return f"id_{stem}"


def map_folder_to_class(folder_name: str) -> Optional[str]:
    """
    Maps folder names to canonical class names (case-insensitive).
    """
    cleaned = folder_name.strip().lower().replace(" ", "_").replace("-", "_")
    for canonical in CLASS_NAMES:
        if cleaned == canonical:
            return canonical

    # Handle common variations
    if cleaned in {"dr", "diabetic", "diabeticretinopathy"}:
        return "diabetic_retinopathy"
    if cleaned in {"cat", "cataracts"}:
        return "cataract"
    if cleaned in {"glauc", "glaucomatous"}:
        return "glaucoma"
    if cleaned in {"norm", "healthy", "non_disease"}:
        return "normal"

    return None


def compute_file_sha256(filepath: str) -> str:
    """Calculates SHA-256 checksum of an image file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def discover_and_validate_dataset(
    dataset_dir: Optional[str] = None,
    verify_image_integrity: bool = True,
) -> Dict[str, Any]:
    """
    Discovers, validates, and audits all fundus image files in the dataset directory.

    Validates:
        1. Class folder label mapping against the 4 canonical classes.
        2. Image decodability via OpenCV (flags corrupt or truncated images).
        3. Exact SHA-256 content hashes (flags duplicate images).
        4. Cross-class label conflicts (identical image in two different class folders).
        5. Patient grouping derivation.

    Args:
        dataset_dir: Root dataset folder. Resolved automatically if None.
        verify_image_integrity: If True, attempts to decode each image with OpenCV.

    Returns:
        dict containing:
            - root_dir (str)
            - total_images_found (int)
            - usable_images_count (int)
            - class_counts (dict[str, int])
            - unreadable_files (list[str])
            - duplicate_pairs (list[dict])
            - ambiguous_excluded_files (list[dict])
            - valid_records (list[dict])
    """
    root = resolve_dataset_directory(dataset_dir)
    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

    valid_records: List[Dict[str, Any]] = []
    unreadable_files: List[str] = []
    hash_to_paths: Dict[str, List[Tuple[str, str]]] = {}
    class_counts: Dict[str, int] = {c: 0 for c in CLASS_NAMES}

    for dirpath, _, filenames in os.walk(root):
        rel_path = os.path.relpath(dirpath, root)
        parts = rel_path.split(os.sep)

        # Determine class from folder path
        detected_class: Optional[str] = None
        for part in reversed(parts):
            mapped = map_folder_to_class(part)
            if mapped:
                detected_class = mapped
                break

        if not detected_class:
            continue

        parent_split = parts[0] if parts and parts[0] in {"train", "val", "test"} else ""

        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            full_path = os.path.join(dirpath, f)

            # Check readability and file size
            if os.path.getsize(full_path) == 0:
                unreadable_files.append(full_path)
                continue

            if verify_image_integrity:
                img = cv2.imread(full_path, cv2.IMREAD_COLOR)
                if img is None or img.size == 0:
                    unreadable_files.append(full_path)
                    continue

            # Compute content hash
            sha = compute_file_sha256(full_path)
            patient_id = parse_patient_identifier(f, parent_split)

            if sha not in hash_to_paths:
                hash_to_paths[sha] = []
            hash_to_paths[sha].append((full_path, detected_class))

            valid_records.append(
                {
                    "filepath": full_path,
                    "filename": f,
                    "class_name": detected_class,
                    "class_index": CLASS_TO_IDX[detected_class],
                    "patient_id": patient_id,
                    "sha256": sha,
                }
            )

    # Detect duplicate pairs and conflicting labels
    duplicate_pairs: List[Dict[str, Any]] = []
    ambiguous_excluded_files: List[Dict[str, Any]] = []
    ambiguous_hashes: Set[str] = set()

    for sha, paths_classes in hash_to_paths.items():
        if len(paths_classes) > 1:
            classes_assigned = {cls for _, cls in paths_classes}
            paths = [p for p, _ in paths_classes]
            duplicate_pairs.append(
                {
                    "sha256": sha,
                    "count": len(paths_classes),
                    "classes": list(classes_assigned),
                    "paths": paths,
                }
            )
            # Conflicting labels across classes
            if len(classes_assigned) > 1:
                ambiguous_hashes.add(sha)
                for p, cls in paths_classes:
                    ambiguous_excluded_files.append(
                        {
                            "filepath": p,
                            "class_name": cls,
                            "reason": f"Contradictory label duplicate: identical image exists in {classes_assigned}",
                            "sha256": sha,
                        }
                    )

    # Filter out ambiguous conflicting duplicates from usable records
    usable_records: List[Dict[str, Any]] = []
    for r in valid_records:
        if r["sha256"] in ambiguous_hashes:
            continue
        usable_records.append(r)
        class_counts[r["class_name"]] += 1

    return {
        "root_dir": root,
        "total_images_found": len(valid_records),
        "usable_images_count": len(usable_records),
        "class_counts": class_counts,
        "unreadable_files": unreadable_files,
        "duplicate_pairs": duplicate_pairs,
        "ambiguous_excluded_files": ambiguous_excluded_files,
        "records": usable_records,
    }


def create_stratified_patient_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Creates a reproducible train / validation / test split grouped by patient/subject identifier.

    Guarantees:
        1. Patient Isolation: All images belonging to a single patient_id are allocated
           to the EXACT SAME split. Zero patient leakage across train, val, and test.
        2. Stratification: Preserves class balance across all splits according to target ratios.
        3. Determinism: Controlled via fixed random_seed (default: 42).
        4. Completeness: Every input record is assigned to exactly one split (0 omissions).

    Args:
        records: List of validated image record dictionaries.
        train_ratio: Proportion for training set (default: 0.70).
        val_ratio: Proportion for validation set (default: 0.15).
        test_ratio: Proportion for test set (default: 0.15).
        random_seed: Seed for deterministic patient assignment.

    Returns:
        dict: {"train": list[dict], "val": list[dict], "test": list[dict]}
    """
    if not (0.99 <= train_ratio + val_ratio + test_ratio <= 1.01):
        raise ValueError(
            f"Split ratios must sum to 1.0. Got {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}."
        )

    # 1. Group records by patient_id
    patient_to_records: Dict[str, List[Dict[str, Any]]] = {}
    patient_to_primary_class: Dict[str, str] = {}

    for r in records:
        pid = r["patient_id"]
        if pid not in patient_to_records:
            patient_to_records[pid] = []
        patient_to_records[pid].append(r)

    # Determine primary class per patient (majority vote or first)
    for pid, p_records in patient_to_records.items():
        classes = [rec["class_name"] for rec in p_records]
        # Most frequent class for this patient
        primary = max(set(classes), key=classes.count)
        patient_to_primary_class[pid] = primary

    # 2. Stratified patient grouping per class
    class_to_patients: Dict[str, List[str]] = {c: [] for c in CLASS_NAMES}
    for pid, primary_cls in patient_to_primary_class.items():
        if primary_cls in class_to_patients:
            class_to_patients[primary_cls].append(pid)
        else:
            class_to_patients[CLASS_NAMES[0]].append(pid)

    # 3. Deterministic split per class group
    rng = random.Random(random_seed)
    train_patients: Set[str] = set()
    val_patients: Set[str] = set()
    test_patients: Set[str] = set()

    for cls in CLASS_NAMES:
        p_list = sorted(class_to_patients[cls])  # Sort for cross-platform stability
        rng.shuffle(p_list)

        n = len(p_list)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        # Ensure at least 1 patient in val and test if group is large enough
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = max(1, min(n_val, n - n_train - 1))

        train_pts = p_list[:n_train]
        val_pts = p_list[n_train : n_train + n_val]
        test_pts = p_list[n_train + n_val :]

        train_patients.update(train_pts)
        val_patients.update(val_pts)
        test_patients.update(test_pts)

    # 4. Assign records to splits
    split_dict: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for r in records:
        pid = r["patient_id"]
        record_copy = dict(r)
        if pid in train_patients:
            record_copy["split"] = "train"
            split_dict["train"].append(record_copy)
        elif pid in val_patients:
            record_copy["split"] = "val"
            split_dict["val"].append(record_copy)
        elif pid in test_patients:
            record_copy["split"] = "test"
            split_dict["test"].append(record_copy)
        else:
            # Fallback to train to prevent any record omission
            record_copy["split"] = "train"
            split_dict["train"].append(record_copy)

    # Verification: Assert zero patient leakage across splits
    actual_train_pts = {r["patient_id"] for r in split_dict["train"]}
    actual_val_pts = {r["patient_id"] for r in split_dict["val"]}
    actual_test_pts = {r["patient_id"] for r in split_dict["test"]}

    leak_train_val = actual_train_pts & actual_val_pts
    leak_train_test = actual_train_pts & actual_test_pts
    leak_val_test = actual_val_pts & actual_test_pts

    if leak_train_val or leak_train_test or leak_val_test:
        raise RuntimeError(
            f"Patient leakage detected! Train-Val: {len(leak_train_val)}, "
            f"Train-Test: {len(leak_train_test)}, Val-Test: {len(leak_val_test)}."
        )

    # Verification: Total count matching
    total_assigned = len(split_dict["train"]) + len(split_dict["val"]) + len(split_dict["test"])
    if total_assigned != len(records):
        raise RuntimeError(
            f"Record count mismatch: input {len(records)} != split total {total_assigned}."
        )

    return split_dict


def save_split_manifest(
    split_dict: Dict[str, List[Dict[str, Any]]],
    manifest_path: str,
) -> str:
    """
    Saves the partitioned records to a structured JSON manifest file on disk.
    """
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)

    def _count_classes(records: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {c: 0 for c in CLASS_NAMES}
        for r in records:
            counts[r["class_name"]] = counts.get(r["class_name"], 0) + 1
        return counts

    manifest_payload = {
        "metadata": {
            "classes": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "total_images": sum(len(records) for records in split_dict.values()),
            "train_count": len(split_dict["train"]),
            "val_count": len(split_dict["val"]),
            "test_count": len(split_dict["test"]),
            "class_distribution": {
                "train": _count_classes(split_dict["train"]),
                "val": _count_classes(split_dict["val"]),
                "test": _count_classes(split_dict["test"]),
            },
        },
        "splits": split_dict,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    return os.path.abspath(manifest_path)


def load_split_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Reads a split manifest JSON file from disk.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Split manifest file not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return manifest


class EyeDiseaseDataset(Dataset):
    """
    PyTorch Dataset for retinal fundus eye disease classification.

    Loads from pre-partitioned split records or a saved JSON manifest.
    Applies standard ImageNet normalization and data augmentation.
    """

    def __init__(
        self,
        records_or_manifest: Union[List[Dict[str, Any]], str],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        verify_images: bool = False,
    ):
        """
        Args:
            records_or_manifest: List of record dicts or path to manifest JSON.
            split: 'train', 'val', or 'test' (required if passing manifest path).
            transform: Optional torchvision transform. If None, uses project defaults.
            verify_images: If True, validates readability of all images on initialization.
        """
        self.split = split
        if isinstance(records_or_manifest, str):
            manifest = load_split_manifest(records_or_manifest)
            if not split or split not in manifest.get("splits", {}):
                raise ValueError(
                    f"Must specify a valid split ({list(manifest.get('splits', {}).keys())}) when loading from manifest."
                )
            self.records = manifest["splits"][split]
        elif isinstance(records_or_manifest, list):
            self.records = records_or_manifest
        else:
            raise TypeError(
                f"Expected records_or_manifest as list or str path, got {type(records_or_manifest).__name__}."
            )

        if transform is not None:
            self.transform = transform
        else:
            is_train = split == "train"
            self.transform = get_image_transforms(is_train=is_train)

        if verify_images:
            valid_recs = []
            for r in self.records:
                resolved = self._resolve_image_path(r)
                if os.path.isfile(resolved):
                    r["filepath"] = resolved
                    valid_recs.append(r)
            self.records = valid_recs

    def _resolve_image_path(self, record: Dict[str, Any]) -> str:
        """Resolves record filepath, falling back to active dataset root if path was generated on another machine."""
        fp = record.get("filepath", "")
        if fp and os.path.isfile(fp):
            return fp

        root = resolve_dataset_directory()
        fn = record.get("filename") or os.path.basename(fp)
        cls_name = record.get("class_name", "")
        split = record.get("split", "")

        candidates = [
            os.path.join(root, split, cls_name, fn),
            os.path.join(root, cls_name, fn),
            os.path.join(root, split, fn),
            os.path.join(root, fn),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

        return fp

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        record = self.records[idx]
        image_path = self._resolve_image_path(record)
        class_idx = int(record["class_index"])

        try:
            pil_img = Image.open(image_path).convert("RGB")
        except Exception as err:
            raise IOError(f"Unable to read image at {image_path}: {err}") from err

        tensor = self.transform(pil_img)
        return tensor, class_idx

    def get_class_counts(self) -> Dict[str, int]:
        counts = {c: 0 for c in CLASS_NAMES}
        for r in self.records:
            counts[r["class_name"]] = counts.get(r["class_name"], 0) + 1
        return counts
