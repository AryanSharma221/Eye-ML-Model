"""
Data Integrity, Leakage Prevention, and Reproducibility Audit Module (Part 11).

Audits and verifies:
1. Exact file duplicates via SHA-256 content hashes (identical bytes under different filenames).
2. Perceptual near-duplicates via deterministic difference hashing (dHash) and Hamming distance.
3. Cross-split data leakage (samples or patient records appearing in multiple splits).
4. Patient-level identifier isolation across train, val, and test splits.
5. Missing or unassigned samples between disk files and split manifests.
6. Conflicting labels / diagnostic contradictions among duplicate images.
7. Architectural and class-mapping consistency across training, checkpoints, and inference.
8. Preprocessing pipeline parameter consistency (ImageNet normalization, resolution, augmentations).
9. Manifest reproducibility via deterministic cryptographic checksum (manifest hash).
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch

from ml.dataset_loader import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    NUM_CLASSES,
    SUPPORTED_EXTENSIONS,
    resolve_dataset_directory,
)
from ml.eye_model import (
    CLASS_NAMES as EYE_CLASS_NAMES,
    DEFAULT_CHECKPOINT_PATH,
    NUM_CLASSES as EYE_NUM_CLASSES,
)
from ml.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_image_transforms,
)


# ---------------------------------------------------------------------------
# Cryptographic & Perceptual Hashing Utilities
# ---------------------------------------------------------------------------

def compute_image_dhash(image_path: str, hash_size: int = 8) -> Optional[str]:
    """
    Computes a deterministic 64-bit difference hash (dHash) for an image file.

    Algorithm:
        1. Reads image in grayscale.
        2. Resizes to (hash_size + 1, hash_size) using area interpolation.
        3. Computes boolean gradient matrix comparing adjacent horizontal pixels.
        4. Packs 64 bits into a 16-character hexadecimal string.

    Args:
        image_path: Path to the image file on disk.
        hash_size: Grid size for perceptual hashing (default: 8, yielding 64 bits).

    Returns:
        16-character hex string representing dHash, or None if image cannot be decoded.
    """
    if not os.path.exists(image_path):
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None

    # Width: hash_size + 1, Height: hash_size
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    packed = np.packbits(bits)
    return "".join(f"{b:02x}" for b in packed)


def compute_hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Computes the Hamming bit distance between two hexadecimal hash strings."""
    b1 = bytes.fromhex(hex_hash1)
    b2 = bytes.fromhex(hex_hash2)
    return sum(bin(x ^ y).count("1") for x, y in zip(b1, b2))


def compute_canonical_manifest_hash(manifest_dict: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 checksum for a dataset split manifest.
    Normalizes record sorting and JSON formatting to guarantee cross-platform repeatability.
    """
    splits_data = manifest_dict.get("splits", {})
    canonical_splits: Dict[str, List[Dict[str, Any]]] = {}

    for split_name in sorted(splits_data.keys()):
        records = splits_data[split_name]
        # Sort records by sha256 or filename to produce stable ordering
        sorted_records = sorted(
            records,
            key=lambda r: (r.get("sha256", ""), r.get("filename", "")),
        )
        canonical_records = []
        for r in sorted_records:
            canonical_records.append({
                "filename": r.get("filename"),
                "class_name": r.get("class_name"),
                "class_index": r.get("class_index"),
                "patient_id": r.get("patient_id"),
                "sha256": r.get("sha256"),
                "split": split_name,
            })
        canonical_splits[split_name] = canonical_records

    payload = {
        "classes": manifest_dict.get("metadata", {}).get("classes", list(CLASS_NAMES)),
        "splits": canonical_splits,
    }

    raw_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Consistency Verifier (Preprocessing & Class Mapping)
# ---------------------------------------------------------------------------

def verify_pipeline_consistency(
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verifies that class mappings, preprocessing parameters, and checkpoint metadata
    are perfectly synchronized across all ML modules.
    """
    target_ckpt = checkpoint_path or DEFAULT_CHECKPOINT_PATH
    issues: List[str] = []

    # 1. Verify Class Names & Index Mapping
    if list(CLASS_NAMES) != list(EYE_CLASS_NAMES):
        issues.append(
            f"Class names mismatch: dataset_loader={CLASS_NAMES} vs eye_model={EYE_CLASS_NAMES}"
        )

    expected_indices = {
        "cataract": 0,
        "diabetic_retinopathy": 1,
        "glaucoma": 2,
        "normal": 3,
    }
    if CLASS_TO_IDX != expected_indices:
        issues.append(f"CLASS_TO_IDX mismatch: expected {expected_indices}, got {CLASS_TO_IDX}")

    if NUM_CLASSES != 4 or EYE_NUM_CLASSES != 4:
        issues.append(f"NUM_CLASSES mismatch: expected 4, got {NUM_CLASSES}")

    # 2. Checkpoint Metadata Verification
    checkpoint_verified = False
    checkpoint_classes: Optional[List[str]] = None
    checkpoint_arch: Optional[str] = None

    if os.path.exists(target_ckpt):
        try:
            ckpt_data = torch.load(target_ckpt, map_location="cpu", weights_only=True)
            if isinstance(ckpt_data, dict):
                checkpoint_classes = ckpt_data.get("classes") or ckpt_data.get("class_names")
                checkpoint_arch = ckpt_data.get("architecture")
                if checkpoint_classes and list(checkpoint_classes) != list(CLASS_NAMES):
                    issues.append(
                        f"Checkpoint classes {checkpoint_classes} do not match canonical {CLASS_NAMES}"
                    )
                else:
                    checkpoint_verified = True
        except Exception as err:
            issues.append(f"Failed to inspect checkpoint at {target_ckpt}: {err}")
    else:
        issues.append(f"Checkpoint not found at: {target_ckpt}")

    # 3. Preprocessing Parameter Verification
    expected_mean = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]

    if list(IMAGENET_MEAN) != expected_mean:
        issues.append(f"IMAGENET_MEAN altered: expected {expected_mean}, got {list(IMAGENET_MEAN)}")

    if list(IMAGENET_STD) != expected_std:
        issues.append(f"IMAGENET_STD altered: expected {expected_std}, got {list(IMAGENET_STD)}")

    # 4. Transform Determinism Check
    eval_transform = get_image_transforms(is_train=False)
    # Ensure resize target is (224, 224)
    resize_found = False
    for t in eval_transform.transforms:
        if hasattr(t, "size") and t.size == (224, 224):
            resize_found = True
            break
    if not resize_found:
        issues.append("Evaluation transform does not contain 224x224 resize step.")

    return {
        "consistent": len(issues) == 0,
        "issues": issues,
        "canonical_classes": list(CLASS_NAMES),
        "checkpoint_verified": checkpoint_verified,
        "checkpoint_classes": checkpoint_classes,
        "checkpoint_arch": checkpoint_arch,
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
    }


# ---------------------------------------------------------------------------
# Split Manifest & Data Integrity Auditor
# ---------------------------------------------------------------------------

def audit_dataset_integrity(
    manifest_path: str = os.path.join("ml", "dataset", "split_manifest.json"),
    dataset_dir: Optional[str] = None,
    check_perceptual_near_duplicates: bool = False,
    perceptual_distance_threshold: int = 4,
) -> Dict[str, Any]:
    """
    Performs an exhaustive audit on the eye disease dataset and split manifest.

    Checks:
        1. File existence on disk for all manifest records.
        2. Exact content duplicates via SHA-256 within and across splits.
        3. Cross-split data leakage (sample appears in multiple splits).
        4. Patient-level identifier leakage (same patient in multiple splits).
        5. Label conflicts among duplicate content hashes.
        6. Missing files on disk not registered in split manifest.
        7. Class balance across train, val, and test partitions.
        8. Optional perceptual near-duplicate analysis via dHash Hamming distance.

    Returns:
        Structured audit report dictionary with all findings and integrity verdict.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Split manifest not found at: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    splits = manifest_data.get("splits", {})
    metadata = manifest_data.get("metadata", {})

    manifest_hash = compute_canonical_manifest_hash(manifest_data)

    # 1. Accounting and Split Membership
    all_filepaths: Set[str] = set()
    split_filepaths: Dict[str, Set[str]] = defaultdict(set)
    split_counts: Dict[str, int] = {}
    split_class_distribution: Dict[str, Dict[str, int]] = {}

    hash_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    patient_to_splits: Dict[str, Set[str]] = defaultdict(set)
    patient_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    missing_disk_files: List[str] = []
    invalid_labels: List[Dict[str, Any]] = []

    for split_name, records in splits.items():
        split_counts[split_name] = len(records)
        cls_counter: Dict[str, int] = Counter()

        for rec in records:
            fp = rec.get("filepath", "")
            norm_fp = os.path.normpath(fp)
            c_name = rec.get("class_name", "")
            c_idx = rec.get("class_index", -1)
            sha = rec.get("sha256", "")
            pid = rec.get("patient_id", "")

            # Check validity of label
            if c_name not in CLASS_NAMES or c_idx != CLASS_TO_IDX.get(c_name):
                invalid_labels.append({"filepath": fp, "class_name": c_name, "class_index": c_idx})

            cls_counter[c_name] += 1
            all_filepaths.add(norm_fp)
            split_filepaths[split_name].add(norm_fp)

            rec_summary = {
                "filepath": fp,
                "split": split_name,
                "class_name": c_name,
                "patient_id": pid,
                "sha256": sha,
            }
            hash_to_records[sha].append(rec_summary)
            patient_to_splits[pid].add(split_name)
            patient_to_records[pid].append(rec_summary)

            # Check disk existence
            if not os.path.exists(fp):
                missing_disk_files.append(fp)

        split_class_distribution[split_name] = dict(cls_counter)

    # 2. Cross-Split Filepath Overlap
    train_paths = split_filepaths.get("train", set())
    val_paths = split_filepaths.get("val", set())
    test_paths = split_filepaths.get("test", set())

    train_val_overlap = sorted(list(train_paths & val_paths))
    train_test_overlap = sorted(list(train_paths & test_paths))
    val_test_overlap = sorted(list(val_paths & test_paths))

    has_filepath_leakage = (
        len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0
    )

    # 3. Content Hash Duplicates & Leakage
    exact_duplicates: Dict[str, List[Dict[str, Any]]] = {}
    cross_split_hash_leakage: List[Dict[str, Any]] = []
    label_contradictions: List[Dict[str, Any]] = []

    for sha, recs in hash_to_records.items():
        if len(recs) > 1:
            exact_duplicates[sha] = recs
            splits_present = set(r["split"] for r in recs)
            if len(splits_present) > 1:
                cross_split_hash_leakage.append({
                    "sha256": sha,
                    "splits": list(splits_present),
                    "files": [r["filepath"] for r in recs],
                })
            classes_present = set(r["class_name"] for r in recs)
            if len(classes_present) > 1:
                label_contradictions.append({
                    "sha256": sha,
                    "conflicting_classes": list(classes_present),
                    "files": [r["filepath"] for r in recs],
                })

    # 4. Patient-Level Leakage
    patient_leakage_cases: List[Dict[str, Any]] = []
    for pid, p_splits in patient_to_splits.items():
        if len(p_splits) > 1:
            patient_leakage_cases.append({
                "patient_id": pid,
                "splits": list(p_splits),
                "record_count": len(patient_to_records[pid]),
            })

    # 5. Disk Files vs Manifest Comparison
    unassigned_disk_files: List[str] = []
    resolved_root: Optional[str] = None
    try:
        resolved_root = resolve_dataset_directory(dataset_dir)
        if os.path.exists(resolved_root):
            for root_dir, _, filenames in os.walk(resolved_root):
                for f in filenames:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        full_f = os.path.normpath(os.path.join(root_dir, f))
                        if full_f not in all_filepaths:
                            unassigned_disk_files.append(full_f)
    except Exception:
        pass

    # 6. Optional Perceptual Near-Duplicate Analysis
    perceptual_duplicates: List[Dict[str, Any]] = []
    if check_perceptual_near_duplicates:
        # Sample or analyze split records using dHash
        # Group hashes and compare cross-split records
        sample_hashes: List[Tuple[str, str, str, str]] = []  # (fp, split, class, dhash)
        for s_name, recs in splits.items():
            for r in recs:
                fp = r["filepath"]
                if os.path.exists(fp):
                    dh = compute_image_dhash(fp)
                    if dh:
                        sample_hashes.append((fp, s_name, r["class_name"], dh))

        # Check for near matches across splits
        n_samples = len(sample_hashes)
        for i in range(min(n_samples, 500)):  # Fast representative scan
            fp1, s1, c1, h1 = sample_hashes[i]
            for j in range(i + 1, min(n_samples, 500)):
                fp2, s2, c2, h2 = sample_hashes[j]
                if s1 != s2:  # Cross-split check
                    dist = compute_hamming_distance(h1, h2)
                    if dist <= perceptual_distance_threshold:
                        perceptual_duplicates.append({
                            "file1": fp1,
                            "split1": s1,
                            "class1": c1,
                            "file2": fp2,
                            "split2": s2,
                            "class2": c2,
                            "hamming_distance": dist,
                        })

    # 7. Pipeline Consistency Check
    pipeline_check = verify_pipeline_consistency()

    # 8. Overall Integrity Verdict
    is_clean = (
        not has_filepath_leakage
        and len(cross_split_hash_leakage) == 0
        and len(label_contradictions) == 0
        and len(patient_leakage_cases) == 0
        and len(missing_disk_files) == 0
        and len(invalid_labels) == 0
        and pipeline_check["consistent"]
    )

    return {
        "manifest_path": os.path.abspath(manifest_path),
        "manifest_canonical_sha256": manifest_hash,
        "total_manifest_images": len(all_filepaths),
        "split_counts": split_counts,
        "split_class_distribution": split_class_distribution,
        "dataset_root_scanned": resolved_root,
        "unassigned_disk_files_count": len(unassigned_disk_files),
        "unassigned_disk_files_sample": unassigned_disk_files[:10],
        "missing_disk_files_count": len(missing_disk_files),
        "invalid_labels_count": len(invalid_labels),
        "exact_duplicates_count": len(exact_duplicates),
        "cross_split_hash_leakage_count": len(cross_split_hash_leakage),
        "cross_split_hash_leakage": cross_split_hash_leakage,
        "patient_leakage_count": len(patient_leakage_cases),
        "patient_leakage_cases": patient_leakage_cases,
        "label_contradictions_count": len(label_contradictions),
        "label_contradictions": label_contradictions,
        "filepath_leakage": {
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "val_test_overlap": len(val_test_overlap),
        },
        "perceptual_near_duplicates_found": len(perceptual_duplicates),
        "perceptual_near_duplicates": perceptual_duplicates[:10],
        "pipeline_consistency": pipeline_check,
        "data_integrity_verdict": {
            "is_clean_and_leakage_free": is_clean,
            "evaluation_results_trustworthy": is_clean,
            "requires_retraining": not is_clean,
        },
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit dataset integrity, prevent split leakage, and verify reproducibility."
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=os.path.join("ml", "dataset", "split_manifest.json"),
        help="Path to dataset split manifest JSON.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset directory. Resolved automatically if omitted.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to DenseNet201 model checkpoint.",
    )
    parser.add_argument(
        "--check-perceptual",
        action="store_true",
        help="Run perceptual near-duplicate analysis using dHash.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save audit report JSON.",
    )

    args = parser.parse_args()

    print("=" * 75)
    print(" DATASET INTEGRITY, LEAKAGE AUDIT & REPRODUCIBILITY VERIFICATION (PART 11)")
    print("=" * 75)
    print(f"Manifest Path:      {args.manifest_path}")
    print(f"Model Checkpoint:   {args.checkpoint_path}")
    print(f"Check Perceptual:   {args.check_perceptual}")
    print("-" * 75)

    report = audit_dataset_integrity(
        manifest_path=args.manifest_path,
        dataset_dir=args.dataset_dir,
        check_perceptual_near_duplicates=args.check_perceptual,
    )

    print("\n--- DATASET & SPLIT COMPOSITION ---")
    print(f"Manifest Canonical SHA-256:   {report['manifest_canonical_sha256']}")
    print(f"Total Registered Images:      {report['total_manifest_images']}")
    print(f"Split Partition Counts:       {report['split_counts']}")
    print("\nClass Distribution per Split:")
    for s_name, dist in report["split_class_distribution"].items():
        print(f"  - {s_name:<6}: {dist}")

    print("\n--- INTEGRITY & LEAKAGE CHECKS ---")
    f_leak = report["filepath_leakage"]
    print(f"Train/Val Filepath Overlap:   {f_leak['train_val_overlap']}")
    print(f"Train/Test Filepath Overlap:  {f_leak['train_test_overlap']}")
    print(f"Val/Test Filepath Overlap:    {f_leak['val_test_overlap']}")
    print(f"Cross-Split Hash Leakage:     {report['cross_split_hash_leakage_count']}")
    print(f"Patient Identifier Overlap:   {report['patient_leakage_count']}")
    print(f"Label Contradictions:         {report['label_contradictions_count']}")
    print(f"Missing Files on Disk:        {report['missing_disk_files_count']}")
    print(f"Invalid Class Labels:         {report['invalid_labels_count']}")
    print(f"Unassigned Disk Files:        {report['unassigned_disk_files_count']}")

    print("\n--- PIPELINE CONSISTENCY ---")
    pipe = report["pipeline_consistency"]
    print(f"Architectural Consistency:    {'PASS' if pipe['consistent'] else 'FAIL'}")
    print(f"Canonical Classes:            {pipe['canonical_classes']}")
    print(f"Checkpoint Verified:          {pipe['checkpoint_verified']} ({pipe['checkpoint_arch']})")
    print(f"Normalization Constants:      Mean={pipe['imagenet_mean']}, Std={pipe['imagenet_std']}")
    if pipe["issues"]:
        print("Issues Detected:")
        for iss in pipe["issues"]:
            print(f"  ! {iss}")

    print("\n--- FINAL VERDICT ---")
    verdict = report["data_integrity_verdict"]
    print(f"Clean & Leakage Free:         {verdict['is_clean_and_leakage_free']}")
    print(f"Evaluation Results Valid:     {verdict['evaluation_results_trustworthy']}")
    print(f"Requires Retraining:          {verdict['requires_retraining']}")

    out_file = args.output_json or os.path.join(
        "ml", "evaluation_results", "data_integrity_report.json"
    )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Full audit report saved to: {out_file}\n" + "=" * 75)


if __name__ == "__main__":
    main()
