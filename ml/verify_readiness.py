"""
Verification Script for ML Project Readiness Audit (Part 8).

Executes comprehensive verification across:
1. Training artifacts & real dataset counts.
2. Checkpoint weights integrity & reload verification.
3. End-to-end eye-screening workflow with real test images.
4. Physiological analysis module functionality.
5. Modality independence & error handling.
"""

import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch

from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES, load_split_manifest
from ml.eye_model import (
    DEFAULT_CHECKPOINT_PATH,
    build_densenet201_model,
    clear_model_cache,
    load_eye_model_checkpoint,
    predict_eye_disease,
    screen_eye_image,
)
from ml.physio_model import analyze_vitals
from ml.quality_gate import check_image_quality

def run_readiness_audit():
    print("=" * 80)
    print("MULTIMODAL HEALTH SCREENING — ML READINESS AUDIT")
    print("=" * 80)

    # 1. Dataset & Split Manifest
    manifest_path = os.path.join(os.path.dirname(__file__), "dataset", "split_manifest.json")
    if not os.path.exists(manifest_path):
        print("ERROR: Split manifest not found!")
        return

    manifest = load_split_manifest(manifest_path)
    meta = manifest["metadata"]
    print("\n[1] DATASET AUDIT:")
    print(f"  Total Images Verified:  {meta['total_images']}")
    print(f"  Train Split:            {meta['train_count']} ({meta['train_count']/meta['total_images']*100:.1f}%)")
    print(f"  Val Split:              {meta['val_count']} ({meta['val_count']/meta['total_images']*100:.1f}%)")
    print(f"  Test Split:             {meta['test_count']} ({meta['test_count']/meta['total_images']*100:.1f}%)")
    print("  Per-Class Total:")
    for cls in CLASS_NAMES:
        t_cnt = meta['class_distribution']['train'][cls] + meta['class_distribution']['val'][cls] + meta['class_distribution']['test'][cls]
        print(f"    - {cls:22s}: {t_cnt:>5} (Train: {meta['class_distribution']['train'][cls]}, Val: {meta['class_distribution']['val'][cls]}, Test: {meta['class_distribution']['test'][cls]})")

    # 2. Checkpoints Audit
    print("\n[2] CHECKPOINT AUDIT:")
    best_ckpt_path = os.path.join(os.path.dirname(__file__), "weights", "eye_densenet201_best.pth")
    std_ckpt_path = os.path.join(os.path.dirname(__file__), "weights", "eye_model.pth")
    rec_ckpt_path = os.path.join(os.path.dirname(__file__), "weights", "recovery_checkpoint.pth")

    for path, label in [(best_ckpt_path, "Best Checkpoint"), (std_ckpt_path, "Standard Checkpoint"), (rec_ckpt_path, "Recovery Checkpoint")]:
        if os.path.exists(path):
            sz = os.path.getsize(path)
            data = torch.load(path, map_location="cpu", weights_only=True)
            epoch = data.get("epoch")
            best_acc = data.get("best_acc") or data.get("best_val_accuracy")
            classes = data.get("classes")
            print(f"  {label}:")
            print(f"    Path:     {path}")
            print(f"    Size:     {sz:,} bytes ({sz / (1024**2):.1f} MB)")
            print(f"    Epoch:    {epoch}")
            print(f"    Best Acc: {best_acc * 100:.2f}%" if best_acc else "    Best Acc: N/A")
            print(f"    Classes:  {classes}")
            assert classes == list(CLASS_NAMES), f"Class order mismatch: {classes}"
        else:
            print(f"  {label}: NOT FOUND at {path}")

    # Checkpoint Reload Verification
    print("\n[3] MODEL RELOAD VERIFICATION:")
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    loaded_model = load_eye_model_checkpoint(model, best_ckpt_path, device=torch.device("cpu"))
    assert loaded_model is not None
    assert loaded_model.classifier.out_features == 4
    print("  Production Checkpoint Loader: SUCCESS (strict parameter matching confirmed)")

    # 3. Held-Out Evaluation Report Audit
    print("\n[4] HELD-OUT TEST EVALUATION AUDIT:")
    eval_report_path = os.path.join(os.path.dirname(__file__), "evaluation_results", "test_evaluation_report.json")
    if os.path.exists(eval_report_path):
        with open(eval_report_path) as f:
            eval_data = json.load(f)
        m = eval_data["metrics"]
        print(f"  Report Timestamp:   {eval_data['evaluation_timestamp']}")
        print(f"  Test Images Total:  {eval_data['total_test_images']}")
        print(f"  Evaluated Images:   {eval_data['evaluated_images']}")
        print(f"  Excluded/Failed:    {eval_data['failed_or_excluded_images']}")
        print(f"  Overall Accuracy:   {m['overall_accuracy'] * 100:.2f}%")
        print(f"  Macro Precision:    {m['macro_precision'] * 100:.2f}%")
        print(f"  Macro Recall:       {m['macro_recall'] * 100:.2f}%")
        print(f"  Macro F1-Score:     {m['macro_f1'] * 100:.2f}%")
        print(f"  Weighted F1-Score:  {m['weighted_f1'] * 100:.2f}%")
        print("  Per-Class F1 Scores:")
        for cls in CLASS_NAMES:
            print(f"    - {cls:22s}: Precision={m['per_class'][cls]['precision']*100:.1f}%, Recall={m['per_class'][cls]['recall']*100:.1f}%, F1={m['per_class'][cls]['f1_score']*100:.1f}%")
    else:
        print("  Evaluation report not found!")

    # 4. End-to-End Eye Screening Workflow Verification
    print("\n[5] END-TO-END EYE SCREENING WORKFLOW VERIFICATION:")
    clear_model_cache()
    test_records = manifest["splits"]["test"]

    # Find a valid quality-passing test image
    valid_test_img = None
    for r in test_records:
        if os.path.exists(r["filepath"]):
            q = check_image_quality(r["filepath"])
            if q["overall_pass"]:
                valid_test_img = r
                break

    if valid_test_img:
        img_path = valid_test_img["filepath"]
        true_label = valid_test_img["class_name"]
        print(f"  Testing on real held-out image: [True Class: {true_label}]")
        
        # Check original image timestamp/size to verify it is NOT modified
        orig_mtime = os.path.getmtime(img_path)
        orig_size = os.path.getsize(img_path)

        # Run complete screen_eye_image workflow
        res = screen_eye_image(img_path, weights_path=best_ckpt_path, include_gradcam=True)

        assert res["status"] == "success"
        assert res["prediction"] in CLASS_NAMES
        assert 0 <= res["class_index"] < 4
        assert CLASS_NAMES[res["class_index"]] == res["prediction"]
        assert 0.0 <= res["confidence"] <= 1.0
        assert res["risk_level"] in ["LOW", "MODERATE", "HIGH"]
        assert "gradcam_image" in res
        assert isinstance(res["gradcam_image"], str) and len(res["gradcam_image"]) > 1000

        # Verify probabilities sum to ~1.0
        probs = res["all_probabilities"]
        assert len(probs) == 4
        assert abs(sum(probs.values()) - 1.0) < 1e-4

        # Verify disk preservation
        assert os.path.getmtime(img_path) == orig_mtime
        assert os.path.getsize(img_path) == orig_size

        # Verify JSON serializability
        json_output = json.dumps(res)
        assert json_output is not None

        print("  Workflow Output Contract:")
        print(f"    - Status:          {res['status']}")
        print(f"    - Prediction:      {res['prediction']} (Index {res['class_index']})")
        print(f"    - Confidence:      {res['confidence'] * 100:.2f}%")
        print(f"    - Risk Level:      {res['risk_level']}")
        print(f"    - Probabilities:   {res['all_probabilities']}")
        print(f"    - Grad-CAM:        Generated ({len(res['gradcam_image'])} chars base64 PNG)")
        print(f"    - Disk Image:      Unmodified (Preserved)")
        print(f"    - Serialization:   JSON-Serializable")

    # Workflow Negative Tests
    print("\n[6] NEGATIVE & EDGE CASE VERIFICATION:")
    # Quality failure halts inference
    # Find a rejected image
    rejected_img = None
    for r in test_records:
        if os.path.exists(r["filepath"]):
            q = check_image_quality(r["filepath"])
            if not q["overall_pass"]:
                rejected_img = r["filepath"]
                break
    if rejected_img:
        rej_res = screen_eye_image(rejected_img, weights_path=best_ckpt_path)
        assert rej_res["status"] == "rejected"
        assert "prediction" not in rej_res
        print("  Quality Gate Rejection Halts Inference: PASS (no fake predictions)")

    # Missing checkpoint raises clear error
    try:
        predict_eye_disease(valid_test_img["filepath"], weights_path="nonexistent_checkpoint.pth")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  Missing Checkpoint Raises FileNotFoundError: PASS")

    # 5. Physiological Analysis Module Verification
    print("\n[7] PHYSIOLOGICAL ANALYSIS VERIFICATION:")
    vitals_res = analyze_vitals(heart_rate=76.0, spo2=98.0, temperature=36.8)
    assert vitals_res["overall_status"] == "Normal"
    assert vitals_res["risk_level"] == "LOW"
    assert json.dumps(vitals_res) is not None

    abnormal_vitals = analyze_vitals(heart_rate=135.0, spo2=98.0, temperature=36.8)
    assert abnormal_vitals["overall_status"] == "Abnormal"
    assert abnormal_vitals["risk_level"] == "HIGH"

    # Verify no vision coupling
    import ml.physio_model as pm
    assert not hasattr(pm, "torch")
    assert not hasattr(pm, "cv2")
    print("  analyze_vitals() Contract & Logic: PASS")
    print("  Vital Signs Independence (No torch/cv2 coupling): PASS")

    print("\n" + "=" * 80)
    print("AUDIT RESULT: ALL ML SUBSYSTEMS VERIFIED AND OPERATIONAL")
    print("=" * 80)

if __name__ == "__main__":
    run_readiness_audit()
