"""
Final ML Acceptance Test and Verified Deliverables Runner (Part 20).

Executes comprehensive end-to-end acceptance testing of the ML pipeline:
1. Production checkpoint loading, architecture validation, and rejection of invalid weights.
2. Quality gate verification on real test split images (pass vs reject).
3. Preprocessing and 4-class inference verification.
4. Output schema, probability validity, and sum-to-1 verification.
5. Grad-CAM visual explainability generation and Base64 PNG verification.
6. Independent physiological vitals triage analysis and type-safety verification.
7. End-to-end backend response serialization verification.
8. Held-out test split evaluation verification (894 images).
9. Generation of a persistent machine-readable acceptance report.
"""

import base64
from datetime import datetime, timezone
import io
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.dataset_loader import CLASS_NAMES, NUM_CLASSES, load_split_manifest
from ml.evaluate import evaluate_model_checkpoint
from ml.exceptions import (
    CheckpointCorruptError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    MLException,
    PhysiologicalValidationError,
    PhysiologicalTypeValidationError,
    QualityGateRejectionError,
)
from ml.eye_model import (
    CLASS_TO_IDX,
    DEFAULT_CHECKPOINT_PATH,
    IDX_TO_CLASS,
    build_densenet201_model,
    clear_model_cache,
    get_cached_model,
    get_default_device,
    load_eye_model_checkpoint,
    predict_eye_disease,
    screen_eye_image,
)
from ml.gradcam import generate_gradcam
from ml.monitoring import sanitize_path
from ml.physio_model import analyze_vitals
from ml.preprocessing import preprocess_image
from ml.quality_gate import check_image_quality

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("acceptance_test")

ACCEPTANCE_REPORT_PATH = os.path.join(
    os.path.dirname(__file__), "evaluation_results", "acceptance_report.json"
)


def run_acceptance_test(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    manifest_path: str = os.path.join(os.path.dirname(__file__), "dataset", "split_manifest.json"),
    report_output_path: str = ACCEPTANCE_REPORT_PATH,
) -> Dict[str, Any]:
    """
    Executes all acceptance checks using real artifacts and returns a full verified report.
    """
    print("=" * 80)
    print("PART 20: FINAL ML ACCEPTANCE TEST & VERIFIED DELIVERABLES")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    device = get_default_device()
    acceptance_results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "checkpoint_path": sanitize_path(checkpoint_path),
        "canonical_classes": CLASS_NAMES,
        "checks": {},
    }

    # --------------------------------------------------------------------------
    # CHECK 1: Verify Dataset & Split Manifest
    # --------------------------------------------------------------------------
    print("\n[CHECK 1] Inspecting Real Dataset & Split Manifest...")
    assert os.path.exists(manifest_path), f"Split manifest missing at {manifest_path}"
    manifest_data = load_split_manifest(manifest_path)
    meta = manifest_data["metadata"]
    total_imgs = meta["total_images"]
    train_cnt = meta["train_count"]
    val_cnt = meta["val_count"]
    test_cnt = meta["test_count"]
    class_dist = meta["class_distribution"]

    print(f"  • Total Registered Images: {total_imgs}")
    print(f"  • Splits: Train={train_cnt} (70.1%), Val={val_cnt} (14.7%), Test={test_cnt} (15.2%)")
    print(f"  • Class Distribution (Train/Val/Test):")
    for c in CLASS_NAMES:
        print(f"      - {c:<22}: {class_dist['train'][c]} / {class_dist['val'][c]} / {class_dist['test'][c]}")

    assert total_imgs == 5889, f"Expected 5,889 usable images, found {total_imgs}"
    assert test_cnt == 894, f"Expected 894 test images, found {test_cnt}"
    assert train_cnt + val_cnt + test_cnt == total_imgs
    acceptance_results["checks"]["dataset_manifest"] = {
        "status": "PASS",
        "total_images": total_imgs,
        "splits": {"train": train_cnt, "val": val_cnt, "test": test_cnt},
        "class_distribution": class_dist,
    }

    # --------------------------------------------------------------------------
    # CHECK 2: Trained Checkpoint Verification & Safe Loading
    # --------------------------------------------------------------------------
    print("\n[CHECK 2] Loading Trained Checkpoint via Production Loader...")
    assert os.path.exists(checkpoint_path), f"Checkpoint missing at {checkpoint_path}"
    ckpt_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    ckpt_epoch = ckpt_raw.get("epoch", 0)
    ckpt_val_acc = ckpt_raw.get("best_val_accuracy", ckpt_raw.get("best_acc", 0.0))
    ckpt_arch = ckpt_raw.get("architecture", "DenseNet201")
    ckpt_classes = ckpt_raw.get("classes", [])

    print(f"  • Checkpoint File: {os.path.basename(checkpoint_path)} ({os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB)")
    print(f"  • Architecture: {ckpt_arch}")
    print(f"  • Epoch Completed: {ckpt_epoch}")
    print(f"  • Best Validation Accuracy: {ckpt_val_acc * 100:.2f}%")
    print(f"  • Class Mapping: {ckpt_classes}")

    assert ckpt_classes == CLASS_NAMES, f"Class mismatch: {ckpt_classes} vs {CLASS_NAMES}"
    assert ckpt_arch == "DenseNet201", f"Architecture mismatch: {ckpt_arch}"

    # Load through production loader
    clear_model_cache()
    model = get_cached_model(weights_path=checkpoint_path, device=device)
    assert isinstance(model, nn.Module)
    assert not model.training, "Model must be in eval() mode"

    # Verify invalid checkpoint handling (no silent prediction)
    invalid_handled = False
    try:
        dummy_model = build_densenet201_model()
        load_eye_model_checkpoint(dummy_model, "non_existent_weights.pth")
    except (CheckpointNotFoundError, FileNotFoundError):
        invalid_handled = True
    assert invalid_handled, "Missing checkpoint must raise CheckpointNotFoundError"

    acceptance_results["checks"]["checkpoint_loading"] = {
        "status": "PASS",
        "epoch": ckpt_epoch,
        "best_val_accuracy": ckpt_val_acc,
        "architecture": ckpt_arch,
        "classes": ckpt_classes,
        "device": str(device),
        "invalid_checkpoint_protection": "VERIFIED",
    }

    # --------------------------------------------------------------------------
    # CHECK 3 & 4: Quality Gate, Preprocessing, and 4-Class Inference
    # --------------------------------------------------------------------------
    print("\n[CHECK 3 & 4] Running Real Image Quality Gate, Preprocessing & Inference...")
    test_records = manifest_data["splits"]["test"]

    # Locate one valid passing sample for each class
    test_samples_by_class: Dict[str, Dict[str, Any]] = {}
    for rec in test_records:
        cls = rec["class_name"]
        fp = rec["filepath"]
        if cls not in test_samples_by_class and os.path.exists(fp):
            q_res = check_image_quality(fp)
            if q_res["overall_pass"]:
                test_samples_by_class[cls] = {"record": rec, "quality": q_res}
        if len(test_samples_by_class) == NUM_CLASSES:
            break

    assert len(test_samples_by_class) == NUM_CLASSES, "Failed to find passing test image per class"

    inference_results: List[Dict[str, Any]] = []
    for cls_name in CLASS_NAMES:
        sample_info = test_samples_by_class[cls_name]
        raw_fp = sample_info["record"]["filepath"]
        anonymized_fp = sanitize_path(raw_fp)

        # Preprocess
        tensor = preprocess_image(raw_fp)
        assert tensor.shape == (1, 3, 224, 224), f"Bad tensor shape: {tensor.shape}"
        assert tensor.dtype == torch.float32

        # Inference
        pred_res = predict_eye_disease(tensor, model=model, device=device)
        pred = pred_res["prediction"]
        c_idx = pred_res["class_index"]
        conf = pred_res["confidence"]
        probs = pred_res["all_probabilities"]

        # Output correctness assertions
        assert pred in CLASS_NAMES, f"Prediction {pred} not in {CLASS_NAMES}"
        assert c_idx == CLASS_NAMES.index(pred), f"Index mismatch: {c_idx} vs {CLASS_NAMES.index(pred)}"
        assert abs(conf - probs[pred]) < 1e-6, f"Confidence {conf} != prob {probs[pred]}"
        assert len(probs) == 4, f"Expected 4 class probabilities, got {len(probs)}"
        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 0.002, f"Probabilities do not sum approximately to 1.0: {prob_sum}"
        for c, p in probs.items():
            assert 0.0 <= p <= 1.0, f"Probability {p} out of range [0, 1]"
            assert not np.isnan(p) and not np.isinf(p)

        print(f"  • Real Sample [{cls_name}]: GroundTruth={cls_name:<21} | Pred={pred:<21} | Conf={conf:.4f} | Sum={prob_sum:.4f}")
        inference_results.append({
            "ground_truth": cls_name,
            "sample": anonymized_fp,
            "prediction": pred,
            "class_index": c_idx,
            "confidence": conf,
            "all_probabilities": probs,
            "probabilities_sum": prob_sum,
        })

    acceptance_results["checks"]["inference_correctness"] = {
        "status": "PASS",
        "samples_evaluated": len(inference_results),
        "results": inference_results,
    }

    # --------------------------------------------------------------------------
    # CHECK 5: Quality Gate Rejection Blocks Inference
    # --------------------------------------------------------------------------
    print("\n[CHECK 5] Verifying Quality Rejection Blocks Classification...")
    # Find a real failing test image
    failing_sample_path: Optional[str] = None
    failing_reason: Optional[str] = None
    for rec in test_records:
        fp = rec["filepath"]
        if os.path.exists(fp):
            q_res = check_image_quality(fp)
            if not q_res["overall_pass"]:
                failing_sample_path = fp
                failing_reason = q_res.get("reason", "Quality check failed")
                break

    assert failing_sample_path is not None, "Failed to find quality failing test sample"
    screening_res = screen_eye_image(failing_sample_path, use_cache=True)
    print(f"  • Rejected Sample: {sanitize_path(failing_sample_path)}")
    print(f"  • Status: {screening_res['status']}")
    print(f"  • Reason: {screening_res.get('reason')}")
    assert screening_res["status"] == "rejected", "Expected status 'rejected'"
    assert "prediction" not in screening_res or screening_res.get("prediction") is None

    acceptance_results["checks"]["quality_rejection"] = {
        "status": "PASS",
        "sample": sanitize_path(failing_sample_path),
        "rejection_reason": failing_reason,
        "inference_blocked": True,
    }

    # --------------------------------------------------------------------------
    # CHECK 6: Grad-CAM Explainability Generation
    # --------------------------------------------------------------------------
    print("\n[CHECK 6] Generating Grad-CAM Visual Explainability...")
    first_passing_img = test_samples_by_class[CLASS_NAMES[0]]["record"]["filepath"]
    gradcam_b64 = generate_gradcam(first_passing_img, predicted_class_idx=0, model=model)
    assert isinstance(gradcam_b64, str) and len(gradcam_b64) > 100, "Grad-CAM must be non-empty base64 string"

    # Decode and verify valid PNG image
    img_bytes = base64.b64decode(gradcam_b64)
    pil_img = Image.open(io.BytesIO(img_bytes))
    assert pil_img.format == "PNG", f"Expected PNG format, got {pil_img.format}"
    print(f"  • Generated Grad-CAM Base64 String: {len(gradcam_b64)} chars")
    print(f"  • Decoded Format: {pil_img.format} | Dimensions: {pil_img.size}")

    acceptance_results["checks"]["gradcam"] = {
        "status": "PASS",
        "image_format": pil_img.format,
        "dimensions": list(pil_img.size),
        "base64_length_chars": len(gradcam_b64),
    }

    # --------------------------------------------------------------------------
    # CHECK 7: Independent Physiological Vitals Analysis
    # --------------------------------------------------------------------------
    print("\n[CHECK 7] Running Independent Physiological Vitals Analysis...")
    vitals_normal = analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=36.8)
    assert vitals_normal["overall_status"] == "Normal"
    assert vitals_normal["risk_level"] == "LOW"

    vitals_critical = analyze_vitals(heart_rate=135.0, spo2=88.0, temperature=38.9)
    assert vitals_critical["overall_status"] == "Abnormal"
    assert vitals_critical["risk_level"] == "HIGH"
    assert vitals_critical["is_abnormal"] is True

    # Type safety check: reject booleans
    bool_rejected = False
    try:
        analyze_vitals(heart_rate=True, spo2=98.0, temperature=36.8)  # type: ignore
    except (TypeError, ValueError, MLException):
        bool_rejected = True
    assert bool_rejected, "Boolean inputs must be rejected"

    print(f"  • Normal Case:    Status={vitals_normal['overall_status']} | Risk={vitals_normal['risk_level']}")
    print(f"  • Critical Case:  Status={vitals_critical['overall_status']} | Risk={vitals_critical['risk_level']}")
    print(f"  • Type Safety:    Boolean coercion rejected: {bool_rejected}")

    acceptance_results["checks"]["physiological_analysis"] = {
        "status": "PASS",
        "normal_case_risk": vitals_normal["risk_level"],
        "critical_case_risk": vitals_critical["risk_level"],
        "boolean_rejection_verified": bool_rejected,
    }

    # --------------------------------------------------------------------------
    # CHECK 8: End-to-End Backend Contract & JSON Serialization
    # --------------------------------------------------------------------------
    print("\n[CHECK 8] Testing Composite Screening Endpoint & Serialization...")
    composite_res = screen_eye_image(first_passing_img, include_gradcam=True, use_cache=True)
    assert composite_res["status"] == "success"
    assert composite_res["prediction"] in CLASS_NAMES
    assert "gradcam_image" in composite_res and len(composite_res["gradcam_image"]) > 100
    assert "quality_checks" in composite_res

    # Test full JSON serialization
    serialized_json = json.dumps(composite_res)
    assert len(serialized_json) > 100
    print(f"  • Composite Screening Status: {composite_res['status']}")
    print(f"  • JSON Serialization: OK ({len(serialized_json)} bytes)")

    acceptance_results["checks"]["backend_contract"] = {
        "status": "PASS",
        "composite_status": composite_res["status"],
        "json_serializable": True,
    }

    # --------------------------------------------------------------------------
    # CHECK 9: Held-Out Test Split Evaluation (Real Metrics)
    # --------------------------------------------------------------------------
    print("\n[CHECK 9] Evaluating Trained Checkpoint on Held-Out Test Split (N=894)...")
    eval_report = evaluate_model_checkpoint(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        batch_size=16,
        device=device,
        save_report=True,
    )

    metrics = eval_report["metrics"]
    acc = metrics["overall_accuracy"]
    macro_p = metrics["macro_precision"]
    macro_r = metrics["macro_recall"]
    macro_f1 = metrics["macro_f1"]
    weighted_f1 = metrics["weighted_f1"]
    per_class = metrics["per_class"]
    conf_mat = metrics["confusion_matrix"]
    eval_imgs = eval_report["evaluated_images"]
    failed_imgs = eval_report["failed_or_excluded_images"]

    print(f"  • Evaluated Images:      {eval_imgs} / {eval_report['total_test_images']}")
    print(f"  • Excluded / Failed:     {failed_imgs}")
    print(f"  • Overall Accuracy:      {acc * 100:.2f}%")
    print(f"  • Macro Precision:       {macro_p * 100:.2f}%")
    print(f"  • Macro Recall:          {macro_r * 100:.2f}%")
    print(f"  • Macro F1-Score:        {macro_f1 * 100:.2f}%")
    print(f"  • Weighted F1-Score:     {weighted_f1 * 100:.2f}%")
    print(f"  • Per-Class Metrics:")
    for c in CLASS_NAMES:
        pcm = per_class[c]
        print(f"      - {c:<22}: Precision={pcm['precision']*100:>5.2f}% | Recall={pcm['recall']*100:>5.2f}% | F1={pcm['f1_score']*100:>5.2f}% (Support: {pcm['support']})")

    assert eval_imgs == 894, f"Expected 894 evaluated images, got {eval_imgs}"
    assert failed_imgs == 0, f"Expected 0 failed images, got {failed_imgs}"
    assert acc >= 0.85, f"Accuracy {acc} fell below 85% acceptance baseline"

    acceptance_results["checks"]["held_out_evaluation"] = {
        "status": "PASS",
        "evaluated_images": eval_imgs,
        "failed_or_excluded_images": failed_imgs,
        "overall_accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": conf_mat,
    }

    # --------------------------------------------------------------------------
    # Final Verdict & Report Persistence
    # --------------------------------------------------------------------------
    all_passed = all(chk["status"] == "PASS" for chk in acceptance_results["checks"].values())
    acceptance_results["final_verdict"] = {
        "all_acceptance_checks_passed": all_passed,
        "ready_for_backend_integration": all_passed,
        "status": "READY_FOR_INTEGRATION" if all_passed else "NOT_READY",
        "medical_prototype_disclaimer": (
            "This system is an automated multimodal health screening prototype engineered for "
            "clinical triage research and decision support. It does not provide definitive medical "
            "diagnoses. All flagged findings must be reviewed by certified medical professionals."
        ),
    }

    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(acceptance_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"FINAL ACCEPTANCE VERDICT: {acceptance_results['final_verdict']['status']}")
    print(f"Acceptance Report Saved To: {report_output_path}")
    print("=" * 80)

    return acceptance_results


if __name__ == "__main__":
    run_acceptance_test()
