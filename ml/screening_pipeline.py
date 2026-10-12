"""
Multimodal Health Screening Standalone Pipeline.

Orchestrates the end-to-end multimodal health screening workflow combining:
1. Retinal fundus image quality verification (check_image_quality)
2. Production image preprocessing (preprocess_image)
3. Four-class DenseNet201 deep learning inference (predict_eye_disease)
4. Class-aligned Grad-CAM saliency map visualization (generate_gradcam)
5. Independent physiological vital-sign assessment (analyze_vitals)

Architecture & Safety Principles:
- Complete Independence: Retinal screening and vital-sign screening are evaluated
  independently. An image defect does not prevent vital signs assessment, and invalid
  vital signs do not abort retinal screening.
- Hard Quality Gate: Images failing quality criteria (blur, resolution, illumination,
  retinal visibility) are strictly rejected and NEVER proceed to deep learning inference.
- Grad-CAM Fault Isolation: A failure during gradient backpropagation or visualization
  rendering does not invalidate or corrupt an otherwise successful eye disease inference.
- Production Security: File paths are sanitized; stack traces and internal secrets are
  never exposed in output dictionaries.
- Clinical Disclaimer: Explicitly framed as a clinical screening & decision-support
  prototype, not an autonomous diagnostic system.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import sys

# Ensure repository root is on sys.path for standalone script execution
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import uuid
from typing import Any, Dict, Optional, Union

import torch

from ml.exceptions import (
    CheckpointCorruptError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    PhysiologicalTypeValidationError,
    PhysiologicalValidationError,
)
from ml.config import get_default_checkpoint_path, resolve_device
from ml.eye_model import (
    get_cached_model,
    get_model,
    predict_eye_disease,
)
from ml.gradcam import generate_gradcam
from ml.monitoring import (
    ErrorCode,
    LatencyTimer,
    log_ml_event,
    measure_ml_latency,
    sanitize_path,
)
from ml.physio_model import analyze_vitals
from ml.preprocessing import preprocess_image
from ml.quality_gate import check_image_quality

logger = logging.getLogger(__name__)

SYSTEM_SCREENING_DISCLAIMER: str = (
    "This system is an automated multimodal health screening prototype engineered for "
    "clinical triage research and decision support. It does not provide definitive "
    "autonomous medical diagnoses. All flagged findings must be reviewed by qualified "
    "ophthalmic and medical practitioners."
)


def _aggregate_overall_risk(eye_risk: Optional[str], vitals_risk: Optional[str]) -> str:
    """Aggregates eye and vitals risk levels into an overall screening risk category."""
    risks = [r for r in (eye_risk, vitals_risk) if r and r != "UNKNOWN"]
    if not risks:
        return "UNKNOWN"
    if "HIGH" in risks:
        return "HIGH"
    if "MODERATE" in risks:
        return "MODERATE"
    if all(r == "LOW" for r in risks):
        return "LOW"
    return "UNKNOWN"


def run_screening(
    image_path: Optional[str] = None,
    heart_rate: Optional[float] = None,
    spo2: Optional[float] = None,
    temperature: Optional[float] = None,
    weights_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    include_gradcam: bool = True,
    use_cache: bool = True,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the complete multimodal health screening workflow.

    Workflow:
        1. Generates correlation request_id if not supplied.
        2. Executes retinal image quality gate, preprocessing, DenseNet201 inference,
           and class-aligned Grad-CAM visualization.
        3. Independently validates and assesses physiological vital signs.
        4. Compiles unified, structured screening report with aggregated risk triage.

    Args:
        image_path: Path to retinal fundus image file on disk. Optional.
        heart_rate: Heart rate in beats per minute (BPM). Optional.
        spo2: Blood oxygen saturation percentage (%). Optional.
        temperature: Body core temperature in Celsius (°C). Optional.
        weights_path: Path to DenseNet201 model checkpoint. Optional.
        device: Target torch device ('cpu', 'cuda', or torch.device). Optional.
        include_gradcam: Whether to generate Grad-CAM visualization for eligible eye predictions.
        use_cache: Whether to use cached model instance in memory.
        request_id: Correlation identifier for telemetry tracking.

    Returns:
        Dict[str, Any]: Consistent, JSON-serializable multimodal screening result.
    """
    req_id = request_id or str(uuid.uuid4())
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    with measure_ml_latency() as pipeline_timer:
        # Step 0: Emit Screening Start Telemetry Event
        log_ml_event(
            logger,
            event_type="SCREENING_PIPELINE",
            event_name="screening.started",
            stage="pipeline",
            operation="run_screening",
            status="START",
            request_id=req_id,
            details={
                "has_image": image_path is not None,
                "has_vitals": any(v is not None for v in (heart_rate, spo2, temperature)),
            },
        )

        stage_timings: Dict[str, Optional[float]] = {
            "quality_gate": None,
            "preprocessing": None,
            "eye_inference": None,
            "gradcam": None,
            "vitals_analysis": None,
        }

        eye_result: Dict[str, Any] = {
            "status": "skipped",
            "prediction": None,
            "class_index": None,
            "confidence": None,
            "risk_level": None,
            "all_probabilities": None,
            "quality_checks": None,
            "gradcam_image": None,
        }

        # -------------------------------------------------------------
        # STAGE A: Retinal Image Screening Pipeline
        # -------------------------------------------------------------
        if image_path is None:
            eye_result["status"] = "skipped"
            eye_result["reason"] = "No retinal image path provided."
        elif not isinstance(image_path, str) or not image_path.strip():
            eye_result["status"] = "error"
            eye_result["reason"] = "Image path must be a non-empty string."
        elif "\x00" in image_path:
            eye_result["status"] = "error"
            eye_result["reason"] = "Image path contains invalid null byte character."
        elif weights_path is not None and "\x00" in str(weights_path):
            eye_result["status"] = "error"
            eye_result["reason"] = "Checkpoint path contains invalid null byte character."
        elif os.path.isdir(image_path):
            clean_name = sanitize_path(image_path)
            eye_result["status"] = "error"
            eye_result["reason"] = f"Retinal image path points to a directory, not a file: {clean_name}"
            log_ml_event(
                logger,
                event_type="SCREENING_PIPELINE",
                event_name="input.image_is_directory",
                stage="quality_gate",
                operation="run_screening",
                status="INVALID_INPUT",
                error_code=ErrorCode.IMAGE_NOT_FOUND,
                request_id=req_id,
                details={"file": clean_name},
                level=logging.WARNING,
            )
        elif not os.path.isfile(image_path):
            clean_name = sanitize_path(str(image_path))
            eye_result["status"] = "error"
            eye_result["reason"] = f"Retinal image file not found: {clean_name}"
            log_ml_event(
                logger,
                event_type="SCREENING_PIPELINE",
                event_name="input.image_not_found",
                stage="quality_gate",
                operation="run_screening",
                status="IMAGE_NOT_FOUND",
                error_code=ErrorCode.IMAGE_NOT_FOUND,
                request_id=req_id,
                details={"file": clean_name},
                level=logging.WARNING,
            )
        else:
            # Step 1: Run Quality Gate Check
            qg_timer = LatencyTimer().start()
            quality_result = check_image_quality(image_path, request_id=req_id)
            stage_timings["quality_gate"] = qg_timer.stop()
            eye_result["quality_checks"] = quality_result

            if not quality_result.get("overall_pass", False):
                # Hard Quality Gate: REJECT immediately
                eye_result["status"] = "rejected"
                eye_result["reason"] = quality_result.get("error_message") or "Image failed quality gate validation."
                eye_result["risk_level"] = "UNKNOWN"
                log_ml_event(
                    logger,
                    event_type="SCREENING_PIPELINE",
                    event_name="quality_gate.rejected",
                    stage="quality_gate",
                    operation="run_screening",
                    status="QUALITY_REJECTED",
                    error_code=ErrorCode.QUALITY_REJECTED,
                    duration_ms=stage_timings["quality_gate"],
                    request_id=req_id,
                    details={
                        "file": sanitize_path(image_path),
                        "reason": eye_result["reason"],
                    },
                    level=logging.INFO,
                )
            else:
                # Step 2: Quality Passed -> Preprocess Image
                prep_timer = LatencyTimer().start()
                try:
                    tensor = preprocess_image(image_path, request_id=req_id)
                    stage_timings["preprocessing"] = prep_timer.stop()
                except Exception as prep_err:
                    stage_timings["preprocessing"] = prep_timer.stop()
                    eye_result["status"] = "error"
                    eye_result["reason"] = f"Image preprocessing failed: {prep_err}"
                    tensor = None
                    log_ml_event(
                        logger,
                        event_type="SCREENING_PIPELINE",
                        event_name="preprocessing.failed",
                        stage="preprocessing",
                        operation="run_screening",
                        status="FAILURE",
                        error_code=ErrorCode.PREPROCESSING_FAILED,
                        duration_ms=stage_timings["preprocessing"],
                        request_id=req_id,
                        details={"error": str(prep_err)},
                        level=logging.ERROR,
                    )

                if tensor is not None:
                    # Step 3: DenseNet201 Inference
                    resolved_device = resolve_device(device)
                    target_weights = weights_path or get_default_checkpoint_path()

                    inf_timer = LatencyTimer().start()
                    try:
                        inference = predict_eye_disease(
                            image_input=tensor,
                            weights_path=target_weights,
                            device=resolved_device,
                            use_cache=use_cache,
                            validate_quality=False,
                            request_id=req_id,
                        )
                        stage_timings["eye_inference"] = inf_timer.stop()
                        eye_result["status"] = "success"
                        eye_result["prediction"] = inference["prediction"]
                        eye_result["class_index"] = inference["class_index"]
                        eye_result["confidence"] = inference["confidence"]
                        eye_result["risk_level"] = inference["risk_level"]
                        eye_result["all_probabilities"] = inference["all_probabilities"]

                        # Step 4: Class-Aligned Grad-CAM Visualization
                        if include_gradcam:
                            gcam_timer = LatencyTimer().start()
                            try:
                                active_model = (
                                    get_cached_model(weights_path=target_weights, device=resolved_device, request_id=req_id)
                                    if use_cache
                                    else get_model(weights_path=target_weights, device=resolved_device, request_id=req_id)
                                )
                                gradcam_b64 = generate_gradcam(
                                    image_path=image_path,
                                    predicted_class_idx=inference["class_index"],
                                    model=active_model,
                                    device=resolved_device,
                                    input_tensor=tensor,
                                    request_id=req_id,
                                )
                                stage_timings["gradcam"] = gcam_timer.stop()
                                eye_result["gradcam_image"] = gradcam_b64
                            except Exception as gcam_err:
                                # Safe isolation: Grad-CAM failure does not corrupt inference
                                stage_timings["gradcam"] = gcam_timer.stop()
                                eye_result["gradcam_image"] = None
                                eye_result["gradcam_error"] = str(gcam_err)
                                log_ml_event(
                                    logger,
                                    event_type="GRADCAM_ISOLATION",
                                    event_name="gradcam.failed",
                                    stage="gradcam",
                                    operation="run_screening",
                                    status="WARNING",
                                    error_code=ErrorCode.GRADCAM_FAILED,
                                    duration_ms=stage_timings["gradcam"],
                                    request_id=req_id,
                                    details={"error": str(gcam_err)},
                                    level=logging.WARNING,
                                )

                    except CheckpointNotFoundError as ckpt_not_found:
                        stage_timings["eye_inference"] = inf_timer.stop()
                        eye_result["status"] = "error"
                        eye_result["reason"] = f"Model checkpoint unavailable: {ckpt_not_found}"
                        log_ml_event(
                            logger,
                            event_type="SCREENING_PIPELINE",
                            event_name="checkpoint.not_found",
                            stage="eye_model",
                            operation="run_screening",
                            status="FAILURE",
                            error_code=ErrorCode.CHECKPOINT_NOT_FOUND,
                            request_id=req_id,
                            details={"error": str(ckpt_not_found)},
                            level=logging.ERROR,
                        )
                    except (CheckpointCorruptError, CheckpointIncompatibleError) as ckpt_err:
                        stage_timings["eye_inference"] = inf_timer.stop()
                        eye_result["status"] = "error"
                        eye_result["reason"] = f"Invalid model checkpoint: {ckpt_err}"
                        log_ml_event(
                            logger,
                            event_type="SCREENING_PIPELINE",
                            event_name="checkpoint.invalid",
                            stage="eye_model",
                            operation="run_screening",
                            status="FAILURE",
                            error_code=ErrorCode.CHECKPOINT_CORRUPT,
                            request_id=req_id,
                            details={"error": str(ckpt_err)},
                            level=logging.ERROR,
                        )
                    except Exception as inf_err:
                        stage_timings["eye_inference"] = inf_timer.stop()
                        eye_result["status"] = "error"
                        eye_result["reason"] = f"Model inference error: {inf_err}"
                        log_ml_event(
                            logger,
                            event_type="SCREENING_PIPELINE",
                            event_name="eye_inference.failed",
                            stage="eye_model",
                            operation="run_screening",
                            status="FAILURE",
                            error_code=ErrorCode.INFERENCE_FAILED,
                            request_id=req_id,
                            details={"error": str(inf_err)},
                            level=logging.ERROR,
                        )

        # -------------------------------------------------------------
        # STAGE B: Physiological Vital-Sign Screening Pipeline
        # -------------------------------------------------------------
        vitals_result: Dict[str, Any] = {
            "status": "skipped",
            "heart_rate": heart_rate,
            "spo2": spo2,
            "temperature": temperature,
            "overall_status": None,
            "risk_level": None,
            "is_abnormal": None,
            "summary": None,
        }

        has_vitals = any(v is not None for v in (heart_rate, spo2, temperature))
        all_vitals = all(v is not None for v in (heart_rate, spo2, temperature))

        if not has_vitals:
            vitals_result["status"] = "skipped"
            vitals_result["reason"] = "No vital sign inputs provided."
        elif not all_vitals:
            vitals_result["status"] = "error"
            missing_fields = []
            if heart_rate is None:
                missing_fields.append("heart_rate")
            if spo2 is None:
                missing_fields.append("spo2")
            if temperature is None:
                missing_fields.append("temperature")
            vitals_result["reason"] = f"Incomplete vital signs; missing: {', '.join(missing_fields)}"
            vitals_result["risk_level"] = "UNKNOWN"
            log_ml_event(
                logger,
                event_type="SCREENING_PIPELINE",
                event_name="vitals.incomplete",
                stage="vitals",
                operation="run_screening",
                status="FAILURE",
                error_code=ErrorCode.VITALS_INCOMPLETE,
                request_id=req_id,
                details={"missing": missing_fields},
                level=logging.WARNING,
            )
        else:
            vitals_timer = LatencyTimer().start()
            try:
                vitals_analysis = analyze_vitals(
                    heart_rate=heart_rate,  # type: ignore
                    spo2=spo2,              # type: ignore
                    temperature=temperature, # type: ignore
                    raise_errors=True,
                    request_id=req_id,
                )
                stage_timings["vitals_analysis"] = vitals_timer.stop()
                vitals_result["status"] = "success"
                vitals_result["hr_status"] = vitals_analysis["hr_status"]
                vitals_result["spo2_status"] = vitals_analysis["spo2_status"]
                vitals_result["temp_status"] = vitals_analysis["temp_status"]
                vitals_result["heart_rate_status"] = vitals_analysis["hr_status"]
                vitals_result["temperature_status"] = vitals_analysis["temp_status"]
                vitals_result["overall_status"] = vitals_analysis["overall_status"]
                vitals_result["risk_level"] = vitals_analysis["risk_level"]
                vitals_result["is_abnormal"] = vitals_analysis["is_abnormal"]
                vitals_result["summary"] = vitals_analysis["summary"]
            except (PhysiologicalValidationError, PhysiologicalTypeValidationError) as val_err:
                stage_timings["vitals_analysis"] = vitals_timer.stop()
                vitals_result["status"] = "error"
                vitals_result["reason"] = str(val_err)
                vitals_result["risk_level"] = "UNKNOWN"
                log_ml_event(
                    logger,
                    event_type="SCREENING_PIPELINE",
                    event_name="vitals.validation_failed",
                    stage="vitals",
                    operation="run_screening",
                    status="FAILURE",
                    error_code=ErrorCode.VITALS_VALIDATION,
                    duration_ms=stage_timings["vitals_analysis"],
                    request_id=req_id,
                    details={"error": str(val_err)},
                    level=logging.WARNING,
                )
            except Exception as vitals_err:
                stage_timings["vitals_analysis"] = vitals_timer.stop()
                vitals_result["status"] = "error"
                vitals_result["reason"] = f"Vital sign analysis error: {vitals_err}"
                vitals_result["risk_level"] = "UNKNOWN"
                log_ml_event(
                    logger,
                    event_type="SCREENING_PIPELINE",
                    event_name="vitals.unexpected_error",
                    stage="vitals",
                    operation="run_screening",
                    status="FAILURE",
                    error_code=ErrorCode.UNEXPECTED_EXCEPTION,
                    duration_ms=stage_timings["vitals_analysis"],
                    request_id=req_id,
                    details={"error": str(vitals_err)},
                    level=logging.ERROR,
                )

        # -------------------------------------------------------------
        # STAGE C: Multi-Modal Aggregation & Overall Status
        # -------------------------------------------------------------
        pipeline_timer.stop()
        stage_timings["total_pipeline"] = round(pipeline_timer.duration_ms, 2)

        eye_status = eye_result["status"]
        vitals_status = vitals_result["status"]

        if eye_status == "success" and vitals_status == "success":
            overall_pipeline_status = "success"
        elif "success" in (eye_status, vitals_status):
            overall_pipeline_status = "partial"
        elif "rejected" in (eye_status, vitals_status):
            overall_pipeline_status = "rejected"
        elif eye_status == "skipped" and vitals_status == "skipped":
            overall_pipeline_status = "empty"
        else:
            overall_pipeline_status = "error"

        overall_risk = _aggregate_overall_risk(
            eye_result.get("risk_level"),
            vitals_result.get("risk_level"),
        )

        response: Dict[str, Any] = {
            "status": overall_pipeline_status,
            "screening_id": req_id,
            "timestamp": timestamp_utc,
            "duration_ms": round(pipeline_timer.duration_ms, 2),
            "timings_ms": stage_timings,
            "eye_screening": eye_result,
            "vitals_screening": vitals_result,
            "overall_risk_level": overall_risk,
            "disclaimer": SYSTEM_SCREENING_DISCLAIMER,
        }

        log_ml_event(
            logger,
            event_type="SCREENING_PIPELINE",
            event_name="screening.completed",
            stage="pipeline",
            operation="run_screening",
            status=overall_pipeline_status.upper(),
            duration_ms=pipeline_timer.duration_ms,
            request_id=req_id,
            details={
                "eye_status": eye_status,
                "vitals_status": vitals_status,
                "overall_risk": overall_risk,
                "overall_pipeline_status": overall_pipeline_status,
            },
            level=logging.INFO,
        )

        return response


def _parse_cli_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal Health Screening Standalone Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, default=None, help="Path to retinal fundus image")
    parser.add_argument("--hr", "--heart-rate", dest="heart_rate", type=float, default=None, help="Heart rate in BPM")
    parser.add_argument("--spo2", dest="spo2", type=float, default=None, help="SpO2 percentage (0-100)")
    parser.add_argument("--temp", "--temperature", dest="temperature", type=float, default=None, help="Core body temperature in deg C (e.g. 36.8)")
    parser.add_argument("--weights", type=str, default=None, help="Path to DenseNet201 checkpoint (.pth)")
    parser.add_argument("--device", type=str, default=None, help="Torch device ('cpu' or 'cuda')")
    parser.add_argument("--no-gradcam", action="store_true", help="Disable Grad-CAM heatmap generation")
    parser.add_argument("--output", type=str, default=None, help="Path to save output JSON result")
    parser.add_argument("--diagnose", "--doctor", dest="diagnose", action="store_true", help="Run ML environment & dependency diagnostics")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point for the standalone screening pipeline."""
    args = _parse_cli_args(argv)

    if getattr(args, "diagnose", False):
        from ml.diagnostics import main as diag_main
        return diag_main(argv=[])

    if args.image is None and args.heart_rate is None and args.spo2 is None and args.temperature is None:
        sys.stderr.write("Error: No inputs provided. Specify --image and/or vital signs (--hr, --spo2, --temp).\n")
        return 2

    try:
        result = run_screening(
            image_path=args.image,
            heart_rate=args.heart_rate,
            spo2=args.spo2,
            temperature=args.temperature,
            weights_path=args.weights,
            device=args.device,
            include_gradcam=not args.no_gradcam,
        )
    except Exception as exc:
        sys.stderr.write(f"Screening execution failure: {exc}\n")
        return 1

    formatted_json = json.dumps(result, indent=2)

    if args.output:
        try:
            from ml.security import safe_atomic_write_json, sanitize_error_message
            safe_atomic_write_json(args.output, result, disallow_critical=True)
            print(f"Results written to {args.output}")
        except (ValueError, PermissionError, IOError, OSError) as io_err:
            from ml.security import sanitize_error_message
            sanitized_err = sanitize_error_message(io_err)
            sys.stderr.write(f"Failed to write output to {sanitize_path(args.output)}: {sanitized_err}\n")
            return 1
    else:
        print(formatted_json)

    # Return exit code based on overall status
    if result["status"] in ("success", "partial", "rejected"):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
