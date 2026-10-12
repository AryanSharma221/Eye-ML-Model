"""
ML Service Layer for Multimodal Health Screening.
Encapsulates all direct interactions with the standalone ML layer:
- Pre-warming DenseNet201 model weights with latency monitoring
- Model caching, verification state, and device management
- Thread-safe inference and Grad-CAM generation with sub-stage timing
- Retinal image quality checks and preprocessing instrumentation
- Physiological vitals rule engine analysis with timing
- Combined multimodal screening execution with correlation tracing
"""

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import torch

from backend.config import DEFAULT_CHECKPOINT_PATH, ML_DEVICE
from ml.config import resolve_device
from ml.exceptions import (
    CheckpointCorruptError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    InvalidImageInputError,
    ModelInferenceError,
    PhysiologicalTypeValidationError,
    PhysiologicalValidationError,
    QualityGateRejectionError,
)
from ml.eye_model import (
    CLASS_NAMES,
    get_cached_model,
    get_default_device,
    predict_eye_disease,
    _MODEL_CACHE,
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
from ml.screening_pipeline import run_screening

logger = logging.getLogger("backend.services.ml_service")


class MLService:
    """
    Thread-safe ML service managing eye disease detection and vitals analysis
    with granular sub-stage observability and timing.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
        target_device = device or ML_DEVICE
        try:
            self.device = resolve_device(target_device)
        except Exception:
            self.device = get_default_device()
        self._gradcam_lock = threading.Lock()
        self.model_load_duration_ms: Optional[float] = None
        self.checkpoint_verified: bool = False
        self.checkpoint_error: Optional[str] = None

    def prewarm(self) -> bool:
        """
        Pre-warms the deep learning model into device memory.
        Measures loading duration and records verification status.
        """
        if not os.path.isfile(self.checkpoint_path):
            self.checkpoint_verified = False
            self.checkpoint_error = "CheckpointFileNotFound"
            logger.warning(
                f"Checkpoint not found at '{sanitize_path(self.checkpoint_path)}'. Pre-warm skipped."
            )
            return False

        load_timer = LatencyTimer().start()
        try:
            _ = get_cached_model(weights_path=self.checkpoint_path, device=self.device)
            self.model_load_duration_ms = load_timer.stop()
            self.checkpoint_verified = True
            self.checkpoint_error = None

            log_ml_event(
                logger,
                event_type="MODEL",
                event_name="model.prewarmed",
                stage="model_loading",
                operation="prewarm",
                status="SUCCESS",
                duration_ms=self.model_load_duration_ms,
                device=str(self.device),
                details={"verified": True},
            )
            logger.info(
                f"DenseNet201 model successfully pre-warmed on device: {self.device} "
                f"({self.model_load_duration_ms:.2f}ms)"
            )
            return True
        except Exception as exc:
            self.model_load_duration_ms = load_timer.stop()
            self.checkpoint_verified = False
            self.checkpoint_error = type(exc).__name__

            log_ml_event(
                logger,
                event_type="MODEL",
                event_name="model.prewarm_failed",
                stage="model_loading",
                operation="prewarm",
                status="FAILURE",
                error_code=ErrorCode.CHECKPOINT_CORRUPT if isinstance(exc, (CheckpointCorruptError, CheckpointIncompatibleError)) else ErrorCode.UNEXPECTED_EXCEPTION,
                duration_ms=self.model_load_duration_ms,
                device=str(self.device),
                details={"error_type": type(exc).__name__},
            )
            logger.error(f"Failed to pre-warm model from '{sanitize_path(self.checkpoint_path)}': {exc}")
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """
        Returns safe system health information without exposing local host paths.
        """
        checkpoint_available = os.path.isfile(self.checkpoint_path)
        model_loaded = len(_MODEL_CACHE) > 0
        if not model_loaded and checkpoint_available and not self.checkpoint_verified:
            # Ensure model is pre-warmed on demand if not triggered by lifespan context
            self.prewarm()
            model_loaded = len(_MODEL_CACHE) > 0

        is_healthy = bool(checkpoint_available and (model_loaded or self.checkpoint_verified))

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "model_loaded": model_loaded,
            "device": str(self.device),
            "checkpoint_available": checkpoint_available,
            "checkpoint_verified": self.checkpoint_verified or model_loaded,
            "checkpoint_error": self.checkpoint_error,
            "model_load_duration_ms": self.model_load_duration_ms,
            "supported_classes": CLASS_NAMES,
        }

    def predict_eye(
        self,
        image_path: str,
        include_gradcam: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes retinal image quality gate, preprocessing, inference, and Grad-CAM
        with granular latency tracking for every stage.

        Returns:
            Dict containing prediction results or quality gate rejection details.
        """
        total_timer = LatencyTimer().start()
        stage_timings: Dict[str, Optional[float]] = {
            "quality_gate": None,
            "preprocessing": None,
            "eye_inference": None,
            "gradcam": None,
            "total": None,
        }

        # 1. Quality Gate Check
        qg_timer = LatencyTimer().start()
        quality_result = check_image_quality(image_path, request_id=request_id)
        stage_timings["quality_gate"] = qg_timer.stop()

        if not quality_result.get("overall_pass", False):
            stage_timings["total"] = total_timer.stop()
            log_ml_event(
                logger,
                event_type="QUALITY_GATE",
                event_name="quality_gate.rejected",
                stage="quality_gate",
                operation="predict_eye",
                status="QUALITY_REJECTED",
                error_code=ErrorCode.QUALITY_REJECTED,
                duration_ms=stage_timings["quality_gate"],
                request_id=request_id,
                details={
                    "reason": quality_result.get("error_message"),
                    "file": sanitize_path(image_path),
                },
            )
            return {
                "status": "rejected",
                "overall_pass": False,
                "reason": quality_result.get("error_message", "Image failed quality gate validation."),
                "quality_checks": quality_result,
                "timings_ms": stage_timings,
            }

        log_ml_event(
            logger,
            event_type="QUALITY_GATE",
            event_name="quality_gate.passed",
            stage="quality_gate",
            operation="predict_eye",
            status="SUCCESS",
            duration_ms=stage_timings["quality_gate"],
            request_id=request_id,
        )

        # 2. Preprocessing
        prep_timer = LatencyTimer().start()
        try:
            tensor = preprocess_image(image_path, request_id=request_id)
            stage_timings["preprocessing"] = prep_timer.stop()
        except Exception as prep_err:
            stage_timings["preprocessing"] = prep_timer.stop()
            stage_timings["total"] = total_timer.stop()
            log_ml_event(
                logger,
                event_type="PREPROCESSING",
                event_name="preprocessing.failed",
                stage="preprocessing",
                operation="predict_eye",
                status="FAILURE",
                error_code=ErrorCode.PREPROCESSING_FAILED,
                duration_ms=stage_timings["preprocessing"],
                request_id=request_id,
            )
            raise

        # 3. Model Inference
        inf_timer = LatencyTimer().start()
        inference = predict_eye_disease(
            image_input=tensor,
            weights_path=self.checkpoint_path,
            device=self.device,
            use_cache=True,
            validate_quality=False,
            request_id=request_id,
        )
        stage_timings["eye_inference"] = inf_timer.stop()

        # 4. Grad-CAM Generation (Thread-safe)
        gradcam_b64 = None
        if include_gradcam:
            gcam_timer = LatencyTimer().start()
            try:
                cached_model = get_cached_model(weights_path=self.checkpoint_path, device=self.device)
                with self._gradcam_lock:
                    gradcam_b64 = generate_gradcam(
                        image_path=image_path,
                        predicted_class_idx=inference["class_index"],
                        model=cached_model,
                        device=self.device,
                        input_tensor=tensor,
                        request_id=request_id,
                    )
                stage_timings["gradcam"] = gcam_timer.stop()
            except Exception as gcam_err:
                stage_timings["gradcam"] = gcam_timer.stop()
                log_ml_event(
                    logger,
                    event_type="GRADCAM_ISOLATION",
                    event_name="gradcam.failed",
                    stage="gradcam",
                    operation="predict_eye",
                    status="WARNING",
                    error_code=ErrorCode.GRADCAM_FAILED,
                    duration_ms=stage_timings["gradcam"],
                    request_id=request_id,
                    details={"error": str(gcam_err)},
                )
                logger.warning(f"Grad-CAM generation non-critical error: {gcam_err}")
                gradcam_b64 = None

        stage_timings["total"] = total_timer.stop()

        log_ml_event(
            logger,
            event_type="INFERENCE",
            event_name="predict_eye.completed",
            stage="pipeline",
            operation="predict_eye",
            status="SUCCESS",
            duration_ms=stage_timings["total"],
            device=str(self.device),
            request_id=request_id,
            details={
                "prediction": inference["prediction"],
                "confidence": inference["confidence"],
                "risk_level": inference["risk_level"],
            },
        )

        return {
            "status": "success",
            "overall_pass": True,
            "prediction": inference["prediction"],
            "class_index": inference["class_index"],
            "confidence": inference["confidence"],
            "risk_level": inference["risk_level"],
            "all_probabilities": inference["all_probabilities"],
            "quality_checks": quality_result,
            "gradcam_image": gradcam_b64,
            "timings_ms": stage_timings,
            "disclaimer": (
                "Screening prototype output for clinical decision support. Not a standalone diagnosis."
            ),
        }

    def analyze_vitals(
        self,
        heart_rate: float,
        spo2: float,
        temperature: float,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Invokes the physiological analysis module with latency measurement.
        """
        timer = LatencyTimer().start()
        try:
            result = analyze_vitals(
                heart_rate=heart_rate,
                spo2=spo2,
                temperature=temperature,
                raise_errors=True,
                request_id=request_id,
            )
            duration = timer.stop()
            result["timings_ms"] = {"vitals_analysis": duration}

            log_ml_event(
                logger,
                event_type="PHYSIOLOGICAL",
                event_name="vitals.analyzed",
                stage="vitals",
                operation="analyze_vitals",
                status="SUCCESS",
                duration_ms=duration,
                request_id=request_id,
                details={
                    "overall_status": result.get("overall_status"),
                    "risk_level": result.get("risk_level"),
                },
            )
            return result
        except (PhysiologicalValidationError, PhysiologicalTypeValidationError) as val_err:
            duration = timer.stop()
            log_ml_event(
                logger,
                event_type="PHYSIOLOGICAL",
                event_name="vitals.validation_failed",
                stage="vitals",
                operation="analyze_vitals",
                status="WARNING",
                error_code=ErrorCode.VITALS_VALIDATION,
                duration_ms=duration,
                request_id=request_id,
                details={"error": str(val_err)},
            )
            raise

    def run_multimodal(
        self,
        image_path: Optional[str] = None,
        heart_rate: Optional[float] = None,
        spo2: Optional[float] = None,
        temperature: Optional[float] = None,
        include_gradcam: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes unified multimodal health screening pipeline with correlation tracing.
        """
        return run_screening(
            image_path=image_path,
            heart_rate=heart_rate,
            spo2=spo2,
            temperature=temperature,
            include_gradcam=include_gradcam,
            use_cache=True,
            request_id=request_id,
        )


# Global singleton instance
ml_service = MLService()
