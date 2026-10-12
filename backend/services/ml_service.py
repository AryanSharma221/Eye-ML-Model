"""
ML Service Layer for Multimodal Health Screening.
Encapsulates all direct interactions with the standalone ML layer:
- Pre-warming DenseNet201 model weights
- Model caching and device management
- Thread-safe inference and Grad-CAM generation
- Retinal image quality checks and preprocessing
- Physiological vitals rule engine analysis
- Combined multimodal screening execution
"""

import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

import torch

from backend.config import DEFAULT_CHECKPOINT_PATH
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
from ml.physio_model import analyze_vitals
from ml.preprocessing import preprocess_image
from ml.quality_gate import check_image_quality
from ml.screening_pipeline import run_screening

logger = logging.getLogger("backend.services.ml_service")


class MLService:
    """
    Thread-safe ML service managing eye disease detection and vitals analysis.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
        self.device = get_default_device()
        self._gradcam_lock = threading.Lock()

    def prewarm(self) -> bool:
        """
        Pre-warms the deep learning model into device memory.
        Returns True if successful, False otherwise.
        """
        if not os.path.isfile(self.checkpoint_path):
            logger.warning(f"Checkpoint not found at '{self.checkpoint_path}'. Pre-warm skipped.")
            return False

        try:
            _ = get_cached_model(weights_path=self.checkpoint_path, device=self.device)
            logger.info(f"Model pre-warmed on device: {self.device}")
            return True
        except Exception as exc:
            logger.error(f"Failed to pre-warm model from '{self.checkpoint_path}': {exc}")
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """
        Returns safe system health information without exposing local paths.
        """
        checkpoint_available = os.path.isfile(self.checkpoint_path)
        model_loaded = len(_MODEL_CACHE) > 0
        return {
            "status": "healthy",
            "model_loaded": model_loaded,
            "device": str(self.device),
            "checkpoint_available": checkpoint_available,
            "supported_classes": CLASS_NAMES,
        }

    def predict_eye(
        self,
        image_path: str,
        include_gradcam: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes retinal image quality gate, preprocessing, inference, and Grad-CAM.

        Returns:
            Dict containing prediction results or quality gate rejection details.
        """
        # 1. Quality Gate Check
        quality_result = check_image_quality(image_path)
        if not quality_result.get("overall_pass", False):
            return {
                "status": "rejected",
                "overall_pass": False,
                "reason": quality_result.get("error_message", "Image failed quality gate validation."),
                "quality_checks": quality_result,
            }

        # 2. Preprocessing
        tensor = preprocess_image(image_path)

        # 3. Model Inference
        inference = predict_eye_disease(
            image_input=tensor,
            weights_path=self.checkpoint_path,
            device=self.device,
            use_cache=True,
            validate_quality=False,
        )

        # 4. Grad-CAM Generation (Thread-safe)
        gradcam_b64 = None
        if include_gradcam:
            try:
                cached_model = get_cached_model(weights_path=self.checkpoint_path, device=self.device)
                with self._gradcam_lock:
                    gradcam_b64 = generate_gradcam(
                        image_path=image_path,
                        predicted_class_idx=inference["class_index"],
                        model=cached_model,
                        device=self.device,
                        input_tensor=tensor,
                    )
            except Exception as gcam_err:
                logger.warning(f"Grad-CAM generation non-critical error: {gcam_err}")
                gradcam_b64 = None

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
            "disclaimer": (
                "Screening prototype output for clinical decision support. Not a standalone diagnosis."
            ),
        }

    def analyze_vitals(
        self,
        heart_rate: float,
        spo2: float,
        temperature: float,
    ) -> Dict[str, Any]:
        """
        Invokes the physiological analysis module.
        """
        return analyze_vitals(
            heart_rate=heart_rate,
            spo2=spo2,
            temperature=temperature,
            raise_errors=True,
        )

    def run_multimodal(
        self,
        image_path: Optional[str] = None,
        heart_rate: Optional[float] = None,
        spo2: Optional[float] = None,
        temperature: Optional[float] = None,
        include_gradcam: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes unified multimodal health screening pipeline.
        """
        return run_screening(
            image_path=image_path,
            heart_rate=heart_rate,
            spo2=spo2,
            temperature=temperature,
            include_gradcam=include_gradcam,
            use_cache=True,
        )


# Global singleton instance
ml_service = MLService()
