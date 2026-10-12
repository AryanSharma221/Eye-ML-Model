"""
API route handler for eye-disease screening (/api/v1/predict_eye).
Reuses teammate's endpoint contract, integrates real ML modules, and instruments pipeline observability.
"""

from datetime import datetime, timezone
import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_db import EyeData, Session as ScreeningSession
from backend.schemas import EyePredictionResponse
from backend.services.ml_service import ml_service
from backend.utils import cleanup_file, save_base64_gradcam, save_uploaded_image
from ml.exceptions import (
    CheckpointCorruptError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    InvalidImageInputError,
    ModelInferenceError,
)

logger = logging.getLogger("backend.routes.eye_routes")
router = APIRouter(prefix="/api/v1", tags=["Eye Screening"])


@router.post("/predict_eye")
async def predict_eye(
    request: Request,
    image: UploadFile = File(..., description="Retinal fundus image (JPG, JPEG, PNG, max 30MB)"),
    session_id: Optional[int] = Form(None, description="Optional screening session ID to attach results to"),
    include_gradcam: bool = Form(True, description="Whether to generate Grad-CAM visualization"),
    db: Session = Depends(get_db),
):
    """
    Submits a retinal fundus image for automated quality gate checks, 4-class DenseNet201
    inference, and optional Grad-CAM explainability heatmap generation.
    Instruments sub-stage latencies and correlation ID tracking.
    """
    request_id = getattr(request.state, "request_id", None)

    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid retinal image file must be uploaded.",
        )

    # 1. Check session existence if session_id provided
    session_obj = None
    if session_id is not None:
        session_obj = db.query(ScreeningSession).filter(ScreeningSession.id == session_id).first()
        if not session_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found.",
            )

    # 2. Save and validate uploaded image safely to backend/uploads
    disk_path, rel_url = await save_uploaded_image(image, subfolder="fundus")

    # 3. Execute ML Service Inference (Quality gate -> Preprocessing -> DenseNet201 -> Grad-CAM)
    try:
        result = ml_service.predict_eye(
            image_path=disk_path,
            include_gradcam=include_gradcam,
            request_id=request_id,
        )
        request.state.ml_timings = result.get("timings_ms")
    except CheckpointNotFoundError:
        cleanup_file(disk_path)
        request.state.error_category = "CheckpointNotFoundError"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model checkpoint weights file not found on server.",
        )
    except (CheckpointCorruptError, CheckpointIncompatibleError) as err:
        cleanup_file(disk_path)
        request.state.error_category = type(err).__name__
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model checkpoint is corrupted or incompatible: {err}",
        )
    except InvalidImageInputError as err:
        cleanup_file(disk_path)
        request.state.error_category = "InvalidImageInputError"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(err)}",
        )
    except Exception as err:
        cleanup_file(disk_path)
        request.state.error_category = type(err).__name__
        logger.error(f"Eye screening execution failure [req_id={request_id}]: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error occurred during retinal image analysis.",
        )

    # 4. Handle Quality Gate Rejection
    if not result.get("overall_pass", False):
        cleanup_file(disk_path)
        request.state.error_category = "QUALITY_REJECTED"
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "rejected",
                "reason": result.get("reason", "Image failed quality gate validation."),
                "quality_checks": result.get("quality_checks", {}),
                "image_url": rel_url,
                "timings_ms": result.get("timings_ms"),
                "request_id": request_id,
            },
        )

    # 5. Persist Grad-CAM heatmap to disk if generated
    gradcam_b64 = result.get("gradcam_image")
    gradcam_url = None
    if gradcam_b64:
        gradcam_url = save_base64_gradcam(gradcam_b64)

    # 6. Database Persistence (if session_id provided)
    now = datetime.now(timezone.utc)
    if session_obj:
        eye_record = EyeData(
            session_id=session_id,
            image_path=rel_url,
            prediction=result["prediction"],
            class_index=result.get("class_index"),
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            gradcam_path=gradcam_url,
            all_probabilities=result["all_probabilities"],
            quality_checks=result["quality_checks"],
            created_at=now,
        )
        db.add(eye_record)

        if result["risk_level"] == "HIGH":
            session_obj.overall_risk_level = "HIGH"
        elif result["risk_level"] == "MODERATE" and session_obj.overall_risk_level != "HIGH":
            session_obj.overall_risk_level = "MODERATE"
        db.commit()

    return {
        "status": "success",
        "prediction": result["prediction"],
        "class_index": result.get("class_index"),
        "confidence": result["confidence"],
        "risk_level": result["risk_level"],
        "all_probabilities": result["all_probabilities"],
        "quality_checks": result["quality_checks"],
        "gradcam_image": gradcam_b64,
        "gradcam_url": gradcam_url,
        "image_url": rel_url,
        "session_id": session_id,
        "timings_ms": result.get("timings_ms"),
        "request_id": request_id,
        "timestamp": now.isoformat(),
        "disclaimer": result.get("disclaimer"),
    }
