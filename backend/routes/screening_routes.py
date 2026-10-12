"""
Unified multimodal health screening API route (/api/v1/screen).
Combines retinal fundus imagery and physiological measurements into a single screening session.
"""

from datetime import datetime, timezone
import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_db import EyeData, PhysiologicalData, Session as ScreeningSession
from backend.schemas import ScreeningResponse
from backend.services.ml_service import ml_service
from backend.utils import cleanup_file, save_base64_gradcam, save_uploaded_image

logger = logging.getLogger("backend.routes.screening_routes")
router = APIRouter(prefix="/api/v1", tags=["Multimodal Screening"])


@router.post("/screen", response_model=ScreeningResponse)
async def screen_multimodal(
    image: Optional[UploadFile] = File(None, description="Retinal fundus image (JPG, JPEG, PNG, max 30MB)"),
    heart_rate: Optional[float] = Form(None, description="Heart rate in BPM (20 - 300)"),
    spo2: Optional[float] = Form(None, description="Blood oxygen saturation (0 - 100%)"),
    temperature: Optional[float] = Form(None, description="Body core temperature in °C (25.0 - 45.0)"),
    patient_id: Optional[int] = Form(None, description="Optional patient ID to attach session to"),
    session_id: Optional[int] = Form(None, description="Optional existing session ID"),
    include_gradcam: bool = Form(True, description="Whether to compute Grad-CAM heatmap"),
    db: Session = Depends(get_db),
):
    """
    Unified multimodal screening endpoint. Accepts retinal fundus imagery and/or
    physiological measurements together in a single request.
    """
    has_image = image is not None and bool(image.filename)
    has_any_vitals = any(v is not None for v in (heart_rate, spo2, temperature))

    if not has_image and not has_any_vitals:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one modality (retinal fundus image or vital signs) must be provided.",
        )

    # 1. Range Validation for Vital Signs (if provided)
    if heart_rate is not None and (heart_rate < 20.0 or heart_rate > 300.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Heart rate {heart_rate} BPM is outside physiological range [20, 300].",
        )
    if spo2 is not None and (spo2 < 0.0 or spo2 > 100.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"SpO2 {spo2}% is outside physiological range [0, 100].",
        )
    if temperature is not None and (temperature < 25.0 or temperature > 45.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Temperature {temperature}°C is outside physiological range [25.0, 45.0].",
        )

    # 2. Process and Save Image (if provided)
    disk_path = None
    rel_img_url = None
    if has_image and image is not None:
        disk_path, rel_img_url = await save_uploaded_image(image, subfolder="fundus")

    # 3. Execute Unified Screening Pipeline via ML Service
    try:
        pipeline_output = ml_service.run_multimodal(
            image_path=disk_path,
            heart_rate=heart_rate,
            spo2=spo2,
            temperature=temperature,
            include_gradcam=include_gradcam,
        )
    except Exception as exc:
        if disk_path:
            cleanup_file(disk_path)
        logger.error(f"Multimodal screening execution failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while executing multimodal screening pipeline.",
        )

    # 4. Save Grad-CAM Artifact to Disk (if generated)
    eye_res = pipeline_output.get("eye_screening", {})
    if rel_img_url:
        eye_res["image_url"] = rel_img_url

    gradcam_b64 = eye_res.get("gradcam_image")
    if gradcam_b64:
        gradcam_url = save_base64_gradcam(gradcam_b64)
        eye_res["gradcam_url"] = gradcam_url

    # 5. Database Session Recording
    active_session_id = session_id
    if patient_id or session_id:
        try:
            if not active_session_id and patient_id:
                sess_record = ScreeningSession(
                    patient_id=patient_id,
                    overall_risk_level=pipeline_output.get("overall_risk_level", "UNKNOWN"),
                )
                db.add(sess_record)
                db.flush()
                active_session_id = sess_record.id
            elif active_session_id:
                sess_record = db.query(ScreeningSession).filter(ScreeningSession.id == active_session_id).first()
                if sess_record:
                    sess_record.overall_risk_level = pipeline_output.get("overall_risk_level", "UNKNOWN")

            if active_session_id:
                # Record eye screening if processed
                if eye_res.get("status") == "success":
                    eye_record = EyeData(
                        session_id=active_session_id,
                        image_path=rel_img_url,
                        prediction=eye_res.get("prediction"),
                        class_index=eye_res.get("class_index"),
                        confidence=eye_res.get("confidence"),
                        risk_level=eye_res.get("risk_level"),
                        gradcam_path=eye_res.get("gradcam_url"),
                        all_probabilities=eye_res.get("all_probabilities"),
                        quality_checks=eye_res.get("quality_checks"),
                    )
                    db.add(eye_record)

                # Record vitals screening if processed
                vitals_res = pipeline_output.get("vitals_screening", {})
                if vitals_res.get("status") == "success":
                    physio_record = PhysiologicalData(
                        session_id=active_session_id,
                        heart_rate_bpm=heart_rate,
                        spo2_percent=spo2,
                        temperature_celsius=temperature,
                        hr_status=vitals_res.get("hr_status"),
                        spo2_status=vitals_res.get("spo2_status"),
                        temp_status=vitals_res.get("temp_status"),
                        overall_status=vitals_res.get("overall_status"),
                        risk_level=vitals_res.get("risk_level"),
                        summary=vitals_res.get("summary"),
                    )
                    db.add(physio_record)

                db.commit()
        except Exception as db_err:
            logger.warning(f"Database persistence error: {db_err}")
            db.rollback()

    pipeline_output["session_id"] = active_session_id

    # If only an image was submitted and it failed quality gate, return 400 Bad Request
    if has_image and not has_any_vitals and eye_res.get("status") == "rejected":
        if disk_path and not active_session_id:
            cleanup_file(disk_path)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=pipeline_output,
        )

    return pipeline_output
