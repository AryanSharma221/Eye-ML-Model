"""
API route handler for physiological vital signs assessment (/api/v1/submit_vitals).
Reuses teammate's hardware integration contract and integrates the ML physio model.
"""

from datetime import datetime, timezone
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_db import PhysiologicalData, Session as ScreeningSession
from backend.schemas import VitalsSubmit
from backend.services.ml_service import ml_service
from ml.exceptions import (
    PhysiologicalTypeValidationError,
    PhysiologicalValidationError,
)

logger = logging.getLogger("backend.routes.vitals_routes")
router = APIRouter(prefix="/api/v1", tags=["Vital Signs"])


@router.post("/submit_vitals")
def submit_vitals(
    payload: VitalsSubmit,
    db: Session = Depends(get_db),
):
    """
    Submits physiological measurements (heart rate, SpO2, core body temperature)
    for rule-based/ML physiological triage and anomaly detection.
    """
    hr, ox, temp = payload.resolve_vitals()

    if hr is None or ox is None or temp is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="All three vital signs (heart_rate, spo2, temperature) must be provided.",
        )

    # 1. Validate session existence if session_id provided
    session_obj = None
    if payload.session_id is not None:
        session_obj = db.query(ScreeningSession).filter(ScreeningSession.id == payload.session_id).first()
        if not session_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {payload.session_id} not found.",
            )

    # 2. Invoke ML Service Physiological Analysis
    try:
        analysis = ml_service.analyze_vitals(
            heart_rate=hr,
            spo2=ox,
            temperature=temp,
        )
    except (PhysiologicalValidationError, PhysiologicalTypeValidationError) as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as err:
        logger.error(f"Vitals assessment error: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error occurred while evaluating physiological readings.",
        )

    # 3. Database Persistence (if session_id provided)
    now = datetime.now(timezone.utc)
    if session_obj:
        physio_record = PhysiologicalData(
            session_id=payload.session_id,
            heart_rate_bpm=hr,
            spo2_percent=ox,
            temperature_celsius=temp,
            hr_status=analysis.get("hr_status"),
            spo2_status=analysis.get("spo2_status"),
            temp_status=analysis.get("temp_status"),
            overall_status=analysis.get("overall_status"),
            risk_level=analysis.get("risk_level"),
            summary=analysis.get("summary"),
            timestamp=now,
        )
        db.add(physio_record)

        if analysis.get("risk_level") == "HIGH":
            session_obj.overall_risk_level = "HIGH"
        elif analysis.get("risk_level") == "MODERATE" and session_obj.overall_risk_level != "HIGH":
            session_obj.overall_risk_level = "MODERATE"
        db.commit()

    # 4. Return structured response matching both teammate and pipeline schemas
    return {
        "status": "success",
        "analysis": {
            "heart_rate": {
                "value": hr,
                "unit": "BPM",
                "status": analysis["hr_status"],
            },
            "spo2": {
                "value": ox,
                "unit": "%",
                "status": analysis["spo2_status"],
            },
            "temperature": {
                "value": temp,
                "unit": "°C",
                "status": analysis["temp_status"],
            },
            "overall_status": analysis["overall_status"],
        },
        "session_id": payload.session_id,
        "timestamp": now.isoformat(),
        # Top-level fields
        "heart_rate": {
            "value": hr,
            "status": analysis["hr_status"],
            "unit": "BPM",
        },
        "spo2": {
            "value": ox,
            "status": analysis["spo2_status"],
            "unit": "%",
        },
        "temperature": {
            "value": temp,
            "status": analysis["temp_status"],
            "unit": "°C",
        },
        "hr_status": analysis["hr_status"],
        "spo2_status": analysis["spo2_status"],
        "temp_status": analysis["temp_status"],
        "overall_status": analysis["overall_status"],
        "risk_level": analysis["risk_level"],
        "is_abnormal": analysis["is_abnormal"],
        "summary": analysis["summary"],
        "disclaimer": analysis["disclaimer"],
    }
