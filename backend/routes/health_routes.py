"""
Health check endpoints for backend service and ML model readiness.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas import HealthResponse
from backend.services.ml_service import ml_service

router = APIRouter(tags=["Health"])


@router.get("/health/live", tags=["Health"])
@router.get("/api/v1/health/live", tags=["Health"])
def liveness_check():
    """
    Lightweight application liveness probe.
    Returns 200 OK immediately to indicate the web server process is responsive.
    Does not touch database or execute expensive ML operations.
    """
    return {
        "status": "alive",
        "service": "multimodal-health-screening-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/ready", tags=["Health"])
@router.get("/api/v1/health/ready", tags=["Health"])
def readiness_check(response: Response, db: Session = Depends(get_db)):
    """
    Deep application readiness probe for orchestration and load balancers.
    Verifies model weights checkpoint presence, verification state, in-memory pre-warming,
    and database connectivity.
    Returns 200 OK if ready for inference traffic, or 503 Service Unavailable if unready.
    Never exposes internal filesystem paths or sensitive credentials.
    """
    health_info = ml_service.get_health_status()
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    checkpoint_verified = bool(health_info.get("checkpoint_verified", False))
    is_ready = bool(
        health_info["checkpoint_available"]
        and checkpoint_verified
        and health_info["model_loaded"]
        and db_ok
    )

    payload = {
        "status": "ready" if is_ready else "unready",
        "checkpoint_available": health_info["checkpoint_available"],
        "checkpoint_verified": checkpoint_verified,
        "model_loaded": health_info["model_loaded"],
        "database_connected": db_ok,
        "device": health_info["device"],
        "model_load_duration_ms": health_info.get("model_load_duration_ms"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        reasons = []
        if not health_info["checkpoint_available"]:
            reasons.append("Model weights checkpoint not found on server")
        elif not checkpoint_verified:
            err_type = health_info.get("checkpoint_error") or "VerificationFailed"
            reasons.append(f"Model checkpoint verification failed ({err_type})")
        if not health_info["model_loaded"]:
            reasons.append("Model weights not pre-warmed into memory")
        if not db_ok:
            reasons.append("Database connection failure")
        payload["reason"] = "; ".join(reasons)

    return payload


@router.get("/health", response_model=HealthResponse)
@router.get("/api/v1/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """
    Returns runtime health status, model readiness, device, and supported classes.
    Safe: does not expose local filesystem paths or environment secrets.
    """
    health_info = ml_service.get_health_status()

    return HealthResponse(
        status=health_info["status"],
        model_loaded=health_info["model_loaded"],
        device=health_info["device"],
        checkpoint_available=health_info["checkpoint_available"],
        checkpoint_verified=health_info.get("checkpoint_verified"),
        checkpoint_error=health_info.get("checkpoint_error"),
        model_load_duration_ms=health_info.get("model_load_duration_ms"),
        supported_classes=health_info["supported_classes"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
