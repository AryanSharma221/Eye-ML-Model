"""
FastAPI application entry point for the Multimodal Health Screening system.
Integrates teammate's backend routes, ORM persistence, the standalone ML layer,
and production observability (structured logging, request correlation, latency monitoring).
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import os
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import (
    API_TITLE,
    API_VERSION,
    DEBUG,
    LOG_FORMAT,
    LOG_LEVEL,
    get_cors_origins,
    UPLOADS_DIR,
)
from backend.database import init_db
from backend.observability import (
    CorrelationIdMiddleware,
    configure_backend_logging,
)
from backend.routes import (
    eye_router,
    health_router,
    patient_router,
    screening_router,
    session_router,
    vitals_router,
)
from backend.services.ml_service import ml_service
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

# Configure structured backend logging
configure_backend_logging(level=LOG_LEVEL, log_format=LOG_FORMAT)
logger = logging.getLogger("backend.main")

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifespan context.
    Initializes database tables and pre-warms the deep learning model once at startup.
    """
    logger.info("Initializing relational database schema...")
    init_db()

    logger.info("Pre-warming DenseNet201 model weights into memory...")
    success = ml_service.prewarm()
    if success:
        logger.info(
            f"DenseNet201 model successfully pre-warmed on device: {ml_service.device} "
            f"({ml_service.model_load_duration_ms}ms)"
        )
    else:
        logger.warning(
            "Model checkpoint not found or could not be pre-warmed. "
            "Inference requests will fail until checkpoint is provided."
        )

    yield

    logger.info("Shutting down Multimodal Health Screening API service...")


app = FastAPI(
    title=API_TITLE,
    description=(
        "Production-grade backend service integrating 4-class DenseNet201 retinal screening, "
        "Grad-CAM visual explainability, and independent physiological vitals assessment."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    debug=DEBUG,
    lifespan=lifespan,
)

# Correlation ID and Request Observability Middleware (runs on every request)
app.add_middleware(CorrelationIdMiddleware)

# CORS Middleware using environment-based allowed origins
cors_origins = get_cors_origins()
logger.info(f"Configuring CORS with allowed origins: {cors_origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Centralized Error Handlers (with Correlation ID & Safe Sanitization)
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", None)
    request.state.error_category = "ValidationError"
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        errors.append(f"{field}: {err.get('msg')}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_type": "ValidationError",
            "message": "Input validation error occurred.",
            "details": errors,
            "request_id": req_id,
        },
        headers={"X-Request-ID": req_id} if req_id else None,
    )


@app.exception_handler(QualityGateRejectionError)
async def quality_rejection_handler(request: Request, exc: QualityGateRejectionError):
    req_id = getattr(request.state, "request_id", None)
    request.state.error_category = "QUALITY_REJECTED"
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "rejected",
            "reason": str(exc),
            "quality_checks": getattr(exc, "quality_checks", {}),
            "request_id": req_id,
        },
        headers={"X-Request-ID": req_id} if req_id else None,
    )


@app.exception_handler(CheckpointNotFoundError)
async def checkpoint_not_found_handler(request: Request, exc: CheckpointNotFoundError):
    req_id = getattr(request.state, "request_id", None)
    request.state.error_category = "CheckpointNotFoundError"
    logger.error(f"Checkpoint not found error [req_id={req_id}]: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "error",
            "error_type": "CheckpointNotFoundError",
            "message": "Model checkpoint weights file not found on server.",
            "request_id": req_id,
        },
        headers={"X-Request-ID": req_id} if req_id else None,
    )


@app.exception_handler(CheckpointCorruptError)
@app.exception_handler(CheckpointIncompatibleError)
async def checkpoint_invalid_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", None)
    request.state.error_category = type(exc).__name__
    logger.error(f"Checkpoint invalid error [req_id={req_id}]: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "error",
            "error_type": type(exc).__name__,
            "message": "Model checkpoint is corrupted or incompatible with current architecture.",
            "request_id": req_id,
        },
        headers={"X-Request-ID": req_id} if req_id else None,
    )


@app.exception_handler(InvalidImageInputError)
async def invalid_image_handler(request: Request, exc: InvalidImageInputError):
    req_id = getattr(request.state, "request_id", None)
    request.state.error_category = "InvalidImageInputError"
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "error",
            "error_type": "InvalidImageInputError",
            "message": str(exc),
            "request_id": req_id,
        },
        headers={"X-Request-ID": req_id} if req_id else None,
    )


@app.exception_handler(PhysiologicalValidationError)
@app.exception_handler(PhysiologicalTypeValidationError)
async def physio_validation_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", None)
    request.state.error_category = type(exc).__name__
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "request_id": req_id,
        },
        headers={"X-Request-ID": req_id} if req_id else None,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    req_id = getattr(request.state, "request_id", None)
    if isinstance(exc.detail, dict):
        content = dict(exc.detail)
    else:
        content = {
            "status": "error" if exc.status_code >= 400 else "ok",
            "error_type": "HTTPException",
            "message": exc.detail,
            "detail": exc.detail,
        }
    if req_id and "request_id" not in content:
        content["request_id"] = req_id

    headers = dict(exc.headers or {})
    if req_id and "X-Request-ID" not in headers:
        headers["X-Request-ID"] = req_id

    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=headers if headers else None,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", None)
    request.state.error_category = type(exc).__name__
    # Log internal traceback with request ID for server-side debugging
    logger.exception(f"Unhandled exception on {request.method} {request.url.path} [req_id={req_id}]: {exc}")

    # Return safe, non-leaking generic error response to client
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_type": "InternalServerError",
            "message": "An unexpected error occurred while processing the request.",
            "request_id": req_id,
        },
        headers={"X-Request-ID": req_id} if req_id else None,
    )


# ---------------------------------------------------------------------------
# Router Registration
# ---------------------------------------------------------------------------
app.include_router(health_router)
app.include_router(screening_router)
app.include_router(eye_router)
app.include_router(vitals_router)
app.include_router(patient_router)
app.include_router(session_router)

# Mount uploaded media directory
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# If static frontend distribution build exists, mount it at /
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/", tags=["System"])
    def root():
        """Root endpoint — redirect hint to the interactive OpenAPI docs."""
        return {
            "message": "Multimodal Health Screening API",
            "docs": "/docs",
            "redoc": "/redoc",
            "version": app.version,
        }
