"""
FastAPI application entry point for the Multimodal Health Screening system.
Integrates teammate's backend routes, ORM persistence, and the standalone ML layer.
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
    LOG_LEVEL,
    get_cors_origins,
    UPLOADS_DIR,
)
from backend.database import init_db
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

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
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
        logger.info(f"DenseNet201 model successfully pre-warmed on device: {ml_service.device}")
    else:
        logger.warning(
            f"Model checkpoint not found or could not be pre-warmed. "
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
# Centralized Error Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
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
        },
    )


@app.exception_handler(QualityGateRejectionError)
async def quality_rejection_handler(request: Request, exc: QualityGateRejectionError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "rejected",
            "reason": str(exc),
            "quality_checks": getattr(exc, "quality_checks", {}),
        },
    )


@app.exception_handler(CheckpointNotFoundError)
async def checkpoint_not_found_handler(request: Request, exc: CheckpointNotFoundError):
    logger.error(f"Checkpoint not found error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "error",
            "error_type": "CheckpointNotFoundError",
            "message": "Model checkpoint weights file not found on server.",
        },
    )


@app.exception_handler(CheckpointCorruptError)
@app.exception_handler(CheckpointIncompatibleError)
async def checkpoint_invalid_handler(request: Request, exc: Exception):
    logger.error(f"Checkpoint invalid error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "error",
            "error_type": type(exc).__name__,
            "message": f"Model checkpoint is corrupted or incompatible: {exc}",
        },
    )


@app.exception_handler(InvalidImageInputError)
async def invalid_image_handler(request: Request, exc: InvalidImageInputError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "error",
            "error_type": "InvalidImageInputError",
            "message": str(exc),
        },
    )


@app.exception_handler(PhysiologicalValidationError)
@app.exception_handler(PhysiologicalTypeValidationError)
async def physio_validation_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    content = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_type": "InternalServerError",
            "message": "An unexpected error occurred while processing the request.",
        },
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
