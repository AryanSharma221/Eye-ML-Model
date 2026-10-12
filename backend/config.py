"""
Configuration settings for the Multimodal Health Screening Backend.
Environment variables can override default settings.
"""

import os
from pathlib import Path
from typing import List

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
ML_DIR = PROJECT_DIR / "ml"

# Database Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BACKEND_DIR / 'health_screening.db'}",
)
# Heroku / Render legacy postgres:// fix for modern SQLAlchemy
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# CORS Configuration
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

def get_cors_origins() -> List[str]:
    cors_env = os.getenv("CORS_ORIGINS")
    if cors_env:
        return [origin.strip() for origin in cors_env.split(",") if origin.strip()]
    return _DEFAULT_CORS_ORIGINS

# Storage & Upload Configuration
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", str(BACKEND_DIR / "uploads")))
GRADCAM_DIR = UPLOADS_DIR / "gradcam"
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "30"))
MAX_FILE_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Security & Rate Limiting Configuration
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
ENABLE_RATE_LIMIT = os.getenv("ENABLE_RATE_LIMIT", "true").lower() in ("true", "1", "yes")
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))

# ML Checkpoint & Runtime Configuration
ML_DEVICE = os.getenv("ML_DEVICE", "auto")
DEFAULT_CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    str(ML_DIR / "weights" / "eye_model.pth"),
)

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json").lower()
API_TITLE = "Multimodal Health Screening API"
API_VERSION = "1.0.0"

# Ensure upload directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
