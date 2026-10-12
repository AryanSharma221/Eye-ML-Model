"""
API route definitions for the Multimodal Health Screening backend service.
"""

from backend.routes.health_routes import router as health_router
from backend.routes.screening_routes import router as screening_router
from backend.routes.eye_routes import router as eye_router
from backend.routes.vitals_routes import router as vitals_router
from backend.routes.patient_routes import router as patient_router
from backend.routes.session_routes import router as session_router

__all__ = [
    "health_router",
    "screening_router",
    "eye_router",
    "vitals_router",
    "patient_router",
    "session_router",
]
