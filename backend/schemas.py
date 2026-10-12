"""
Pydantic schemas for Multimodal Health Screening API requests and responses.
Unifies teammate's API contracts with the multimodal screening contracts.
"""

from datetime import datetime
import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


# -----------------------------------------------------------------------------
# Health Check Schemas
# -----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str = "healthy"
    model_loaded: bool
    device: str
    checkpoint_available: bool
    checkpoint_verified: Optional[bool] = None
    checkpoint_error: Optional[str] = None
    model_load_duration_ms: Optional[float] = None
    supported_classes: List[str]
    timestamp: str


# -----------------------------------------------------------------------------
# Patient Schemas
# -----------------------------------------------------------------------------
class PatientCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = Field(None, min_length=1, max_length=30)
    contact: Optional[str] = Field(None, max_length=50)


class PatientResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    contact: Optional[str] = None
    created_at: datetime


# -----------------------------------------------------------------------------
# Eye & Vitals Record Schemas
# -----------------------------------------------------------------------------
class EyeDataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    image_path: str
    prediction: str
    class_index: Optional[int] = None
    confidence: float
    risk_level: str
    gradcam_path: Optional[str] = None
    all_probabilities: Dict[str, Any]
    quality_checks: Dict[str, Any]
    created_at: datetime


# Alias for backward compatibility
EyeRecordResponse = EyeDataResponse


class PhysioDataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    heart_rate_bpm: float
    spo2_percent: float
    temperature_celsius: float
    hr_status: str
    spo2_status: str
    temp_status: str
    overall_status: str
    risk_level: Optional[str] = None
    summary: Optional[str] = None
    timestamp: datetime


# Alias for backward compatibility
PhysioRecordResponse = PhysioDataResponse


# -----------------------------------------------------------------------------
# Session Schemas
# -----------------------------------------------------------------------------
class SessionCreate(BaseModel):
    patient_id: Optional[int] = None
    notes: Optional[str] = Field(None, max_length=500)


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_id: Optional[int] = None
    timestamp: datetime
    notes: Optional[str] = None
    overall_risk_level: Optional[str] = "UNKNOWN"
    eye_record: Optional[EyeDataResponse] = None
    eye_records: Optional[List[EyeDataResponse]] = None
    vitals_records: List[PhysioDataResponse] = []
    physio_records: Optional[List[PhysioDataResponse]] = None


# -----------------------------------------------------------------------------
# Vital Signs Submission Schemas
# -----------------------------------------------------------------------------
class VitalsSubmit(BaseModel):
    """
    Payload for submitting physiological vital signs.
    Supports both teammate's field naming (_bpm, _percent, _celsius)
    and short aliases (heart_rate, spo2, temperature).
    """
    session_id: Optional[int] = None
    heart_rate_bpm: Optional[float] = Field(None, description="Heart rate in BPM (30 - 220)")
    heart_rate: Optional[float] = Field(None, description="Alias for heart_rate_bpm")
    spo2_percent: Optional[float] = Field(None, description="Blood oxygen saturation (50 - 100%)")
    spo2: Optional[float] = Field(None, description="Alias for spo2_percent")
    temperature_celsius: Optional[float] = Field(None, description="Core body temperature in °C (30.0 - 45.0)")
    temperature: Optional[float] = Field(None, description="Alias for temperature_celsius")

    def resolve_vitals(self):
        hr = self.heart_rate_bpm if self.heart_rate_bpm is not None else self.heart_rate
        ox = self.spo2_percent if self.spo2_percent is not None else self.spo2
        temp = self.temperature_celsius if self.temperature_celsius is not None else self.temperature
        return hr, ox, temp

    @field_validator("heart_rate_bpm", "heart_rate")
    @classmethod
    def validate_hr(cls, v):
        if v is not None:
            if math.isnan(v) or math.isinf(v) or v < 30.0 or v > 220.0:
                raise ValueError(f"Heart rate {v} BPM is outside physiological range [30, 220].")
        return v

    @field_validator("spo2_percent", "spo2")
    @classmethod
    def validate_spo2(cls, v):
        if v is not None:
            if math.isnan(v) or math.isinf(v) or v < 50.0 or v > 100.0:
                raise ValueError(f"SpO2 {v}% is outside physiological range [50, 100].")
        return v

    @field_validator("temperature_celsius", "temperature")
    @classmethod
    def validate_temp(cls, v):
        if v is not None:
            if math.isnan(v) or math.isinf(v) or v < 30.0 or v > 45.0:
                raise ValueError(f"Temperature {v}°C is outside physiological range [30.0, 45.0].")
        return v


# Alias for backward compatibility
VitalsInput = VitalsSubmit


class VitalDetail(BaseModel):
    value: float
    unit: str
    status: str


class VitalsAnalysisResult(BaseModel):
    heart_rate: VitalDetail
    spo2: VitalDetail
    temperature: VitalDetail
    overall_status: str


class VitalsResponse(BaseModel):
    status: str = "success"
    analysis: Optional[VitalsAnalysisResult] = None
    session_id: Optional[int] = None
    timestamp: Optional[str] = None
    # Top-level fields for convenience
    heart_rate: Optional[Dict[str, Any]] = None
    spo2: Optional[Dict[str, Any]] = None
    temperature: Optional[Dict[str, Any]] = None
    hr_status: Optional[str] = None
    spo2_status: Optional[str] = None
    temp_status: Optional[str] = None
    overall_status: Optional[str] = None
    risk_level: Optional[str] = None
    is_abnormal: Optional[bool] = None
    summary: Optional[str] = None
    disclaimer: Optional[str] = None


# -----------------------------------------------------------------------------
# Eye Prediction Response Schemas
# -----------------------------------------------------------------------------
class EyePredictionResponse(BaseModel):
    status: str = "success"
    prediction: str
    class_index: Optional[int] = None
    confidence: float
    risk_level: str
    all_probabilities: Dict[str, float]
    gradcam_image: Optional[str] = None
    gradcam_url: Optional[str] = None
    quality_checks: Dict[str, Any]
    image_url: Optional[str] = None
    session_id: Optional[int] = None
    timestamp: str
    timings_ms: Optional[Dict[str, Optional[float]]] = None
    disclaimer: Optional[str] = None


class EyeRejectionResponse(BaseModel):
    status: str = "rejected"
    reason: str
    quality_checks: Dict[str, Any]
    image_url: Optional[str] = None
    timings_ms: Optional[Dict[str, Optional[float]]] = None


# -----------------------------------------------------------------------------
# Multimodal Screening Response Schema
# -----------------------------------------------------------------------------
class ScreeningResponse(BaseModel):
    status: str
    screening_id: str
    timestamp: str
    duration_ms: float
    timings_ms: Optional[Dict[str, Optional[float]]] = None
    session_id: Optional[int] = None
    eye_screening: Dict[str, Any]
    vitals_screening: Dict[str, Any]
    overall_risk_level: str
    disclaimer: str
