"""
SQLAlchemy ORM models for Multimodal Health Screening.
Reuses and extends teammate's database schema:
- patients: Registered patients
- sessions: Screening sessions linked to a patient
- eye_data: Eye-disease predictions and Grad-CAM paths
- physiological_data: Vital sign readings and statuses
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    JSON,
    Text,
)
from sqlalchemy.orm import relationship

from backend.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    contact = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=utc_now)

    sessions = relationship(
        "Session", back_populates="patient", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Patient(id={self.id}, name='{self.name}')>"


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    timestamp = Column(DateTime, default=utc_now)
    created_at = Column(DateTime, default=utc_now)
    notes = Column(String(500), nullable=True)
    overall_risk_level = Column(String(30), nullable=True, default="UNKNOWN")

    patient = relationship("Patient", back_populates="sessions")
    eye_record = relationship(
        "EyeData",
        uselist=False,
        back_populates="session",
        cascade="all, delete-orphan",
    )
    vitals_records = relationship(
        "PhysiologicalData",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    @property
    def eye_records(self):
        """Convenience property for multi-record compatibility."""
        return [self.eye_record] if self.eye_record is not None else []

    @property
    def physio_records(self):
        """Alias for vitals_records."""
        return self.vitals_records

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, patient_id={self.patient_id})>"


# Alias for backward compatibility with existing tests
ScreeningSession = Session


class EyeData(Base):
    __tablename__ = "eye_data"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    image_path = Column(String(255), nullable=False)
    prediction = Column(String(50), nullable=False)
    class_index = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    gradcam_path = Column(String(255), nullable=True)
    all_probabilities = Column(JSON, nullable=False)
    quality_checks = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=utc_now)
    timestamp = Column(DateTime, default=utc_now)

    session = relationship("Session", back_populates="eye_record")

    def __repr__(self) -> str:
        return (
            f"<EyeData(id={self.id}, prediction='{self.prediction}', "
            f"confidence={self.confidence})>"
        )


class PhysiologicalData(Base):
    __tablename__ = "physiological_data"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    heart_rate_bpm = Column(Float, nullable=False)
    spo2_percent = Column(Float, nullable=False)
    temperature_celsius = Column(Float, nullable=False)
    hr_status = Column(String(50), nullable=False)
    spo2_status = Column(String(50), nullable=False)
    temp_status = Column(String(50), nullable=False)
    overall_status = Column(String(50), nullable=False)
    risk_level = Column(String(30), nullable=True)
    summary = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=utc_now)
    created_at = Column(DateTime, default=utc_now)

    session = relationship("Session", back_populates="vitals_records")

    def __repr__(self) -> str:
        return (
            f"<PhysiologicalData(id={self.id}, overall_status='{self.overall_status}')>"
        )
