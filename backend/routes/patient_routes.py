"""
Patient management API routes (/api/v1/patients).
Full CRUD implementation reusing teammate's patient management endpoints.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload

from backend.database import get_db
from backend.models_db import Patient, Session as ScreeningSession
from backend.schemas import PatientCreate, PatientResponse, SessionResponse
from backend.security import delete_physical_file

router = APIRouter(prefix="/api/v1/patients", tags=["Patients"])


@router.post("", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db)):
    """Registers a new patient."""
    patient = Patient(
        name=payload.name.strip(),
        age=payload.age,
        gender=payload.gender,
        contact=payload.contact,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.get("", response_model=List[PatientResponse])
def list_patients(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Lists registered patients with pagination."""
    return db.query(Patient).offset(skip).limit(limit).all()


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    """Fetches details for a single patient."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found.",
        )
    return patient


@router.put("/{patient_id}", response_model=PatientResponse)
def update_patient(
    patient_id: int,
    payload: PatientCreate,
    db: Session = Depends(get_db),
):
    """Updates a patient's details."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found.",
        )

    patient.name = payload.name.strip()
    patient.age = payload.age
    patient.gender = payload.gender
    patient.contact = payload.contact
    db.commit()
    db.refresh(patient)
    return patient


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    """Deletes a patient and all associated screening sessions (cascade), cleaning up physical image files."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found.",
        )

    # Cascading removal of patient's physical retinal images and Grad-CAM artifacts
    patient_sessions = (
        db.query(ScreeningSession)
        .filter(ScreeningSession.patient_id == patient_id)
        .options(joinedload(ScreeningSession.eye_record))
        .all()
    )
    for sess in patient_sessions:
        if sess.eye_record:
            delete_physical_file(sess.eye_record.image_path)
            delete_physical_file(sess.eye_record.gradcam_path)

    db.delete(patient)
    db.commit()


@router.get("/{patient_id}/sessions", response_model=List[SessionResponse])
def get_patient_sessions(patient_id: int, db: Session = Depends(get_db)):
    """Retrieves all screening sessions and records for a patient."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found.",
        )

    sessions = (
        db.query(ScreeningSession)
        .filter(ScreeningSession.patient_id == patient_id)
        .options(
            joinedload(ScreeningSession.eye_record),
            joinedload(ScreeningSession.vitals_records),
        )
        .order_by(ScreeningSession.timestamp.desc())
        .all()
    )
    return sessions
