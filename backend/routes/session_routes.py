"""
Session management and clinical report generation API routes (/api/v1/sessions).
Reuses teammate's ReportLab PDF report generation and session lifecycle management.
"""

from datetime import datetime, timezone
import io
import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session, joinedload

import html
from backend.database import get_db
from backend.models_db import EyeData, Patient, PhysiologicalData, Session as ScreeningSession
from backend.schemas import SessionCreate, SessionResponse
from backend.security import delete_physical_file, sanitize_html_text

router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    """Creates a new screening session, optionally linked to a patient."""
    if payload.patient_id:
        patient = db.query(Patient).filter(Patient.id == payload.patient_id).first()
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient with ID {payload.patient_id} does not exist.",
            )

    session_obj = ScreeningSession(
        patient_id=payload.patient_id,
        notes=payload.notes,
    )
    db.add(session_obj)
    db.commit()
    db.refresh(session_obj)
    return session_obj


@router.get("", response_model=List[SessionResponse])
def list_sessions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Lists screening sessions."""
    return (
        db.query(ScreeningSession)
        .options(
            joinedload(ScreeningSession.eye_record),
            joinedload(ScreeningSession.vitals_records),
        )
        .order_by(ScreeningSession.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: int, db: Session = Depends(get_db)):
    """Fetches details for a single screening session."""
    sess = (
        db.query(ScreeningSession)
        .filter(ScreeningSession.id == session_id)
        .options(
            joinedload(ScreeningSession.eye_record),
            joinedload(ScreeningSession.vitals_records),
        )
        .first()
    )
    if not sess:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID {session_id} not found.",
        )
    return sess


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: int, db: Session = Depends(get_db)):
    """Deletes a screening session and all associated files and records."""
    sess = (
        db.query(ScreeningSession)
        .filter(ScreeningSession.id == session_id)
        .options(
            joinedload(ScreeningSession.eye_record),
            joinedload(ScreeningSession.vitals_records),
        )
        .first()
    )
    if not sess:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID {session_id} not found.",
        )

    # Clean up physical retinal fundus and Grad-CAM image files
    if sess.eye_record:
        delete_physical_file(sess.eye_record.image_path)
        delete_physical_file(sess.eye_record.gradcam_path)

    db.delete(sess)
    db.commit()


@router.get("/{session_id}/report")
def get_session_report(
    session_id: int,
    request: Request,
    format: Optional[str] = Query(None, description="Report format: 'pdf' or 'html'"),
    db: Session = Depends(get_db),
):
    """
    Generates a clinical screening summary report for a session.
    Supports both ReportLab PDF (teammate's contract) and HTML view.
    """
    sess = (
        db.query(ScreeningSession)
        .filter(ScreeningSession.id == session_id)
        .options(
            joinedload(ScreeningSession.patient),
            joinedload(ScreeningSession.eye_record),
            joinedload(ScreeningSession.vitals_records),
        )
        .first()
    )
    if not sess:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID {session_id} not found.",
        )

    patient: Optional[Patient] = sess.patient
    eye: Optional[EyeData] = sess.eye_record
    vitals_list: List[PhysiologicalData] = sess.vitals_records

    # Determine requested format (pdf vs html)
    accept_header = request.headers.get("accept", "").lower()
    is_pdf = False
    if format:
        is_pdf = (format.lower() == "pdf")
    elif "application/pdf" in accept_header and "text/html" not in accept_header:
        is_pdf = True

    # ─────────────────────────────────────────────────────────────────────────
    # A. PDF Report Generation via ReportLab (teammate implementation)
    # ─────────────────────────────────────────────────────────────────────────
    if is_pdf:
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch, mm
            from reportlab.platypus import (
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                topMargin=20 * mm,
                bottomMargin=20 * mm,
                leftMargin=20 * mm,
                rightMargin=20 * mm,
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                "ReportTitle",
                parent=styles["Title"],
                fontSize=18,
                spaceAfter=12,
            )
            heading_style = ParagraphStyle(
                "SectionHeading",
                parent=styles["Heading2"],
                fontSize=14,
                spaceAfter=8,
                spaceBefore=16,
                textColor=colors.HexColor("#2c3e50"),
            )
            body_style = styles["BodyText"]

            elements: list = []

            # Title
            elements.append(Paragraph("Clinical Screening Report (DenseNet201)", title_style))
            elements.append(
                Paragraph(
                    f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                    body_style,
                )
            )
            elements.append(Spacer(1, 12))

            # Patient Demographics
            elements.append(Paragraph("Patient Information", heading_style))
            patient_data = [
                ["Field", "Value"],
                ["Name", patient.name if patient else "Anonymous / Unregistered"],
                ["Age", str(patient.age) if patient and patient.age is not None else "N/A"],
                ["Gender", patient.gender if patient and patient.gender else "N/A"],
                ["Contact", patient.contact or "N/A" if patient else "N/A"],
                ["Patient ID", str(patient.id) if patient else "N/A"],
                ["Session ID", str(sess.id)],
                ["Session Date", sess.timestamp.strftime("%Y-%m-%d %H:%M UTC")],
                ["Overall Risk", sess.overall_risk_level or "UNKNOWN"],
                ["Notes", sess.notes or "—"],
            ]
            patient_table = Table(patient_data, colWidths=[2.0 * inch, 4.0 * inch])
            patient_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            elements.append(patient_table)
            elements.append(Spacer(1, 12))

            # Eye Assessment
            elements.append(Paragraph("Eye Disease Assessment (DenseNet201)", heading_style))
            if eye:
                eye_data = [
                    ["Metric", "Value"],
                    ["Predicted Condition", eye.prediction.upper() if eye.prediction else "N/A"],
                    ["Confidence", f"{eye.confidence * 100:.1f}%" if eye.confidence is not None else "N/A"],
                    ["Risk Level", eye.risk_level or "UNKNOWN"],
                ]
                if eye.all_probabilities:
                    for cls, prob in eye.all_probabilities.items():
                        eye_data.append([f"  P({cls})", f"{float(prob) * 100:.1f}%"])

                eye_table = Table(eye_data, colWidths=[2.5 * inch, 3.5 * inch])
                eye_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2980b9")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 10),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]
                    )
                )
                elements.append(eye_table)
                elements.append(Spacer(1, 8))

            # Physiological Vitals
            elements.append(Paragraph("Physiological Vitals", heading_style))
            if vitals_list:
                vitals_header = ["Time", "HR (BPM)", "HR Status", "SpO2 (%)", "SpO2 Status", "Temp (°C)", "Overall"]
                vitals_rows = [vitals_header]
                for v in vitals_list:
                    vitals_rows.append(
                        [
                            v.timestamp.strftime("%H:%M:%S") if v.timestamp else "—",
                            f"{v.heart_rate_bpm:.1f}",
                            v.hr_status,
                            f"{v.spo2_percent:.1f}",
                            v.spo2_status,
                            f"{v.temperature_celsius:.1f}",
                            v.overall_status,
                        ]
                    )
                vitals_table = Table(vitals_rows, colWidths=[0.9 * inch] + [0.85 * inch] * 6)
                vitals_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#27ae60")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                        ]
                    )
                )
                elements.append(vitals_table)

            # Disclaimer
            elements.append(Spacer(1, 20))
            disclaimer_style = ParagraphStyle("Disclaimer", parent=body_style, fontSize=8, textColor=colors.grey, italic=True)
            elements.append(
                Paragraph(
                    "Clinical Disclaimer: This report is generated by an AI-assisted screening tool "
                    "for clinical decision support only. It does not constitute a medical diagnosis. "
                    "Please consult a qualified healthcare professional.",
                    disclaimer_style,
                )
            )

            doc.build(elements)
            buffer.seek(0)
            filename = f"screening_report_{session_id}.pdf"
            return StreamingResponse(
                buffer,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as exc:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # B. HTML Report Format (Default web view with XSS sanitization)
    # ─────────────────────────────────────────────────────────────────────────
    patient_name = sanitize_html_text(patient.name) if patient else "Anonymous / Unregistered"
    patient_age = f"{patient.age} yrs" if patient and patient.age is not None else "N/A"
    patient_gender = sanitize_html_text(patient.gender) if patient and patient.gender else "N/A"
    overall_risk_sanitized = sanitize_html_text(sess.overall_risk_level or "UNKNOWN")

    eye_html = ""
    if eye:
        prob_rows = ""
        if eye.all_probabilities:
            for k, v in eye.all_probabilities.items():
                k_clean = sanitize_html_text(k.replace('_', ' ').title())
                prob_rows += f"<tr><td>{k_clean}</td><td>{float(v)*100:.2f}%</td></tr>"

        gcam_url_safe = sanitize_html_text(eye.gradcam_path)
        fundus_url_safe = sanitize_html_text(eye.image_path)
        gcam_img = f'<img src="{gcam_url_safe}" style="max-width:250px; border-radius:4px;" />' if eye.gradcam_path else "<i>Not computed</i>"
        fundus_img = f'<img src="{fundus_url_safe}" style="max-width:250px; border-radius:4px;" />' if eye.image_path else "<i>No image</i>"
        pred_safe = sanitize_html_text(eye.prediction.upper() if eye.prediction else "UNKNOWN")
        risk_safe = sanitize_html_text(eye.risk_level or "UNKNOWN")

        eye_html = f"""
        <div class="card">
            <h3>Eye Disease Assessment (DenseNet201)</h3>
            <p><strong>Predicted Condition:</strong> <span class="badge">{pred_safe}</span></p>
            <p><strong>Confidence:</strong> {float(eye.confidence or 0)*100:.2f}%</p>
            <p><strong>Risk Level:</strong> {risk_safe}</p>
            <table class="subtable">
                <thead><tr><th>Condition</th><th>Probability</th></tr></thead>
                <tbody>{prob_rows}</tbody>
            </table>
            <div style="display:flex; gap:16px; margin-top:12px;">
                <div><p><strong>Fundus:</strong></p>{fundus_img}</div>
                <div><p><strong>Grad-CAM:</strong></p>{gcam_img}</div>
            </div>
        </div>
        """
    else:
        eye_html = "<div class='card'><p><i>No retinal eye screening recorded for this session.</i></p></div>"

    vitals_html = ""
    if vitals_list:
        v_rows = ""
        for v in vitals_list:
            hr_stat = sanitize_html_text(v.hr_status)
            spo2_stat = sanitize_html_text(v.spo2_status)
            temp_stat = sanitize_html_text(v.temp_status)
            ov_stat = sanitize_html_text(v.overall_status)
            v_rows += f"""
            <tr>
                <td>{v.timestamp.strftime('%H:%M:%S UTC')}</td>
                <td>{v.heart_rate_bpm:.1f} BPM ({hr_stat})</td>
                <td>{v.spo2_percent:.1f}% ({spo2_stat})</td>
                <td>{v.temperature_celsius:.1f}°C ({temp_stat})</td>
                <td><strong>{ov_stat}</strong></td>
            </tr>
            """
        vitals_html = f"""
        <div class="card">
            <h3>Physiological Vital Signs</h3>
            <table class="subtable">
                <thead><tr><th>Time</th><th>Heart Rate</th><th>SpO2</th><th>Temperature</th><th>Overall</th></tr></thead>
                <tbody>{v_rows}</tbody>
            </table>
        </div>
        """
    else:
        vitals_html = "<div class='card'><p><i>No vital signs recorded for this session.</i></p></div>"

    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Clinical Screening Report - Session #{sess.id}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #1e293b; padding: 24px; }}
            .report-container {{ max-width: 800px; margin: 0 auto; background: white; padding: 32px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-top: 0; }}
            .card {{ background: #f1f5f9; border-radius: 6px; padding: 16px; margin-bottom: 16px; }}
            .badge {{ background: #2563eb; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }}
            table.subtable {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
            table.subtable th, table.subtable td {{ text-align: left; padding: 8px; border-bottom: 1px solid #cbd5e1; }}
            .disclaimer {{ font-size: 0.8em; color: #64748b; margin-top: 24px; font-style: italic; }}
            .actions {{ margin-bottom: 16px; }}
            .btn-pdf {{ display: inline-block; background: #e11d48; color: white; padding: 8px 16px; border-radius: 4px; text-decoration: none; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <div class="actions">
                <a href="?format=pdf" class="btn-pdf">Download Official PDF Report</a>
            </div>
            <h1>Clinical Screening Report</h1>
            <div class="card">
                <p><strong>Session ID:</strong> #{sess.id} | <strong>Date:</strong> {sess.timestamp.strftime('%Y-%m-%d %H:%M UTC')}</p>
                <p><strong>Patient:</strong> {patient_name} (Age: {patient_age}, Gender: {patient_gender})</p>
                <p><strong>Overall Risk:</strong> <span class="badge">{overall_risk_sanitized}</span></p>
            </div>
            {eye_html}
            {vitals_html}
            <div class="disclaimer">
                Clinical Disclaimer: This report is generated by an AI-assisted multimodal screening tool
                utilizing DenseNet201 deep learning for clinical decision support.
                It does not constitute a standalone medical diagnosis. Consult a licensed physician.
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=status.HTTP_200_OK)
