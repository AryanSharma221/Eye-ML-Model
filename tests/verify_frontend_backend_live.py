"""
Verification script for the integrated live stack (React/Vite Frontend + FastAPI Backend + ML).
Uses httpx (installed in venv).
"""
import io
import httpx
from PIL import Image, ImageFilter

def test_live_stack():
    client = httpx.Client(timeout=30.0)

    print("--- 1. Testing Frontend HTTP Serving ---")
    r = client.get("http://localhost:5173/")
    assert r.status_code == 200, f"Status {r.status_code}"
    assert "root" in r.text, "Root element missing"
    assert "RetinaPulse" in r.text or "Multimodal" in r.text
    print("  [PASS] Frontend served at http://localhost:5173/")

    print("--- 2. Testing Vite Dev Proxy to Backend /health ---")
    r = client.get("http://localhost:5173/health")
    assert r.status_code == 200, f"Status {r.status_code}"
    data = r.json()
    assert data["model_loaded"] is True
    assert "cataract" in data["supported_classes"]
    print(f"  [PASS] Health via Vite proxy: model_loaded={data['model_loaded']}, device={data['device']}")

    print("--- 3. Testing Public Sample Image Delivery ---")
    r = client.get("http://localhost:5173/samples/sample_glaucoma.jpg")
    assert r.status_code == 200, f"Status {r.status_code}"
    assert len(r.content) > 10000
    sample_bytes = r.content
    print(f"  [PASS] Sample image delivered ({len(sample_bytes)} bytes)")

    print("--- 4. Testing Patient Intake Creation ---")
    r = client.post("http://localhost:8000/api/v1/patients", json={
        "name": "Live Intake Patient",
        "age": 64,
        "gender": "Female",
        "contact": "+1 555 921 4402"
    })
    assert r.status_code == 201, f"Status {r.status_code}: {r.text}"
    patient = r.json()
    patient_id = patient["id"]
    print(f"  [PASS] Patient created with ID: {patient_id}")

    print("--- 5. Testing Screening Session Initialization ---")
    r = client.post("http://localhost:8000/api/v1/sessions", json={
        "patient_id": patient_id,
        "notes": "Automated E2E Verification Encounter"
    })
    assert r.status_code == 201, f"Status {r.status_code}: {r.text}"
    session = r.json()
    session_id = session["id"]
    print(f"  [PASS] Session initialized with ID: {session_id}")

    print("--- 6. Testing Real DenseNet201 + Grad-CAM Inference ---")
    r = client.post(
        "http://localhost:8000/api/v1/predict_eye",
        files={"image": ("sample_glaucoma.jpg", sample_bytes, "image/jpeg")},
        data={"session_id": str(session_id), "include_gradcam": "true"}
    )
    assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
    pred = r.json()
    assert pred["status"] == "success"
    assert pred["prediction"] == "glaucoma"
    assert pred["confidence"] > 0.90
    assert pred["gradcam_image"] is not None
    assert pred["quality_checks"]["overall_pass"] is True
    print(f"  [PASS] Real DenseNet201 CUDA inference: {pred['prediction']} ({pred['confidence']*100:.2f}%), Risk: {pred['risk_level']}")

    print("--- 7. Testing Physiological Vitals Telemetry Intake ---")
    r = client.post("http://localhost:8000/api/v1/submit_vitals", json={
        "session_id": session_id,
        "heart_rate_bpm": 76.5,
        "spo2_percent": 98.2,
        "temperature_celsius": 36.6
    })
    assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
    vitals = r.json()
    assert vitals["status"] == "success"
    assert vitals["overall_status"] == "Normal"
    print(f"  [PASS] Vitals telemetry recorded: HR={vitals['analysis']['heart_rate']['value']}, Overall={vitals['analysis']['overall_status']}")

    print("--- 8. Testing Longitudinal Encounter Query ---")
    r = client.get(f"http://localhost:8000/api/v1/patients/{patient_id}/sessions")
    assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
    history = r.json()
    assert len(history) >= 1
    assert history[0]["eye_record"]["prediction"] == "glaucoma"
    assert len(history[0]["vitals_records"]) >= 1
    print(f"  [PASS] Retrieved {len(history)} session(s) containing eye & vitals records")

    print("--- 9. Testing ReportLab Clinical PDF Dossier Generation ---")
    r = client.get(f"http://localhost:8000/api/v1/sessions/{session_id}/report?format=pdf")
    assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")
    assert len(r.content) > 1000
    print(f"  [PASS] Clinical PDF successfully compiled ({len(r.content)} bytes)")

    print("--- 10. Testing Quality Gate Rejection ---")
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    img = img.filter(ImageFilter.GaussianBlur(radius=20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    r = client.post(
        "http://localhost:8000/api/v1/predict_eye",
        files={"image": ("blur.jpg", buf.getvalue(), "image/jpeg")},
        data={"include_gradcam": "false"}
    )
    assert r.status_code == 400
    rej = r.json()
    assert rej["status"] == "rejected"
    assert "quality_checks" in rej
    assert rej["quality_checks"]["overall_pass"] is False
    print(f"  [PASS] Blurry image rejected by quality gate with HTTP 400: {rej['reason']}")

    print("\n=======================================================")
    print("ALL 10 LIVE FRONTEND-BACKEND INTEGRATION CHECKS PASSED!")
    print("=======================================================")

if __name__ == "__main__":
    test_live_stack()
