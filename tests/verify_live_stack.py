"""
Live Verification Script for Multimodal Health Screening E2E Stack.
Validates live HTTP communication across Frontend (5173) and Backend (8000).
"""

import io
import httpx
from PIL import Image, ImageFilter

BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://localhost:5173"
TEST_IMG = r"C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset\test\normal\kaggle_2587_right.jpg"

def main():
    print("[1/8] Probing Backend Direct Health...")
    r = httpx.get(f"{BACKEND_URL}/health", timeout=5)
    assert r.status_code == 200, f"Backend health failed: {r.status_code}"
    h_data = r.json()
    print(f"      Backend Status: {h_data['status']}, Device: {h_data['device']}, Classes: {h_data['supported_classes']}")

    print("[2/8] Probing Frontend Vite Root...")
    r = httpx.get(f"{FRONTEND_URL}/", timeout=5)
    assert r.status_code == 200, f"Frontend root failed: {r.status_code}"
    assert "<div id=\"root\">" in r.text or "<div id='root'>" in r.text
    print("      Frontend HTML served successfully.")

    print("[3/8] Probing Vite API Proxy (/api/v1/health)...")
    r = httpx.get(f"{FRONTEND_URL}/api/v1/health", timeout=5)
    assert r.status_code == 200, f"Vite proxy failed: {r.status_code}"
    print("      Vite reverse proxy to FastAPI is functional.")

    print("[4/8] Testing Patient Registration via Proxy...")
    p_payload = {"name": "Live Verification User", "age": 42, "gender": "Non-binary", "contact": "live@test.local"}
    r = httpx.post(f"{FRONTEND_URL}/api/v1/patients", json=p_payload, timeout=5)
    assert r.status_code == 201, f"Patient create failed: {r.status_code}"
    patient = r.json()
    p_id = patient["id"]
    print(f"      Patient registered with ID #{p_id}")

    print("[5/8] Testing Full Multimodal Screening (Normal Eye + Normal Vitals)...")
    with open(TEST_IMG, "rb") as f:
        img_bytes = f.read()
    r = httpx.post(
        f"{FRONTEND_URL}/api/v1/screen",
        files={"image": ("live_retina.jpg", img_bytes, "image/jpeg")},
        data={"patient_id": str(p_id), "heart_rate": "74.0", "spo2": "98.5", "temperature": "36.8", "include_gradcam": "true"},
        timeout=15,
    )
    assert r.status_code == 200, f"Screening failed: {r.status_code}, {r.text}"
    scr_data = r.json()
    sess_id = scr_data["session_id"]
    eye_res = scr_data["eye_screening"]
    vitals_res = scr_data["vitals_screening"]
    print(f"      Screening Session #{sess_id} created.")
    print(f"      Eye: {eye_res['prediction']} ({eye_res['confidence']:.3f}), Grad-CAM: {bool(eye_res.get('gradcam_image'))}")
    print(f"      Vitals: HR={vitals_res['heart_rate']} ({vitals_res['hr_status']}), Risk={vitals_res['risk_level']}")
    print(f"      Overall Risk: {scr_data['overall_risk_level']}, Duration: {scr_data['duration_ms']}ms")

    print("[6/8] Testing Blurry Image Quality Gate Rejection...")
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    img = img.filter(ImageFilter.GaussianBlur(radius=20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    r = httpx.post(
        f"{FRONTEND_URL}/api/v1/predict_eye",
        files={"image": ("blurry.jpg", buf.getvalue(), "image/jpeg")},
        timeout=10,
    )
    assert r.status_code == 400, f"Expected 400 rejection, got {r.status_code}"
    rej_data = r.json()
    assert rej_data["status"] == "rejected"
    print(f"      Quality Gate properly rejected blurry image: {rej_data['reason']}")

    print("[7/8] Testing ReportLab PDF Report Generation...")
    r = httpx.get(f"{FRONTEND_URL}/api/v1/sessions/{sess_id}/report?format=pdf", timeout=10)
    assert r.status_code == 200, f"PDF report failed: {r.status_code}"
    assert r.headers.get("content-type") == "application/pdf"
    assert r.content.startswith(b"%PDF")
    print(f"      ReportLab PDF generated successfully ({len(r.content)} bytes).")

    print("[8/8] Testing HTML Report View...")
    r = httpx.get(f"{FRONTEND_URL}/api/v1/sessions/{sess_id}/report?format=html", timeout=10)
    assert r.status_code == 200, f"HTML report failed: {r.status_code}"
    assert "text/html" in r.headers.get("content-type", "")
    assert "Clinical Screening Report" in r.text
    print("      HTML Clinical Report generated and rendered successfully.")

    print("\n>>> ALL 8 LIVE END-TO-END VERIFICATION CHECKS PASSED PERFECTLY! <<<")

if __name__ == "__main__":
    main()
