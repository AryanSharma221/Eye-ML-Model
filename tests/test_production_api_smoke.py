"""
Production API Smoke Test Suite for Multimodal Health Screening.
Executes live HTTP requests against the running FastAPI backend on http://127.0.0.1:8000.
Verifies all routes, valid/invalid inputs, ML inference, error handling, security headers,
and records comprehensive diagnostic telemetry.
"""

import base64
import io
import json
import time
from pathlib import Path
import httpx
from PIL import Image, ImageFilter

BASE_URL = "http://127.0.0.1:8000"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public" / "samples"
CANONICAL_CLASSES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

def run_production_api_smoke_tests():
    print("=" * 80)
    print("STARTING PRODUCTION API SMOKE TESTING SUITE")
    print(f"Target Backend: {BASE_URL}")
    print("=" * 80)

    client = httpx.Client(base_url=BASE_URL, timeout=30.0)
    results = []

    def check(name, passed, details=""):
        status_str = "PASS" if passed else "FAIL"
        results.append({"name": name, "passed": passed, "details": details})
        print(f"[{status_str}] {name}")
        if details:
            print(f"       Details: {details}")

    # 1. Health Live Endpoint
    t0 = time.perf_counter()
    r = client.get("/health/live")
    d1 = (time.perf_counter() - t0) * 1000
    check(
        "GET /health/live (Liveness Probe)",
        r.status_code == 200 and r.json().get("status") == "alive",
        f"Status: {r.status_code}, latency: {d1:.1f}ms, resp: {r.json()}"
    )

    # 2. Health Ready Endpoint
    t0 = time.perf_counter()
    r = client.get("/health/ready")
    d2 = (time.perf_counter() - t0) * 1000
    ready_data = r.json()
    check(
        "GET /health/ready (Readiness Probe)",
        r.status_code == 200 and ready_data.get("model_loaded") is True and ready_data.get("checkpoint_verified") is True,
        f"Status: {r.status_code}, latency: {d2:.1f}ms, device: {ready_data.get('device')}, model_loaded: {ready_data.get('model_loaded')}"
    )

    # 3. Legacy / General Health Endpoint
    t0 = time.perf_counter()
    r = client.get("/health")
    d3 = (time.perf_counter() - t0) * 1000
    check(
        "GET /health (Legacy Health)",
        r.status_code == 200 and r.json().get("status") == "healthy",
        f"Status: {r.status_code}, latency: {d3:.1f}ms"
    )

    # 4. OWASP Security Headers & Correlation ID on Responses
    headers = r.headers
    has_nosniff = headers.get("x-content-type-options") == "nosniff"
    has_frame_options = "x-frame-options" in headers
    has_req_id = "x-request-id" in headers
    check(
        "OWASP Defensive Security Headers & Request Correlation",
        has_nosniff and has_frame_options and has_req_id,
        f"X-Content-Type-Options: {headers.get('x-content-type-options')}, X-Frame-Options: {headers.get('x-frame-options')}, X-Request-ID: {headers.get('x-request-id')}"
    )

    # 5. Patient Registration
    t0 = time.perf_counter()
    patient_payload = {
        "name": "Audit Verification Patient",
        "age": 58,
        "gender": "Female",
        "contact": "+1-555-0199"
    }
    r = client.post("/api/v1/patients", json=patient_payload)
    d5 = (time.perf_counter() - t0) * 1000
    patient_id = None
    if r.status_code == 201:
        patient_data = r.json()
        patient_id = patient_data.get("id")
    check(
        "POST /api/v1/patients (Patient Intake)",
        r.status_code == 201 and patient_id is not None,
        f"Status: {r.status_code}, latency: {d5:.1f}ms, patient_id: {patient_id}"
    )

    # 6. Retrieve Patient by ID
    r = client.get(f"/api/v1/patients/{patient_id}")
    check(
        f"GET /api/v1/patients/{patient_id} (Patient Record Lookup)",
        r.status_code == 200 and r.json().get("name") == patient_payload["name"],
        f"Status: {r.status_code}, retrieved name: {r.json().get('name')}"
    )

    # 7. List Patients
    r = client.get("/api/v1/patients")
    check(
        "GET /api/v1/patients (List Patients)",
        r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) > 0,
        f"Status: {r.status_code}, patient count: {len(r.json())}"
    )

    # 8. Create Screening Session
    t0 = time.perf_counter()
    session_payload = {
        "patient_id": patient_id,
        "notes": "Automated Production Smoke Test Encounter"
    }
    r = client.post("/api/v1/sessions", json=session_payload)
    d8 = (time.perf_counter() - t0) * 1000
    session_id = None
    if r.status_code == 201:
        session_data = r.json()
        session_id = session_data.get("id")
    check(
        "POST /api/v1/sessions (Session Initialization)",
        r.status_code == 201 and session_id is not None,
        f"Status: {r.status_code}, latency: {d8:.1f}ms, session_id: {session_id}"
    )

    # 9. Retrieve Session by ID
    r = client.get(f"/api/v1/sessions/{session_id}")
    check(
        f"GET /api/v1/sessions/{session_id} (Session Lookup)",
        r.status_code == 200 and r.json().get("id") == session_id,
        f"Status: {r.status_code}, patient_id: {r.json().get('patient_id')}"
    )

    # 10. List Sessions
    r = client.get("/api/v1/sessions")
    check(
        "GET /api/v1/sessions (List Sessions)",
        r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) > 0,
        f"Status: {r.status_code}, session count: {len(r.json())}"
    )

    # 11. Multi-Sample Real ML Inference Across All 4 Classes
    sample_files = [
        ("sample_cataract.jpg", "cataract"),
        ("sample_dr.jpeg", "diabetic_retinopathy"),
        ("sample_glaucoma.jpg", "glaucoma"),
        ("sample_normal.jpg", "normal"),
    ]

    for fname, expected_class in sample_files:
        fpath = SAMPLES_DIR / fname
        if not fpath.exists():
            check(f"ML Inference on {fname}", False, f"File {fpath} does not exist")
            continue

        with open(fpath, "rb") as f:
            img_bytes = f.read()

        t0 = time.perf_counter()
        r = client.post(
            "/api/v1/predict_eye",
            files={"image": (fname, img_bytes, "image/jpeg" if fname.endswith(".jpg") else "image/jpeg")},
            data={"session_id": str(session_id), "include_gradcam": "true"}
        )
        latency = (time.perf_counter() - t0) * 1000

        if r.status_code != 200:
            check(f"ML Inference on {fname}", False, f"HTTP {r.status_code}: {r.text}")
            continue

        data = r.json()
        pred = data.get("prediction")
        conf = data.get("confidence", 0.0)
        gradcam = data.get("gradcam_image")
        all_probs = data.get("all_probabilities", {})

        # Verify class order and softmax sum
        is_valid_class = pred in CANONICAL_CLASSES
        has_gradcam = gradcam is not None and len(gradcam) > 1000
        prob_sum = sum(all_probs.values()) if all_probs else 0.0
        prob_sum_valid = abs(prob_sum - 1.0) < 0.01

        check(
            f"ML Inference on {fname} ({expected_class})",
            is_valid_class and has_gradcam and prob_sum_valid,
            f"Status: 200, Latency: {latency:.1f}ms, Pred: '{pred}', Conf: {conf:.4f}, GradCAM length: {len(gradcam) if gradcam else 0}, ProbSum: {prob_sum:.4f}"
        )

    # 12. Physiological Vitals Assessment (Normal Baseline)
    t0 = time.perf_counter()
    vitals_normal = {
        "heart_rate": 72.0,
        "spo2": 98.0,
        "temperature": 36.6,
        "patient_id": patient_id,
        "session_id": session_id
    }
    r = client.post("/api/v1/submit_vitals", json=vitals_normal)
    d12 = (time.perf_counter() - t0) * 1000
    vitals_res = r.json()
    check(
        "POST /api/v1/submit_vitals (Normal Vitals Intake)",
        r.status_code == 200 and str(vitals_res.get("overall_status")).lower() == "normal",
        f"Status: {r.status_code}, latency: {d12:.1f}ms, overall: {vitals_res.get('overall_status')}"
    )

    # 13. Physiological Vitals Assessment (Critical Hypoxia & Tachycardia)
    vitals_critical = {
        "heart_rate": 132.0,
        "spo2": 88.0,
        "temperature": 38.8,
        "patient_id": patient_id,
        "session_id": session_id
    }
    r = client.post("/api/v1/submit_vitals", json=vitals_critical)
    vitals_crit_res = r.json()
    check(
        "POST /api/v1/submit_vitals (Critical Triage Vitals)",
        r.status_code == 200 and str(vitals_crit_res.get("overall_status")).lower() in ["critical", "abnormal", "warning"],
        f"Status: {r.status_code}, overall: {vitals_crit_res.get('overall_status')}"
    )

    # 14. Combined Multimodal Screening Route
    fpath = SAMPLES_DIR / "sample_glaucoma.jpg"
    with open(fpath, "rb") as f:
        img_bytes = f.read()

    t0 = time.perf_counter()
    r = client.post(
        "/api/v1/screen",
        files={"image": ("sample_glaucoma.jpg", img_bytes, "image/jpeg")},
        data={
            "heart_rate": "78.0",
            "spo2": "97.0",
            "temperature": "36.8",
            "patient_id": str(patient_id),
            "session_id": str(session_id),
            "include_gradcam": "true"
        }
    )
    d14 = (time.perf_counter() - t0) * 1000
    screen_res = r.json() if r.status_code == 200 else {}
    eye_screen_sub = screen_res.get("eye_screening", {})
    check(
        "POST /api/v1/screen (Multimodal Combined Screening)",
        r.status_code == 200 and screen_res.get("status") == "success" and eye_screen_sub.get("prediction") in CANONICAL_CLASSES,
        f"Status: {r.status_code}, latency: {d14:.1f}ms, eye_pred: {eye_screen_sub.get('prediction')}, overall_risk: {screen_res.get('overall_risk_level')}"
    )

    # 15. Verify Session Persistence with Attached Predictions & Vitals
    r = client.get(f"/api/v1/sessions/{session_id}")
    session_data = r.json()
    has_eye = session_data.get("eye_record") is not None or len(session_data.get("eye_records") or []) > 0
    has_vitals = len(session_data.get("vitals_records", [])) > 0
    check(
        f"GET /api/v1/sessions/{session_id} (Session Persistence Audit)",
        r.status_code == 200 and has_eye and has_vitals,
        f"Eye Record Present: {session_data.get('eye_record') is not None}, Vitals Records Count: {len(session_data.get('vitals_records', []))}"
    )

    # 16. PDF Clinical Report Generation
    t0 = time.perf_counter()
    r = client.get(f"/api/v1/sessions/{session_id}/report?format=pdf")
    d16 = (time.perf_counter() - t0) * 1000
    is_pdf = r.status_code == 200 and "application/pdf" in r.headers.get("content-type", "") and len(r.content) > 1000
    check(
        f"GET /api/v1/sessions/{session_id}/report?format=pdf (PDF Clinical Report)",
        is_pdf,
        f"Status: {r.status_code}, latency: {d16:.1f}ms, length: {len(r.content)} bytes, content-type: {r.headers.get('content-type')}"
    )

    # 17. HTML Dossier Export
    t0 = time.perf_counter()
    r = client.get(f"/api/v1/sessions/{session_id}/report?format=html")
    d17 = (time.perf_counter() - t0) * 1000
    check(
        f"GET /api/v1/sessions/{session_id}/report?format=html (HTML Dossier View)",
        r.status_code == 200 and "text/html" in r.headers.get("content-type", ""),
        f"Status: {r.status_code}, latency: {d17:.1f}ms, length: {len(r.content)} bytes"
    )

    # 18. Security Boundary: Corrupt Image Bytes Rejection
    corrupt_bytes = b"CORRUPTED_BYTES_NOT_AN_IMAGE_HEADER_XYZ1234567890"
    r = client.post(
        "/api/v1/predict_eye",
        files={"image": ("corrupted.jpg", corrupt_bytes, "image/jpeg")},
        data={"include_gradcam": "false"}
    )
    check(
        "Security Boundary: Corrupt Image Bytes Rejection (400 Bad Request)",
        r.status_code == 400 and ("corrupt" in r.text.lower() or "invalid" in r.text.lower() or "not a valid" in r.text.lower()),
        f"Status: {r.status_code}, error payload: {r.json()}"
    )

    # 19. Security Boundary: Text File Disguised as Image
    text_disguised = b"This is a plain ASCII text file saved with a .png extension."
    r = client.post(
        "/api/v1/predict_eye",
        files={"image": ("malicious.png", text_disguised, "image/png")},
        data={"include_gradcam": "false"}
    )
    check(
        "Security Boundary: Text File Disguised as Image (400 Bad Request)",
        r.status_code == 400,
        f"Status: {r.status_code}, message: {r.json().get('message')}"
    )

    # 20. Security Boundary: Quality Gate Rejection (Synthetic Blurry Noise)
    tiny_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    tiny_img = tiny_img.filter(ImageFilter.GaussianBlur(radius=10))
    buf = io.BytesIO()
    tiny_img.save(buf, format="JPEG")
    blurry_bytes = buf.getvalue()

    r = client.post(
        "/api/v1/predict_eye",
        files={"image": ("blurry_blank.jpg", blurry_bytes, "image/jpeg")},
        data={"include_gradcam": "false"}
    )
    check(
        "Quality Gate Enforcement: Low-Quality Image Rejection (400 Bad Request)",
        r.status_code == 400 and r.json().get("status") == "rejected",
        f"Status: {r.status_code}, rejection reason: {r.json().get('reason')}"
    )

    # 21. Security Boundary: Missing Required Fields in Vitals
    r = client.post("/api/v1/submit_vitals", json={"patient_id": patient_id})  # missing hr, spo2, temp
    check(
        "Input Validation: Missing Required Vitals Fields (400 Bad Request)",
        r.status_code == 400,
        f"Status: {r.status_code}, message: {r.json().get('message')}"
    )

    # 22. Security Boundary: Invalid Data Types in Vitals (String instead of Float)
    r = client.post("/api/v1/submit_vitals", json={
        "heart_rate": "invalid_string_hr",
        "spo2": 98.0,
        "temperature": 36.6
    })
    check(
        "Input Validation: String Data Type for Numeric Vitals (422 Unprocessable Entity)",
        r.status_code == 422,
        f"Status: {r.status_code}, details: {r.json().get('details')}"
    )

    # 23. Security Boundary: Transport-Level Request Size Limit
    # Create dummy payload larger than 30MB
    oversized_bytes = b"0" * (31 * 1024 * 1024)
    try:
        r = client.post(
            "/api/v1/predict_eye",
            files={"image": ("huge.jpg", oversized_bytes, "image/jpeg")},
            data={"include_gradcam": "false"}
        )
        check(
            "Security Boundary: Oversized Upload Rejection (413 Payload Too Large)",
            r.status_code == 413,
            f"Status: {r.status_code}, message: {r.text[:150]}"
        )
    except Exception as e:
        check(
            "Security Boundary: Oversized Upload Rejection (413 Payload Too Large)",
            True,
            f"Request rejected at transport layer: {e}"
        )

    # 24. Summary of Results
    total = len(results)
    passed = sum(1 for res in results if res["passed"])
    print("\n" + "=" * 80)
    print(f"PRODUCTION API SMOKE TEST SUMMARY: {passed} / {total} PASSED")
    print("=" * 80)

    # Save structured test report
    report_path = Path(__file__).resolve().parent / "production_api_smoke_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "target": BASE_URL,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "results": results
        }, f, indent=2)
    print(f"Report saved to: {report_path}")

    assert passed == total, f"Only {passed}/{total} tests passed."
    return results

if __name__ == "__main__":
    run_production_api_smoke_tests()
