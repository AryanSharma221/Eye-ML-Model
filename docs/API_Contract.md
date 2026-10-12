# Multimodal Health Screening & Analysis System — API Contract & Integration Guide

**Version:** 1.0.0  
**Target Environment:** Python 3.12, PyTorch 2.6.0+cu124, FastAPI 0.115, React 18 + Vite 5  
**Model Checkpoint:** `ml/weights/eye_model.pth` (DenseNet201, 87.81% Test Accuracy on 894 held-out images)  
**Target Classes:** `["cataract", "diabetic_retinopathy", "glaucoma", "normal"]`

---

## 1. Architectural Principles

1. **Modality Independence Principle:**
   - Retinal fundus image evaluation and physiological vital signs assessment are strictly isolated streams.
   - Vital signs never influence eye disease predictions, and image quality failures never abort vital signs triage.
2. **Hard Quality Gate:**
   - Retinal images undergo automated structural checks (Laplacian variance $\ge 100$, brightness $[30, 225]$, contrast $\ge 20$, circular contour) before deep learning inference.
   - Images failing quality validation are immediately rejected with actionable guidance; no model inference or Grad-CAM computation occurs on invalid imagery.
3. **Model Warm-Up at Startup:**
   - The DenseNet201 model weights are pre-warmed once during FastAPI lifespan startup into GPU/CPU memory. Subsequent inference calls incur zero checkpoint reloading latency.
4. **Grad-CAM Saliency Isolation:**
   - Visual explainability heatmaps are generated from the final convolutional block (`features.denseblock4`). Any potential failure during gradient backpropagation is caught safely and does not corrupt the classification response.
5. **Privacy & Security Integrity:**
   - Internal stack traces, raw binary image payloads, and filesystem paths are never exposed to clients in API responses.

---

## 2. API Endpoints

### 2.1 Health & Readiness
- **Route:** `GET /api/v1/health` (also `GET /health`)
- **Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "checkpoint_available": true,
  "supported_classes": ["cataract", "diabetic_retinopathy", "glaucoma", "normal"],
  "timestamp": "2026-10-12T00:00:00Z"
}
```

---

### 2.2 Unified Multimodal Screening
- **Route:** `POST /api/v1/screen`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `image` (file, optional): Retinal fundus image (JPG, JPEG, PNG, max 30 MB).
  - `heart_rate` (float, optional): Heart rate in BPM ($30 \le \text{HR} \le 220$).
  - `spo2` (float, optional): Blood oxygen saturation in % ($50 \le \text{SpO2} \le 100$).
  - `temperature` (float, optional): Core body temperature in °C ($30.0 \le \text{Temp} \le 45.0$).
  - `patient_id` (int, optional): Patient identity to persist screening under.
  - `session_id` (int, optional): Existing session identifier.
  - `include_gradcam` (bool, default `true`): Whether to generate saliency heatmap.

- **Response (200 OK):**
```json
{
  "status": "success",
  "screening_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "timestamp": "2026-10-12T00:00:00Z",
  "duration_ms": 78.4,
  "session_id": 1,
  "overall_risk_level": "LOW",
  "eye_screening": {
    "status": "success",
    "prediction": "normal",
    "class_index": 3,
    "confidence": 0.985,
    "risk_level": "LOW",
    "all_probabilities": {
      "cataract": 0.003,
      "diabetic_retinopathy": 0.007,
      "glaucoma": 0.005,
      "normal": 0.985
    },
    "quality_checks": {
      "overall_pass": true,
      "blur_score": 220.46,
      "brightness": 75.2,
      "contrast": 48.9,
      "retinal_region_detected": true
    },
    "gradcam_image": "data:image/png;base64,...",
    "gradcam_url": "/uploads/gradcam/gradcam_...png",
    "image_url": "/uploads/fundus/...jpg"
  },
  "vitals_screening": {
    "status": "success",
    "heart_rate": 72.0,
    "spo2": 98.5,
    "temperature": 36.6,
    "hr_status": "Normal",
    "spo2_status": "Normal",
    "temp_status": "Normal",
    "overall_status": "Normal",
    "risk_level": "LOW",
    "is_abnormal": false,
    "summary": "All vital parameters are within normal physiological range."
  },
  "disclaimer": "This system is an automated multimodal health screening prototype engineered for clinical triage research and decision support. It does not provide definitive autonomous medical diagnoses. All flagged findings must be reviewed by qualified ophthalmic and medical practitioners."
}
```

---

### 2.3 Discrete Eye Disease Screening
- **Route:** `POST /api/v1/predict_eye`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `image` (file, required): Retinal image file.
  - `session_id` (int, optional): Attached session.
  - `include_gradcam` (bool, default `true`).
- **Quality Rejection (400 Bad Request):**
```json
{
  "status": "rejected",
  "reason": "Excessive image blur (blur score 12.3 < threshold 100.0).",
  "quality_checks": {
    "overall_pass": false,
    "blur_score": 12.3,
    "brightness": 54.2,
    "contrast": 32.1,
    "retinal_region_detected": false
  },
  "image_url": "/uploads/fundus/...jpg"
}
```

---

### 2.4 Discrete Physiological Vital Signs Submission
- **Route:** `POST /api/v1/submit_vitals`
- **Content-Type:** `application/json`
- **Payload:**
```json
{
  "session_id": 1,
  "heart_rate_bpm": 115.0,
  "spo2_percent": 97.0,
  "temperature_celsius": 38.4
}
```
- **Response (200 OK):**
```json
{
  "status": "success",
  "heart_rate": { "value": 115.0, "status": "Higher than usual", "unit": "BPM" },
  "spo2": { "value": 97.0, "status": "Normal", "unit": "%" },
  "temperature": { "value": 38.4, "status": "Higher than usual", "unit": "°C" },
  "hr_status": "Higher than usual",
  "spo2_status": "Normal",
  "temp_status": "Higher than usual",
  "overall_status": "Elevated HR & Body Temp",
  "risk_level": "MODERATE",
  "is_abnormal": true,
  "summary": "Elevated heart rate and body temperature detected.",
  "timestamp": "2026-10-12T00:00:00Z"
}
```

---

### 2.5 Clinical Sessions & Printable Report
- **List Sessions:** `GET /api/v1/sessions`
- **Session Details:** `GET /api/v1/sessions/{id}`
- **Printable Clinical Summary Report:** `GET /api/v1/sessions/{id}/report` (returns rich HTML with styling, patient demographics, 4-class probabilities, and visual heatmap links).

---

## 3. End-to-End Scenarios Validation Matrix

| Scenario | Input Modalities | Expected HTTP Code | Eye Screening Outcome | Vitals Outcome | Overall Risk | Verified |
|:---|:---|:---:|:---|:---|:---:|:---:|
| **A** | Normal Fundus + Normal Vitals | `200 OK` | `normal` (Confidence >90%, Grad-CAM generated) | All Normal | `LOW` | ✅ Pass |
| **B** | Pathological Fundus + Abnormal Vitals | `200 OK` | `cataract` / `glaucoma` (Grad-CAM generated) | Tachycardia + Hypoxia + Fever | `HIGH` | ✅ Pass |
| **C** | Blurry Image + Valid Vitals | `200 OK` (`/screen`) / `400` (`/predict_eye`) | `rejected` (Inference halted, no Grad-CAM) | Evaluated independently | `LOW` / `UNKNOWN` | ✅ Pass |
| **D** | Valid Image + Missing Vitals | `200 OK` | Evaluated successfully | `skipped` | `LOW` / `MODERATE` | ✅ Pass |
| **E** | Corrupted / Invalid File | `400 Bad Request` | Image decode rejection | N/A | N/A | ✅ Pass |
| **F** | Missing File / Empty Request | `400` / `422` | Validation rejection | N/A | N/A | ✅ Pass |
