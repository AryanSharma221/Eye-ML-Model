# Multimodal Health Screening & Analysis System

## 1. Project Overview & Purpose

The **Multimodal Health Screening & Analysis System** is an engineering project designed to combine an image-based eye-disease screening component with independent physiological monitoring. The objective of the platform is to accept clinical fundus photographs and vital sign measurements, evaluate each data stream independently using appropriate machine learning approaches, persist session history, and surface holistic summaries via an operator dashboard.

> ### ⚠️ Medical Disclaimer
> **This system is strictly intended as a preliminary screening and assessment tool, not a diagnostic medical device.**
> All planned outputs, classifications, and indicators are designed to provide decision support and must never supersede clinical judgment or evaluation by certified medical practitioners.

---

## 2. Team Responsibility Division

The development and implementation of this system are divided across three specialized roles:

| Member | Focus Area | Core Responsibilities |
| :--- | :--- | :--- |
| **Member 1 (Current Role)** | **Machine Learning & AI Engineering** | Data preparation, image quality filtering, deep learning classification pipeline, visual explainability (Grad-CAM), and standalone offline evaluation. |
| **Member 2** | **Backend Engineering & Data Persistence** | REST API service, relational database schemas, session orchestration, model execution wrapping, and clinical report export. |
| **Member 3** | **Frontend Engineering & Interface Design** | React operator dashboard, image intake component, explainability visualization, vital signs monitoring panels, and longitudinal session history. |

---

## 3. Repository Structure

```
multimodal-health-screening/
├── ml/
│   ├── __init__.py
│   ├── weights/          # Local directory for model weights (excluded from Git)
│   │   └── .gitkeep
│   ├── dataset/          # Local directory for dataset splits (excluded from Git)
│   │   └── .gitkeep
│   └── tests/            # Test suite for ML modules
│       └── __init__.py
├── backend/
│   ├── __init__.py
│   ├── routes/           # API route handlers
│   │   └── __init__.py
│   └── uploads/          # Local upload directory for session images (excluded from Git)
│       └── gradcam/
│           └── .gitkeep
├── frontend/             # Client-side web application
├── docs/                 # Project documentation and specifications
├── tests/                # System integration and end-to-end tests
├── .gitignore            # Git exclusion rules
└── README.md             # Project overview and repository documentation
```

---

## 4. Implemented Machine Learning Modules (Member 1 Deliverables)

The following production-ready ML modules are fully implemented, tested, and ready for backend integration:

1. **Image Quality Gate (`ml/quality_gate.py`):** Structural and heuristic checks to validate fundus image blur (Laplacian variance), illumination, contrast, resolution (up to $5000 \times 5000$, $\le 30$ MB), and retinal circular contour prior to model inference.
2. **Preprocessing Pipeline (`ml/preprocessing.py`):** Converts validated retinal images into normalized, resized tensors ($3 \times 224 \times 224$) with ImageNet mean and variance scaling, with decompression bomb protection.
3. **Eye Disease Classifier (`ml/eye_model.py`):** PyTorch `DenseNet201` trained on 5,889 patient-stratified images across the canonical 4 classes: `["cataract", "diabetic_retinopathy", "glaucoma", "normal"]`. Delivers **90.29% validation accuracy** and **87.81% held-out test accuracy**.
4. **Visual Explainability (`ml/gradcam.py`):** Generates gradient-weighted class activation maps from the final dense block (`features.denseblock4`), blended onto native fundus imagery and encoded as Base64 PNG strings.
5. **Physiological Evaluator (`ml/physio_model.py`):** Rule-based clinical triage engine for Heart Rate, SpO2, and Body Temperature with strict input type-safety (rejects boolean coercion, strings, NaN, infinity).
6. **Data Integrity & Audit (`ml/data_integrity.py`):** Cryptographic SHA-256 manifest and patient isolation auditor ensuring 0% cross-split leakage across 5,889 images.
7. **Performance Benchmark Suite (`ml/benchmark.py`):** Profiles latency, memory footprints, and batched throughput across CPU and CUDA backends (44.56 ms steady-state inference, 404.3 img/s batch throughput on RTX 4060).
8. **Structured Telemetry & Monitoring (`ml/monitoring.py`):** Structured operational logging with privacy-preserving path sanitization, credential redaction, and vital measurement masking.
9. **Offline Training & Evaluation (`ml/train_eye.py`, `ml/evaluate.py`):** Training loop with class-weighted loss, plateau LR scheduling, atomic checkpointing, and held-out test evaluation with 4x4 confusion matrix export.
10. **Demonstration Runner (`ml/demo.py`):** Standalone demo CLI supporting real inference mode on held-out test split images, pipeline validation mode, and automatic Grad-CAM PNG persistence.
11. **Checkpoint Selection & Lifecycle Manager (`ml/checkpoint_manager.py`):** Safe, auditable model inspection, atomic pointer updates in `checkpoint_selection.json`, clinical sensitivity regression guards, and reversible rollbacks.

*For complete usage instructions and API documentation, refer to [`ml/README.md`](ml/README.md). For the comprehensive handoff and release assessment, refer to [`HANDOFF.md`](HANDOFF.md).*

---

## 5. Quickstart for Developers

### 5.1 ML Diagnostics & Tests
```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run first-run environment diagnostics (19/19 checks)
python -m ml.diagnostics

# 3. Run full automated integration test suite (41 passing tests)
python -m pytest tests/ -v
```

### 5.2 Running the Integrated Full-Stack Application
```powershell
# Start FastAPI backend (port 8000, pre-warms DenseNet201)
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Start Vite React frontend development server (port 5173)
cd frontend
npm run dev
```

### 5.3 Production Deployment & Containerization
For complete deployment instructions across Docker, Cloud Run, Render, Railway, or Vercel, refer to [`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md):
```bash
# Build and run unified production container
docker build -t health-screening .
docker run -p 8000:8000 health-screening
```

---

## 6. Dataset and Model Weight Policy

To ensure repository cleanliness, security, and compliance with data governance standards:
- **No medical images or raw datasets** are committed to Git. Local dataset directories are ignored via `.gitignore`.
- **No binary weight files (`.pth`, `.pkl`, `.onnx`)** are tracked in version control. Checkpoints are stored in secure local storage (`ml/weights/`).
- **No patient identifiers or screening artifacts** may be added to Git history.

---

## 7. Development Status & Clinical Validity Note

- **Current Status:** ML/AI Engineering modules (Member 1) are **complete, fully trained, demonstrable, and verified** (417 passing tests across 25 test suites, acceptance report verdict: `READY_FOR_INTEGRATION` / `READY_FOR_HANDOFF`). Checkpoints and test splits are validated for Member 2 backend handoff.
- **Model Training:** DenseNet201 model training is completed on 5,889 patient-stratified images. Best validation accuracy: **90.29%**, held-out test accuracy: **87.81%** ($N=894$).
- **Clinical Validity:** This system is an automated multimodal health screening prototype engineered for clinical triage research and decision support. It does **not** provide definitive autonomous medical diagnoses. All flagged findings must be reviewed by qualified ophthalmic and medical practitioners.
