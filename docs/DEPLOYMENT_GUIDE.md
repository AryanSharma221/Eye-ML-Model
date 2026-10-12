# Multimodal Health Screening Platform — Deployment Guide

This guide describes how to deploy the **Multimodal Health Screening Platform** (DenseNet201 Retinal AI + Physiological Vital-Signs Triage + Clinical Reporting) reliably across cloud, container, and on-premises environments.

---

## 1. Architecture Overview

The system supports two primary deployment topologies:

```mermaid
flowchart LR
    subgraph OptionA["Option A: Unified Single Container (Recommended)"]
        BrowserA["Client Browser"] -->|Port 8000| NginxA["FastAPI Web Server"]
        NginxA -->|Serves Static Files| DistA["frontend/dist/ (React SPA)"]
        NginxA -->|API Routes| APILayerA["FastAPI Routes (/api/v1)"]
        APILayerA --> MLLayerA["DenseNet201 (CPU/CUDA)"]
        APILayerA --> DBA[(SQLite / PostgreSQL)]
    end
```

```mermaid
flowchart LR
    subgraph OptionB["Option B: Decoupled Multi-Host Deployment"]
        BrowserB["Client Browser"] -->|HTTPS| Vercel["Vercel / Netlify (Frontend SPA)"]
        BrowserB -->|CORS HTTPS API| CloudRun["Render / GCP Cloud Run (FastAPI Backend)"]
        CloudRun --> MLLayerB["DenseNet201 (CPU/CUDA)"]
        CloudRun --> DBB[(Managed PostgreSQL)]
    end
```

- **Option A (Unified Container):** The React frontend is compiled into static HTML/CSS/JS (`frontend/dist/`) and mounted at the root (`/`) of the FastAPI application. A single container image serves both the UI and the API. This eliminates cross-origin CORS latency, simplifies SSL termination, and runs on a single port.
- **Option B (Decoupled Services):** The React frontend is deployed to static hosting (Vercel, Netlify, Cloudflare Pages, AWS S3+CloudFront), while the FastAPI backend runs on a container platform (Render, Railway, Fly.io, AWS ECS, GCP Cloud Run). The frontend communicates with the backend via `VITE_BACKEND_URL` with explicit CORS headers.

---

## 2. System & Runtime Prerequisites

| Component | Minimum Version | Recommended / Verified | Notes |
| :--- | :--- | :--- | :--- |
| **Python** | 3.10 | 3.12.4 | Strict compatibility with PyTorch 2.4+ |
| **Node.js** | 18 LTS | 20 LTS (20.18.0) | Required only for building frontend |
| **RAM** | 2.0 GB | 4.0 GB+ | DenseNet201 (~74MB weights) requires ~1.2GB resident memory on CPU |
| **VRAM (Optional)** | 2.0 GB | 4.0 GB+ | NVIDIA GPU with CUDA 12.x for < 100ms inference; falls back cleanly to CPU |
| **Linux Libraries** | N/A | `libgl1`, `libglib2.0-0`, `curl` | Required for OpenCV (`cv2`) image decoding on Debian/Ubuntu/Alpine |
| **Storage** | 500 MB | 2.0 GB+ | For Docker base image, model checkpoint (73.7 MB), and temporary uploads |

---

## 3. Environment Variables Reference

### Backend Configuration

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `HOST` | string | `0.0.0.0` | Network interface to bind (binds all interfaces in production containers) |
| `PORT` | integer | `8000` | Port for the ASGI server (auto-detected on Render, Heroku, Cloud Run) |
| `DEBUG` | boolean | `false` | Enables FastAPI debug mode (strictly `false` in production) |
| `LOG_LEVEL` | string | `INFO` | Logging severity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `DATABASE_URL` | string | `sqlite:///./backend/health_screening.db` | SQLAlchemy connection string (e.g. `postgresql://user:pass@host:5432/db`) |
| `CORS_ORIGINS` | string | `http://localhost:5173,...` | Comma-separated allowed frontend origins. In production: `https://screening.yourhospital.org` |
| `CHECKPOINT_PATH` | string | `ml/weights/eye_model.pth` | Path to the trained champion DenseNet201 checkpoint |
| `ML_DEVICE` | string | `auto` | Execution target: `auto` (detects CUDA, falls back to CPU), `cpu`, or `cuda` |
| `MAX_UPLOAD_SIZE_MB` | integer | `30` | Maximum file size for fundus uploads in MB (default: 30 MB) |

### Frontend Configuration

| Variable | Scope | Default | Description |
| :--- | :--- | :--- | :--- |
| `VITE_BACKEND_URL` | Build-time | `""` (empty) | Base URL of backend API. Empty string uses same-origin `/api/v1` (Option A). Set to `https://api.yourdomain.com` for detached hosting (Option B). |

---

## 4. Model Artifact Management

- **Champion Checkpoint File:** `ml/weights/eye_model.pth`
- **File Size:** 73,754,057 bytes (~73.7 MB)
- **Cryptographic Hash (SHA-256):**
  ```text
  e819440c35f166c90e6469b071cc83136857e95aa80bcd6697336894bb459f5f
  ```
- **Integrity Rule:** The backend verifies weights with `torch.load(weights_only=True)`. If the checkpoint is missing or corrupted, the service returns **HTTP 503 Service Unavailable**. The deployment pipeline **never** substitutes randomized weights or mock predictions.
- **In-Memory Caching:** Weights are loaded once at startup during the FastAPI `lifespan` event and held in a thread-safe singleton cache. Inference requests do not reload files from disk.

---

## 5. Health, Liveness, and Readiness Monitoring

The platform provides Kubernetes- and cloud-standard health endpoints:

### Liveness Probe (`GET /health/live` or `GET /api/v1/health/live`)
- **Purpose:** Verifies the Python web process is running and accepting HTTP requests.
- **Success:** `200 OK`
  ```json
  {
    "status": "alive",
    "service": "multimodal-health-screening-backend",
    "timestamp": "2026-10-12T17:15:00.000Z"
  }
  ```

### Readiness Probe (`GET /health/ready` or `GET /api/v1/health/ready`)
- **Purpose:** Verifies that the model weights are loaded, pre-warmed, and database connection is active. Load balancers route traffic only when this returns 200.
- **Success:** `200 OK`
  ```json
  {
    "status": "ready",
    "checkpoint_available": true,
    "model_loaded": true,
    "database_connected": true,
    "device": "cuda",
    "timestamp": "2026-10-12T17:15:00.000Z"
  }
  ```
- **Failure:** `503 Service Unavailable` with `status: "unready"` and specific diagnostic reason.

### Detailed Health Probe (`GET /health` or `GET /api/v1/health`)
- Returns class order, device name, and model readiness for the frontend header indicator.

---

## 6. Build and Deployment Instructions

### Method 1: Docker Container (Option A — Recommended)

Build and run the self-contained production container:

```bash
# 1. Build the production image (includes frontend build & CPU PyTorch)
docker build -t health-screening:latest .

# 2. Run the container binding to port 8000
docker run -d \
  --name health-screening \
  -p 8000:8000 \
  -e PORT=8000 \
  -e HOST=0.0.0.0 \
  -e ML_DEVICE=cpu \
  health-screening:latest

# 3. Verify container health
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

### Method 2: Docker Compose

For local staging with persistent uploads volume:

```bash
docker-compose up -d --build
```

### Method 3: Bare-Metal / Virtual Machine (Systemd)

#### Step 1: Build Frontend
```bash
cd frontend
npm ci
npm run build
cd ..
```

#### Step 2: Set up Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

#### Step 3: Run Service
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```
*(Note: Maintain `--workers 1` or use Gunicorn with multiple processes only if host has sufficient RAM for multiple model replicas (~1.5GB per worker).)*

### Method 4: Decoupled Cloud Deployment (Option B)

#### Backend on Render / Railway / Fly.io / GCP Cloud Run:
1. Connect repository root.
2. Build Command: `pip install -r requirements.txt && pip install -e .`
3. Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Set Environment Variables:
   - `CORS_ORIGINS`: `https://your-frontend.vercel.app`
   - `ML_DEVICE`: `cpu`
   - `DATABASE_URL`: Managed PostgreSQL connection string.

#### Frontend on Vercel / Netlify:
1. Root Directory: `frontend`
2. Build Command: `npm run build`
3. Output Directory: `dist`
4. Environment Variable:
   - `VITE_BACKEND_URL`: `https://your-backend.onrender.com`

---

## 7. Pre-Deployment Verification Checklist

Before promoting a build to production, run the following verification sequence:

- [ ] **Dependency check:** `pip check` (verifies zero dependency conflicts)
- [ ] **Backend test suite:** `python -m pytest tests/ -v` (39+ tests passing)
- [ ] **Liveness / Readiness probe check:** `curl http://localhost:8000/health/ready` returns 200
- [ ] **ML environment diagnostics:** `python -m ml.diagnostics` returns HEALTHY
- [ ] **Frontend production bundle:** `npm run build` compiles without syntax or asset errors
- [ ] **Checkpoint SHA-256 verification:** matches `e819440c35f1...`
- [ ] **CORS configuration:** includes exact frontend domain(s)
- [ ] **Disk cleanup:** temporary files in `backend/uploads/` are unlinked upon quality rejection

---

## 8. Clinical Prototype Disclaimer

> **IMPORTANT:** This software is an automated screening decision-support prototype. It is not an autonomous diagnostic medical device and has not received FDA 510(k) clearance or CE-mark certification. All model predictions and vital-sign interpretations must be reviewed by qualified ophthalmologists and clinicians.
