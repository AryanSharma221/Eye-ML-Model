# FULL-STACK ARCHITECTURE & TECH STACK VIVA GUIDE
## Multimodal Health Screening Platform: Frontend, Backend, API Contracts, and System Engineering

---

```
Document Reference : DOCS/PANEL-PREPARATION/03-FULL-STACK-ARCHITECTURE-AND-TECH-STACK.MD
Project Title      : Multimodal Health Screening: AI-Assisted Retinal Disease Detection & Explainable Decision Support
Target Audience    : Undergraduate / Capstone Engineering Students Defending System Architecture & Software Engineering
Academic Context   : Computer Science & Engineering Capstone Evaluation Panel / Viva Voce Examination
Active Frontend    : React 19.2.8 / Vite 8.3.4 / Tailwind CSS 4.3.3 / Chart.js 4.5.1 / Axios 1.20.0
Active Backend     : FastAPI 0.143.0 / Python 3.12.4 / Uvicorn 0.54.0 / Starlette 1.7.0 / Pydantic 2.14.0
Active Database    : SQLite 3.x / SQLAlchemy 2.1.4 ORM / ACID Compliant Relational Storage
Active ML Engine   : PyTorch 2.6.0+cu124 / Torchvision 0.21.0 / DenseNet-201 (70.34 MB Checkpoint)
Security Hardening : CSP Headers, 30 MB Request Limiter, Sliding-Window Rate Limiter, Correlation IDs, CORS
Verified Status    : 638 Automated Tests Passing (575 ML + 63 Backend) + Full Puppeteer E2E Verification
```

---

# TABLE OF CONTENTS

1. [Chapter 1: Technology Stack Inventory](#chapter-1-technology-stack-inventory)
2. [Chapter 2: Frontend Architecture Deep-Dive](#chapter-2-frontend-architecture-deep-dive)
3. [Chapter 3: Backend Architecture Deep-Dive](#chapter-3-backend-architecture-deep-dive)
4. [Chapter 4: API Communication Specification & Contracts](#chapter-4-api-communication-specification--contracts)
5. [Chapter 5: Complete End-to-End Request Lifecycle](#chapter-5-complete-end-to-end-request-lifecycle)
6. [Chapter 6: Grad-CAM Delivery Lifecycle & Forensic Defect Analysis (Part 14)](#chapter-6-grad-cam-delivery-lifecycle--forensic-defect-analysis-part-14)
7. [Chapter 7: Database Architecture & Data Integrity](#chapter-7-database-architecture--data-integrity)
8. [Chapter 8: Deployment, Environment & Infrastructure](#chapter-8-deployment-environment--infrastructure)
9. [Chapter 9: Engineering Trade-Offs & Architecture Defense](#chapter-9-engineering-trade-offs--architecture-defense)
10. [Chapter 10: Full-Stack Software Engineering Viva Suite (75 Questions)](#chapter-10-full-stack-software-engineering-viva-suite-75-questions)
11. [Chapter 11: 1-Page Full-Stack Cheat Sheet / Quick Defense Card](#chapter-11-1-page-full-stack-cheat-sheet--quick-defense-card)

---

# CHAPTER 1: TECHNOLOGY STACK INVENTORY

This inventory documents every production technology, library, framework, and utility verified directly from `package.json`, `requirements.txt`, `pyproject.toml`, and runtime configuration.

| # | Technology / Library | Actual Version | Architectural Purpose | Where Used in Codebase | Why Suitable for this Project | Alternative Considered | Engineering Trade-off | Likely Panel Viva Question |
|---|----------------------|----------------|-----------------------|------------------------|-------------------------------|------------------------|-----------------------|----------------------------|
| **1** | **React** | `19.2.8` | Client-Side UI View Engine | `frontend/src/App.jsx`, `frontend/src/components/*` | Declarative component hierarchy, concurrent rendering, virtual DOM reconciliation, robust state hooks. | Vue.js 3, Angular 18, Svelte 5 | Steeper bundle overhead than Svelte, but superior ecosystem maturity, community support, and medical widget libraries. | *"Why React 19 over vanilla JavaScript or Angular?"* |
| **2** | **Vite** | `8.3.4` | Modern Frontend Build Tool & Dev Server | `frontend/vite.config.js`, `frontend/package.json` | Native ES modules during development give instant Hot Module Replacement (HMR; <50ms); Rollup-based production minification. | Webpack 5, Create-React-App (CRA), Turbopack | CRA is deprecated and slow. Vite requires explicit Rollup build config for legacy polyfills, but cuts build times by 10x. | *"What makes Vite faster than Webpack during development?"* |
| **3** | **Tailwind CSS** | `4.3.3` | Utility-First Responsive Styling Engine | `frontend/src/index.css`, `frontend/vite.config.js` | Zero runtime style overhead, rapid prototyping of dark-mode clinical UI, purge of unused CSS rules down to <25 KB. | Bootstrap 5, Material-UI (MUI), Chakra UI | Class names in JSX can look cluttered, but eliminates bloated runtime CSS-in-JS stylesheets and context overhead. | *"Why utility-first CSS rather than a pre-styled library like Material UI?"* |
| **4** | **Axios** | `1.20.0` | Promise-based HTTP Client | `frontend/src/api.js` | Built-in request/response interceptors, automatic JSON transformation, cross-browser upload progress tracking, timeout handling. | Native browser `fetch()` API | Slightly larger bundle size than native `fetch()`, but provides cleaner error rejection and standardized FormData handling. | *"Why add Axios when the browser provides native fetch()?"* |
| **5** | **Chart.js** & **react-chartjs-2** | `4.5.1` / `5.3.0` | Canvas-based Biometric Data Visualization | `frontend/src/components/VitalsAssessment.jsx` | High-performance HTML5 Canvas rendering for vital sign sparklines and triage gauges; responsive redraws. | D3.js, Recharts, Plotly.js | D3.js offers lower-level control but requires 5x more boilerplate code; Chart.js is ideal for standardized gauges. | *"Why render charts on HTML5 Canvas instead of SVG?"* |
| **6** | **FastAPI** | `0.143.0` | High-Performance Asynchronous Web Framework | `backend/main.py`, `backend/routes/*` | Native Python asynchronous (`async`/`await`) support, automatic OpenAPI/Swagger docs, high throughput on ASGI. | Flask, Django, Tornado | Django is too heavyweight with unnecessary ORM overhead; Flask lacks native type-hint validation and native async performance. | *"What architectural advantage does FastAPI give over Flask?"* |
| **7** | **Uvicorn** | `0.54.0` | Lightning-Fast ASGI Web Server | `backend/main.py`, `backend/run_server.py` | Implementation of ASGI specification on top of `uvloop` (C-based event loop) and `httptools`; handles async I/O. | Gunicorn, Waitress, Hypercorn | Uvicorn alone runs single-process dev; in production it is paired with process supervision or container orchestrators. | *"What is ASGI and how does Uvicorn implement it?"* |
| **8** | **Starlette** | `1.7.0` | Core ASGI Toolkit underlying FastAPI | `backend/security.py`, `backend/main.py` | Provides foundational middleware architecture, HTTP request/response abstractions, static file mounts, and streaming responses. | Werkzeug, Twisted | Starlette is FastAPI's engine; gives lightweight middleware primitives without framework lock-in. | *"What role does Starlette play inside a FastAPI application?"* |
| **9** | **Pydantic** | `2.14.0` | Data Validation & Settings Management | `backend/schemas.py`, `backend/config.py` | Core rewritten in Rust (v2) for 10x faster validation; strict Python type-hint enforcement and JSON Schema generation. | Marshmallow, Cerberus, dataclasses | Requires strict schema definition up front, but guarantees that malformed payloads fail before touching application logic. | *"How does Pydantic enforce data integrity at the API boundary?"* |
| **10** | **SQLAlchemy** | `2.1.4` | Python SQL Toolkit & Object Relational Mapper | `backend/database.py`, `backend/models_db.py` | Modern 2.0-style ORM syntax (`Mapped`, `mapped_column`), safe parameterized queries preventing SQL injection, connection pooling. | Django ORM, Peewee, Tortoise ORM | Slight abstraction overhead vs raw SQL, but eliminates SQL syntax errors and supports switching from SQLite to PostgreSQL. | *"What is an ORM and why use SQLAlchemy instead of raw SQL queries?"* |
| **11** | **SQLite** | `3.x` | Serverless Embedded Relational Database | `backend/health_screening.db` | Zero-configuration, zero network latency, ACID-compliant file-based database; perfect for offline edge deployment in clinics. | PostgreSQL, MySQL, MongoDB | Lacks high concurrent-write scaling, but matches single-clinic workflow where writes take <2ms and read concurrency is high. | *"Why choose SQLite over PostgreSQL for a clinical screening platform?"* |
| **12** | **ReportLab** | `5.0.1` | Programmatic PDF Generation Engine | `backend/pdf_service.py`, `backend/routes/reports.py` | Pixel-perfect clinical diagnostic reports generated directly on the server; supports tables, colors, embedded images, and page numbers. | WeasyPrint, jsPDF, Puppeteer PDF | WeasyPrint has heavy system dependencies (Pango/Cairo); ReportLab is pure Python with fast, reliable, reproducible vector rendering. | *"Why generate diagnostic PDFs on the backend rather than using window.print() in React?"* |
| **13** | **PyTorch** | `2.6.0+cu124` | Deep Learning Tensor & Neural Network Engine | `ml/model.py`, `ml/inference.py`, `ml/gradcam.py` | Dynamic computation graph, native CUDA acceleration, comprehensive `torchvision` models, backward hook support for Grad-CAM. | TensorFlow 2, ONNX Runtime, OpenVINO | Larger runtime memory footprint than ONNX, but provides native hooks for Grad-CAM backpropagation and direct model training. | *"Why PyTorch instead of TensorFlow or ONNX Runtime for inference?"* |
| **14** | **Torchvision** | `0.21.0` | Computer Vision Datasets & Transforms | `ml/transforms.py`, `ml/inference.py` | Standardized bilinear image interpolation, ImageNet tensor normalization, tensor conversions matching training pipeline. | OpenCV only, Albumentations | Keeps preprocessing pipeline 100% mathematically identical between PyTorch training scripts and live inference. | *"Why must inference normalization match training normalization exactly?"* |
| **15** | **Pillow (PIL)** & **OpenCV** | `11.1.0` / `4.11.0` | Image Decoding, Quality Metrics, Heatmap Blend | `ml/image_quality.py`, `ml/gradcam.py` | OpenCV provides fast C++ Laplacian variance and color conversions; Pillow handles cross-format PNG/JPEG decoding and Base64 streams. | scikit-image, imageio | Dual dependency balances OpenCV's speed for array math with Pillow's safe memory handling for network file bytes. | *"How does OpenCV compute image blur using Laplacian variance?"* |
| **16** | **Oxlint** | `1.81.0` | Rust-based High-Speed Linter | `frontend/package.json` | 50x faster than ESLint; detects dead code, unused imports, and syntax bugs instantly in CI/CD pipeline. | ESLint, Biome | Oxlint focuses on correctness rather than stylistic formatting, running across 100 files in under 15 milliseconds. | *"Why use a Rust-based linter like Oxlint in a JavaScript project?"* |
| **17** | **Puppeteer Core** | `25.13.0` | Headless Chrome Browser Automation | `scripts/smoke_frontend_puppeteer.js` | Verifies real browser rendering, client-side routing, canvas draw calls, button click event handlers, and Base64 image display. | Cypress, Playwright, Selenium | Direct Chromium DevTools Protocol (CDP) control without heavyweight test runner overhead; runs lightweight smoke tests in <3 seconds. | *"How did Puppeteer verify that the Grad-CAM image bug was fixed?"* |

---

# CHAPTER 2: FRONTEND ARCHITECTURE DEEP-DIVE

```
+----------------------------------------------------------------------------------------------------+
|                                    REACT 19 COMPONENT TREE                                         |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                         [App.jsx]                                                  |
|                                             |                                                      |
|                   +-------------------------+-------------------------+                            |
|                   |                         |                         |                            |
|             [Navbar.jsx]             [Routes / Views]           [Footer.jsx]                       |
|                                             |                                                      |
|           +---------------------------------+---------------------------------+                    |
|           |                                 |                                 |                    |
|  [ScreeningDashboard.jsx]           [PatientsPage.jsx]             [Static Legal Pages]            |
|       (Primary Workflow)            (Clinical Records)             (/terms, /privacy, etc.)        |
|           |                                 |                                                      |
|    +------+------+                          +---> [PatientDetailModal]                             |
|    |             |                                                                                 |
| [EyeAssessment]  [VitalsAssessment]                                                                |
|    |             |                                                                                 |
|    |             +---> [Chart.js Sparklines]                                                       |
|    |                                                                                               |
|    +---> [ImageDropzone]                                                                           |
|    +---> [ReticleHUD (Canvas/SVG)]                                                                 |
|    +---> [GradCAMToggle & Slider]                                                                  |
|    +---> [FallbackErrorAlert]                                                                      |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 2.1 Component-Based Hierarchy
The user interface is built as a Single Page Application (SPA) using React 19. The application follows strict modular encapsulation:
1. **Application Shell (`App.jsx`)**:
   - Renders the global layout: `Navbar` (branding, system status badge, session indicators), main route container, and `Footer` (regulatory disclaimer).
   - Manages global toast notifications and modal overlay states.
2. **Screening Dashboard (`ScreeningDashboard.jsx`)**:
   - The central operational workspace for clinicians.
   - Coordinates state between two independent clinical panels:
     - **Eye Disease Assessment (`EyeAssessment.jsx`)**: Manages image upload, preset selection, quality gate status, classification output, probability distribution, and Grad-CAM heatmap visualization.
     - **Physiological Assessment (`VitalsAssessment.jsx`)**: Manages heart rate, $\text{SpO}_2$, and temperature inputs, computes deterministic triage statuses, and renders biometric risk gauges.
   - Houses the unified "Run Multimodal Screening" dispatch button and session recording controls.
3. **Patients Directory (`PatientsPage.jsx`)**:
   - Displays historical patient records, past screening sessions, diagnostic reports, and timestamps.
   - Includes real-time search, sorting, and modal drill-down into past session details.

### 2.2 Client-Side Routing (React Router DOM v7)
Navigation is handled entirely client-side without full-page browser reloads:
- `/`: Primary screening workspace (`ScreeningDashboard`).
- `/patients`: Historical patient management and session records (`PatientsPage`).
- `/domain-guide`: Clinical knowledgebase and user manual.
- `/terms`: Regulatory, legal, and clinical decision support disclaimers.
- `/privacy`: Data privacy, retention, and patient consent documentation.

### 2.3 State Management Philosophy
Rather than introducing the boilerplate and memory overhead of Redux or Zustand, state is managed using React's native hooks:
- **`useState`**: Encapsulates local component states (e.g., active tab, slider opacity, current preset, loading spinners).
- **`useEffect`**: Coordinates side effects such as pre-loading preset images, fetching backend health on mount, and synchronizing slider transitions.
- **`useRef`**: Holds direct DOM references to the HTML5 Canvas reticle and file input elements without triggering re-renders.
- **`useCallback` & `useMemo`**: Memoizes calculation of risk levels and prevents unnecessary redraws of high-resolution images.

### 2.4 Form Validation & File Input Sanitization
Before any byte leaves the browser, client-side validation executes:
- **Image Type Filter**: Accepts only `image/jpeg`, `image/png`, and `image/webp`. Rejects executable files, PDFs, or SVGs.
- **File Size Ceiling**: Rejects files larger than 15 MB before upload begins, presenting an inline error alert.
- **Biometric Range Guardrails**:
  - Heart Rate: Guarded to $30 \le \text{BPM} \le 250$.
  - $\text{SpO}_2$: Guarded to $50\% \le \text{SpO}_2 \le 100\%$.
  - Temperature: Guarded to $30.0^\circ\text{C} \le \text{Temp} \le 45.0^\circ\text{C}$.
  - Values outside biological feasibility trigger red outline borders and disable the submission trigger.

### 2.5 API Communication Layer (`api.js`)
All HTTP requests flow through a centralized Axios client configured in `frontend/src/api.js`:
- **Dynamic Base URL Resolution**:
  ```javascript
  const getApiBaseUrl = () => {
    if (import.meta.env.VITE_API_BASE_URL) {
      return import.meta.env.VITE_API_BASE_URL;
    }
    // In production build, relative URL routes through the same origin
    if (import.meta.env.PROD) {
      return '/api/v1';
    }
    // In local development, route to backend port 8000
    return 'http://localhost:8000/api/v1';
  };
  ```
- **Interceptors**: Catch network failures, log correlation IDs, and standardize error messages so UI components receive clean error strings.

### 2.6 The Reticle HUD & Inspection Canvas
To provide a specialized clinical experience, the retinal image viewer includes:
- **Interactive Inspection Grid**: Superimposes crosshairs and concentric calibration rings over the fundus image.
- **Coordinate Tracking**: Displays normalized $(X, Y)$ coordinates of the clinician's cursor to allow precise reference to specific lesions.
- **Grad-CAM Blend Slider**: Enables continuous $\alpha$-blending ($0.0 \to 1.0$) between the raw fundus photograph and the attention heatmap.

---

# CHAPTER 3: BACKEND ARCHITECTURE DEEP-DIVE

```
+----------------------------------------------------------------------------------------------------+
|                                    FASTAPI BACKEND ARCHITECTURE                                    |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|                                       [Uvicorn ASGI Server]                                        |
|                                                 |                                                  |
|                        +------------------------v------------------------+                         |
|                        |                  MIDDLEWARE PIPELINE            |                         |
|                        |  1. CorrelationIdMiddleware (X-Request-ID)      |                         |
|                        |  2. InMemoryRateLimiter (60 req/min sliding)    |                         |
|                        |  3. RequestSizeLimitMiddleware (30 MB limit)    |                         |
|                        |  4. SecurityHeadersMiddleware (CSP, nosniff)    |                         |
|                        |  5. CORSMiddleware (Allowed Origins)            |                         |
|                        +------------------------+------------------------+                         |
|                                                 |                                                  |
|                                       [FastAPI Router Engine]                                      |
|                                                 |                                                  |
|         +-------------------+-------------------+-------------------+--------------------+         |
|         |                   |                   |                   |                    |         |
|   [routes/eye.py]    [routes/vitals.py]  [routes/screen.py]  [routes/patients.py]  [routes/reports.py]
|         |                   |                   |                   |                    |         |
|   +-----v-----+             |             +-----v-----+             |              +-----v-----+   |
|   | MLService |             |             | MLService |             |              |PDFService |   |
|   +-----------+             |             +-----------+             |              +-----------+   |
|         |                   |                   |                   |                    |         |
|   +-----v-----+             +-------------------+-------------------+                    |         |
|   | PyTorch   |                                 |                                        |         |
|   | DenseNet  |                       [SQLAlchemy 2.1 ORM]                               |         |
|   | & GradCAM |                                 |                                        |         |
|   +-----------+                    [SQLite: health_screening.db]                         |         |
|                                                 |                                        |         |
|                                 +---------------+---------------+                        |         |
|                                 |               |               |                        |         |
|                             patients        sessions        eye_data / physiological_data|         |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 3.1 Asynchronous Event Loop & Concurrency
FastAPI runs on the ASGI standard powered by Uvicorn. The backend leverages:
- **Non-Blocking I/O for Database & Network**: Asynchronous route handlers (`async def`) yield execution to the event loop while waiting for database queries, file reads, or client network packets.
- **CPU-Bound ML Thread Isolation**: Heavy PyTorch matrix operations and OpenCV calculations are processed synchronously inside service routines. By structuring the model forward pass with `torch.no_grad()`, Python avoids computing gradient graphs, releasing the GIL during tensor operations.

### 3.2 Application Lifespan & GPU Pre-Warming
Model loading is an expensive operation (loading 70 MB of weights from disk to RAM/VRAM takes 1–3 seconds). FastAPI's modern `@asynccontextmanager` lifespan manages this:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP SEQUENCE:
    logger.info("Initializing system: Pre-warming DenseNet-201 model...")
    ml_service.prewarm() # Loads PyTorch checkpoint, sends dummy tensor through GPU
    init_db()            # Verifies SQLite tables and indices exist
    yield
    # SHUTDOWN SEQUENCE:
    logger.info("Flushing database connections and releasing memory...")
```
**Why Pre-Warming Matters**: If the model was loaded lazily on the first user request, that patient would experience a 3-second lag ("cold start"). Pre-warming guarantees that every user request receives consistent sub-50ms inference.

### 3.3 Security & Middleware Pipeline
Every HTTP request passes through five security gates before reaching a route handler:
1. **Correlation ID Middleware (`CorrelationIdMiddleware`)**:
   - Inspects the incoming headers for `X-Request-ID`. If absent, generates a cryptographically random UUID v4.
   - Injects this ID into log records and appends it to the response header, enabling end-to-end distributed tracing.
2. **Sliding-Window Rate Limiter (`InMemoryRateLimiter`)**:
   - Tracks request timestamps per client IP in a deque with a 60-second window.
   - Enforces a ceiling of 60 requests per minute. Excess requests immediately receive `429 Too Many Requests` with a `Retry-After` header.
3. **Request Size Limiter (`RequestSizeLimitMiddleware`)**:
   - Inspects the `Content-Length` header. Rejects any payload exceeding 30 MB (`31,457,280 bytes`) with `413 Payload Too Large`.
   - Protects the server from Denial of Service (DoS) memory exhaustion attacks via oversized file uploads.
4. **Security Headers Middleware (`SecurityHeadersMiddleware`)**:
   - Injects defensive HTTP response headers:
     - `Content-Security-Policy`: Restricts script, style, and font execution.
     - `X-Frame-Options: DENY`: Prevents clickjacking attacks inside `<iframe>` tags.
     - `X-Content-Type-Options: nosniff`: Prevents MIME-type confusion attacks.
     - `Referrer-Policy: strict-origin-when-cross-origin`: Protects clinical URL privacy.
5. **CORS Middleware (`CORSMiddleware`)**:
   - Regulates Cross-Origin Resource Sharing. Configured via `CORS_ORIGINS` in `config.py` to allow the frontend dev server (`http://localhost:5173`) while denying unauthorized external origins.

### 3.4 Pydantic Schema Validation & Serialization (`schemas.py`)
Incoming JSON payloads and outgoing responses are strictly validated through Pydantic v2 schemas:
- **`VitalsRequest`**: Enforces strict biological boundaries on `heart_rate_bpm`, `spo2_percent`, and `temperature_celsius`. Any non-numeric or out-of-bounds input triggers an immediate `422 Unprocessable Entity` with a descriptive field error.
- **`EyePredictionResponse`**: Guarantees that the classification response adheres to an exact contract containing `prediction`, `confidence`, `risk_level`, `probabilities`, `quality_checks`, `gradcam_image`, and `gradcam_url`.

### 3.5 Layered Separation of Concerns
The backend codebase adheres strictly to clean architecture:
- **Routers (`backend/routes/`)**: Deal only with HTTP protocol details, request unpacking, and response status codes.
- **Services (`backend/ml_service.py`, `backend/pdf_service.py`)**: House pure business logic, model inference, and report compilation.
- **Data Access (`backend/database.py`, `backend/models_db.py`)**: Abstract all database interactions, schema definitions, and queries.

---

# CHAPTER 4: API COMMUNICATION SPECIFICATION & CONTRACTS

```
+----------------------------------------------------------------------------------------------------+
|                                    COMPLETE API ENDPOINT CATALOGUE                                 |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
| METHOD | ROUTE                                | CONTENT TYPE          | PURPOSE                    |
|--------+--------------------------------------+-----------------------+----------------------------|
| POST   | /api/v1/predict_eye                  | multipart/form-data   | Fundus inference & Grad-CAM|
| POST   | /api/v1/submit_vitals                | application/json      | Biometric triage & status  |
| POST   | /api/v1/screen                       | multipart/form-data   | Unified multimodal screen  |
| POST   | /api/v1/patients                     | application/json      | Register new patient       |
| GET    | /api/v1/patients                     | None                  | List all patients          |
| GET    | /api/v1/patients/{id}                | None                  | Get patient by ID          |
| PUT    | /api/v1/patients/{id}                | application/json      | Update patient record      |
| DELETE | /api/v1/patients/{id}                | None                  | Delete patient & sessions  |
| POST   | /api/v1/sessions                     | application/json      | Create screening session   |
| GET    | /api/v1/sessions/{id}                | None                  | Get full session details   |
| GET    | /api/v1/patients/{id}/sessions       | None                  | Get sessions for patient   |
| DELETE | /api/v1/sessions/{id}                | None                  | Delete screening session   |
| GET    | /api/v1/sessions/{id}/report         | None (query: ?format) | Download PDF / view HTML   |
| GET    | /health, /ready, /live               | None                  | System operational probes  |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 4.1 Data Transfer Protocols: Multipart vs JSON
- **`multipart/form-data`**: Used exclusively when binary image data must be transmitted (`predict_eye`, `screen`). Allows efficient boundary-separated streaming of binary image files alongside optional metadata fields (`session_id`, `include_gradcam`) without requiring 33% Base64 transmission overhead.
- **`application/json`**: Used for structured biometric data (`submit_vitals`) and patient records (`patients`, `sessions`). Enables strict schema enforcement, small payload sizes, and native browser parsing.

---

### 4.2 Endpoint 1: Retinal Inference & Explainability
- **Route**: `POST /api/v1/predict_eye`
- **Content-Type**: `multipart/form-data`
- **Request Parameters**:
  - `image`: Binary file (JPEG/PNG, required).
  - `session_id`: Integer (optional foreign key).
  - `include_gradcam`: Boolean (`true` by default).

#### Response Payload Contract (`application/json`):
```json
{
  "prediction": "glaucoma",
  "class_index": 2,
  "confidence": 0.929,
  "risk_level": "HIGH",
  "quality_checks": {
    "passed": true,
    "blur_score": 142.8,
    "is_blurry": false,
    "brightness_score": 118.4,
    "contrast_score": 52.1,
    "resolution": [512, 512],
    "is_circular_retina": true,
    "summary": "Image quality passed all clinical gates."
  },
  "probabilities": {
    "cataract": 0.012,
    "diabetic_retinopathy": 0.041,
    "glaucoma": 0.929,
    "normal": 0.018
  },
  "gradcam_image": "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAYAAAD0eNT6...",
  "gradcam_url": "/uploads/gradcam/gradcam_20261012_183501_a9f2.png",
  "gradcam_layer": "dense_block4",
  "image_path": "/uploads/eye_images/fundus_20261012_183500_b81c.jpg",
  "session_id": 14,
  "created_at": "2026-10-12T18:35:01.412Z"
}
```

---

### 4.3 Endpoint 2: Physiological Biometric Triage
- **Route**: `POST /api/v1/submit_vitals`
- **Content-Type**: `application/json`

#### Request Payload Contract:
```json
{
  "session_id": 14,
  "heart_rate_bpm": 74.0,
  "spo2_percent": 98.0,
  "temperature_celsius": 36.8
}
```

#### Response Payload Contract (`application/json`):
```json
{
  "id": 28,
  "session_id": 14,
  "heart_rate_bpm": 74.0,
  "spo2_percent": 98.0,
  "temperature_celsius": 36.8,
  "hr_status": "NORMAL",
  "spo2_status": "NORMAL",
  "temp_status": "NORMAL",
  "overall_status": "NORMAL",
  "risk_level": "LOW",
  "summary": "All vital parameters within normal clinical physiological range.",
  "timestamp": "2026-10-12T18:35:02.105Z"
}
```

---

### 4.4 Endpoint 3: Unified Multimodal Screening
- **Route**: `POST /api/v1/screen`
- **Content-Type**: `multipart/form-data`
- **Request Form Fields**:
  - `image`: Binary file (JPEG/PNG).
  - `heart_rate_bpm`: Float (e.g., `74.0`).
  - `spo2_percent`: Float (e.g., `98.0`).
  - `temperature_celsius`: Float (e.g., `36.8`).
  - `session_id`: Integer (optional).
  - `patient_id`: Integer (optional).

#### Response Payload Contract:
Contains both `eye_prediction` and `physiological_assessment` objects, alongside an aggregated `overall_multimodal_risk` field computed without cross-polluting domain data.

---

### 4.5 Endpoint 4: Diagnostic PDF Report Generation
- **Route**: `GET /api/v1/sessions/{id}/report?format=pdf`
- **Response Headers**:
  - `Content-Type: application/pdf`
  - `Content-Disposition: attachment; filename="Clinical_Screening_Report_Session_14.pdf"`
- **Response Body**: Binary octet stream containing the fully compiled, vector-rendered ReportLab diagnostic document.

---

# CHAPTER 5: COMPLETE END-TO-END REQUEST LIFECYCLE

```
[Browser: User clicks "Run AI Screening"]
     |
     v
[1. React File Validation & State Transition] (loading=true, lock submit)
     |
     v
[2. Axios FormData Assembly] (image binary + session metadata)
     |
     v
[3. HTTP/1.1 TCP Packet dispatched across network to :8000]
     |
     v
[4. Uvicorn ASGI Socket Receive & Request Object Creation]
     |
     v
[5. Security Middleware Layer]
     |---> CorrelationIdMiddleware: Assigns X-Request-ID
     |---> RateLimiterMiddleware: Checks client IP quota (<60 req/min)
     |---> RequestSizeLimitMiddleware: Inspects Content-Length (<30 MB)
     |---> SecurityHeadersMiddleware: Sets CSP, nosniff, frame-options
     |---> CORSMiddleware: Validates Origin header
     |
     v
[6. FastAPI Dependency Injection & Route Matching] (routes/eye.py)
     |
     v
[7. File Ingestion & Magic Byte Validation] (reads stream, checks JPEG/PNG headers)
     |
     v
[8. Quality Gating Pipeline (ml/image_quality.py)]
     |---> Computes Laplacian Variance (blur check)
     |---> Computes Mean Luminance & Contrast
     |---> Verifies Circular Retinal Mask Geometry
     |
     v
[9. PyTorch Preprocessing & Tensor Normalization] (Resize 224x224, ImageNet normalize)
     |
     v
[10. DenseNet-201 Forward Inference] (torch.no_grad, outputs 4 class logits)
     |
     v
[11. Softmax Calculation & Argmax Class Selection] (Disease label + Confidence)
     |
     v
[12. Grad-CAM Generation Pipeline (ml/gradcam.py)]
     |---> Re-runs forward pass with gradient tracking
     |---> Backpropagates target class gradient to dense_block4
     |---> Global Average Pools gradients -> Computes weighted feature map
     |---> Applies ReLU -> Bilinear upsample to 512x512
     |---> Applies OpenCV JET colormap -> Blends with original photograph (alpha=0.4)
     |---> Encodes blended array to Base64 PNG string
     |
     v
[13. Disk Persistence Layer]
     |---> Saves raw image to backend/uploads/eye_images/
     |---> Saves Grad-CAM heatmap to backend/uploads/gradcam/
     |
     v
[14. Database Persistence Layer]
     |---> SQLAlchemy creates EyeData ORM object
     |---> Commits transaction to backend/health_screening.db
     |
     v
[15. Pydantic Response Serialization] (Serializes EyePredictionResponse to JSON)
     |
     v
[16. HTTP 200 OK Response travels over wire to browser]
     |
     v
[17. Axios Response Interceptor receives JSON payload]
     |
     v
[18. React State Update] (setEyeResult(data), setLoading(false))
     |
     v
[19. DOM Reconciliation & Viewport Redraw]
     |---> Renders classified disease badge (GLAUCOMA, 92.9% High Risk)
     |---> Animates 4-class probability distribution bars
     |---> Multi-source resolver mounts Grad-CAM image onto Reticle HUD
```

### Comprehensive Failure Mode Matrix
What happens when things go wrong? Every stage has verified, deterministic error boundaries:

| Failure Phase | Trigger Cause | Status Code / System Behavior | Client-Side User Experience |
|---------------|---------------|-------------------------------|-----------------------------|
| **Client Input** | User selects 50 MB video or `.exe` file | Request aborted before dispatch | Red banner: *"Invalid file type. Please upload a JPEG or PNG fundus scan under 15 MB."* |
| **Network** | Network drops or backend offline | Axios catch block (`ERR_CONNECTION_REFUSED`) | Toast alert: *"Backend server unavailable. Please check system status."* |
| **Middleware** | Client exceeds 60 requests in 1 minute | `429 Too Many Requests` | Alert banner: *"Rate limit exceeded. Please wait 60 seconds before retrying."* |
| **Middleware** | Upload payload exceeds 30 MB | `413 Payload Too Large` | Alert banner: *"Request payload too large. Please upload an image under 30 MB."* |
| **Image Validation** | User renames `.txt` file to `.jpg` | `400 Bad Request` | Alert banner: *"Uploaded file is corrupted or not a valid image format."* |
| **Quality Gating** | User uploads severely blurry photo | `422 Unprocessable Entity` | Warning modal: *"Image quality check failed: High blur detected. Please re-take scan."* |
| **ML Inference** | GPU Out-Of-Memory or PyTorch runtime fault | `500 Internal Server Error` | Diagnostic error: *"ML inference engine encountered an error. Error reference logged."* |
| **Image Rendering**| Heatmap string corrupted or undefined | Handled by Part 14 Resolver | Fallback UI: Alert icon with *"Heatmap preview unavailable"* while prediction remains fully legible. |

---

# CHAPTER 6: GRAD-CAM DELIVERY LIFECYCLE & FORENSIC DEFECT ANALYSIS (PART 14)

### 6.1 Generation & Encoding Pipeline
The Grad-CAM heatmap is computed through the following mathematical and algorithmic sequence:
1. **Target Feature Capture**: A PyTorch forward hook captures feature map activations $A^k$ at layer `features.denseblock4.denselayer32.conv2` ($7 \times 7 \times 32$ tensor).
2. **Gradient Backpropagation**: The predicted class score $y^c$ is backpropagated; a backward hook captures gradients $\frac{\partial y^c}{\partial A^k}$.
3. **Weight Computation**: Global Average Pooling computes importance weights:
   $$\alpha_k^c = \frac{1}{7 \times 7} \sum_{i=1}^7 \sum_{j=1}^7 \frac{\partial y^c}{\partial A_{i,j}^k}$$
4. **Weighted Combination & Rectification**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
5. **Colormap Blending**: The $7 \times 7$ heatmap is bilinearly upsampled to $512 \times 512$, normalized to $[0, 255]$, converted via OpenCV's `COLORMAP_JET`, and blended with the original RGB image:
   $$I_{\text{blended}} = 0.4 \cdot I_{\text{heatmap}} + 0.6 \cdot I_{\text{original}}$$
6. **Dual Serialization**:
   - Encoded to Base64 PNG: Returned directly in JSON response as `gradcam_image`.
   - Written to disk: Saved as `backend/uploads/gradcam/gradcam_<timestamp>_<uuid>.png` and referenced by `gradcam_url`.

---

### 6.2 Forensic Analysis of the Part 14 Broken-Image Defect
During Part 14, clinicians reported that when selecting the demo preset for **Glaucoma**, the prediction (92.9%) rendered correctly, but clicking the **Grad-CAM Heatmap** button displayed a broken-image icon and an empty dark container.

#### The Root Cause Investigation
1. **Frontend Preset Definition (`frontend/src/api.js`)**:
   In `DEMO_SAMPLES`, the preset object had the following structure:
   ```javascript
   glaucoma: {
     label: "Glaucoma",
     confidence: 0.929,
     image: "/samples/glaucoma.jpg",
     // BUG: The property 'gradcamHeatmap' was completely absent from the object!
   }
   ```
2. **State Initialization in `EyeAssessment.jsx`**:
   When the user clicked the Glaucoma preset, the component initialized the assessment state:
   ```javascript
   setEyeResult({
     prediction: "glaucoma",
     confidence: 0.929,
     gradcam_image: sample.gradcamHeatmap // Evaluated to JavaScript 'undefined'!
   });
   ```
3. **Image Tag String Interpolation**:
   When the clinician toggled to Heatmap mode, the JSX rendered:
   ```jsx
   <img src={`data:image/png;base64,${eyeResult.gradcam_image}`} alt="Grad-CAM" />
   ```
   Because `eyeResult.gradcam_image` was `undefined`, JavaScript coerced the value to the literal string:
   $$\text{"data:image/png;base64,undefined"}$$
4. **Browser Reaction**:
   The browser engine attempted to decode `"undefined"` as a Base64 payload. Since `"undefined"` is not valid Base64 image data, decoding threw `ERR_INVALID_URL`, causing the browser to render the broken-image icon.

---

### 6.3 The Comprehensive Solution Architecture
The repair implemented a three-tier defense ensuring this error can never occur again:
1. **Asset Generation & Preset Repair**:
   Real Grad-CAM heatmaps were generated for all four clinical conditions and saved to `frontend/public/samples/`. `DEMO_SAMPLES` was updated with valid `gradcamHeatmap` paths.
2. **Multi-Source Image Resolver (`resolveGradCamSource`)**:
   In `EyeAssessment.jsx`, raw string interpolation was replaced with a robust multi-source resolver:
   ```javascript
   const resolveGradCamSource = (result) => {
     if (!result) return null;
     const img = result.gradcam_image || result.gradcamHeatmap || result.gradcam_url;
     if (!img || typeof img !== 'string') return null;

     const clean = img.trim();
     // Guard against the literal string "undefined" or empty payloads
     if (clean === 'undefined' || clean === 'null' || clean === '') return null;

     // 1. Direct web path or data URI
     if (clean.startsWith('http') || clean.startsWith('/') || clean.startsWith('data:image')) {
       return clean;
     }
     // 2. Raw Base64 payload -> Prepend data URI header safely
     if (clean.length > 50) {
       return `data:image/png;base64,${clean}`;
     }
     return null;
   };
   ```
3. **Graceful Fallback UI**:
   If the resolver returns `null` or if the browser fires an `onError` event, the broken image is unmounted and replaced with a clean clinical diagnostic alert:
   > *"Visualization Notice: Grad-CAM heatmap preview could not be resolved. Diagnostic classification and clinical confidence metrics remain fully verified."*
4. **Verification**:
   Verified via automated headless browser tests using Puppeteer Core across both ports 5173 and 8000, confirming that the heatmap renders cleanly with a 200 status.

---

# CHAPTER 7: DATABASE ARCHITECTURE & DATA INTEGRITY

```
+----------------------------------------------------------------------------------------------------+
|                                      RELATIONAL SCHEMA (SQLite 3.x)                                 |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  +---------------------------+                 +--------------------------------+                  |
|  |         patients          |                 |            sessions            |                  |
|  |---------------------------|                 |--------------------------------|                  |
|  | id (PK, Integer)          |<------+         | id (PK, Integer)               |<---+             |
|  | name (String 100)         |       |         | patient_id (FK -> patients.id) |----+             |
|  | age (Integer)             |       +---------| timestamp (DateTime)           |    |             |
|  | gender (String 20)        |                 | notes (Text, Nullable)         |    |             |
|  | contact (String 50)       |                 | overall_risk_level (String 20) |    |             |
|  | created_at (DateTime)     |                 | created_at (DateTime)          |    |             |
|  +---------------------------+                 +--------------------------------+    |             |
|                                                                                      |             |
|                   +----------------------------------+-------------------------------+             |
|                   |                                  |                                             |
|  +----------------v------------------+  +------------v-----------------------+                     |
|  |             eye_data              |  |         physiological_data         |                     |
|  |-----------------------------------|  |------------------------------------|                     |
|  | id (PK, Integer)                  |  | id (PK, Integer)                   |                     |
|  | session_id (FK -> sessions.id)    |  | session_id (FK -> sessions.id)     |                     |
|  | image_path (String 255)           |  | heart_rate_bpm (Float)             |                     |
|  | prediction (String 50)            |  | spo2_percent (Float)               |                     |
|  | class_index (Integer)             |  | temperature_celsius (Float)        |                     |
|  | confidence (Float)                |  | hr_status (String 20)              |                     |
|  | risk_level (String 20)            |  | spo2_status (String 20)            |                     |
|  | gradcam_path (String 255)         |  | temp_status (String 20)            |                     |
|  | all_probabilities (JSON)          |  | overall_status (String 20)         |                     |
|  | quality_checks (JSON)             |  | risk_level (String 20)             |                     |
|  | created_at (DateTime)             |  | summary (Text)                     |                     |
|  | timestamp (DateTime)              |  | timestamp (DateTime)               |                     |
|  +-----------------------------------+  +------------------------------------+                     |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 7.1 Relational Architecture & Foreign Key Constraints
Data integrity is maintained through four relational tables managed via SQLAlchemy 2.1 ORM:
- **`patients`**: Master table for demographic records.
- **`sessions`**: Represents an individual clinical encounter. Links to `patients.id` via foreign key.
- **`eye_data`**: 1-to-1 / 1-to-many relationship with `sessions`. Stores classification outputs, confidence, JSON probability distribution, and file paths to raw and Grad-CAM images.
- **`physiological_data`**: 1-to-1 relationship with `sessions`. Stores raw vital measurements and clinical status classifications.

### 7.2 Cascade Deletions & Orphan Cleanup
When a patient record or screening session is deleted:
- The database enforces `cascade="all, delete-orphan"` on foreign key relationships.
- In `backend/routes/sessions.py` and `patients.py`, a post-delete trigger inspects associated `image_path` and `gradcam_path` records, deleting the physical files from the `backend/uploads/` directory on disk to prevent filesystem bloat.

### 7.3 Concurrency & SQLite Performance
- **Write-Ahead Logging (WAL)**: SQLite is initialized in WAL mode. This allows concurrent readers to query patient histories without blocking an active session write operation.
- **Connection Scope**: SQLAlchemy sessions are injected per-request using FastAPI's `Depends(get_db)`. The session is closed automatically at the conclusion of the HTTP request lifecycle, preventing database connection leaks.

---

# CHAPTER 8: DEPLOYMENT, ENVIRONMENT & INFRASTRUCTURE

### 8.1 Development vs. Production Architecture
The application supports two distinct operational modes:

| Dimension | Local Development Mode | Production Unified Container Mode |
|-----------|------------------------|-----------------------------------|
| **Frontend Host** | Vite Dev Server (`localhost:5173`) with HMR | Static bundle compiled to `frontend/dist/` |
| **Backend Host** | Uvicorn (`localhost:8000`) with auto-reload | Uvicorn running production worker pool |
| **Network Routing** | Cross-Origin requests; CORS middleware enabled | Same-Origin; FastAPI serves both API and SPA |
| **Static Assets** | Served dynamically via Vite ES Modules | Served via Starlette `StaticFiles(html=True)` |
| **Process Count** | 2 processes (Node.js + Python) | 1 unified process (`python -m backend.main`) |

### 8.2 Production Single-Process Mounting
To simplify clinic deployment into a single offline executable or Docker container, FastAPI mounts the compiled frontend build directly:
```python
# In backend/main.py
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
```
When a clinician navigates to `http://localhost:8000`, the server delivers `index.html` and static JS/CSS bundles. All API requests route to `/api/v1/*`.

### 8.3 Environment Configuration (`backend/config.py`)
All system parameters are externalized using Pydantic Settings, read from environment variables or `.env`:
- `BACKEND_PORT`: Port for Uvicorn server (default: `8000`).
- `DATABASE_URL`: Database connection string (default: `sqlite:///./backend/health_screening.db`).
- `MODEL_WEIGHTS_PATH`: Path to model checkpoint (default: `ml/weights/eye_model.pth`).
- `RATE_LIMIT_PER_MINUTE`: Request quota per IP (default: `60`).
- `MAX_UPLOAD_SIZE_BYTES`: Upload ceiling (default: `31457280` = 30 MB).
- `CORS_ORIGINS`: JSON array of authorized origins.

### 8.4 Health Check Probes
The backend exposes three standardized health probes for container orchestrators:
- **`GET /health`**: Returns basic operational status (`{"status": "ok", "version": "1.0.0"}`).
- **`GET /ready`**: Verifies that the database connection is responsive and the PyTorch model weights are loaded in memory.
- **`GET /live`**: Verifies that the event loop is active and not deadlocked.

---

# CHAPTER 9: ENGINEERING TRADE-OFFS & ARCHITECTURE DEFENSE

Every architectural choice represents an intentional trade-off between competing technical priorities. Use this matrix to defend design decisions during panel examinations:

```
+----------------------------------------------------------------------------------------------------+
|                                    10 KEY ARCHITECTURAL DECISIONS                                  |
+----------------------------------------------------------------------------------------------------+
```

### 1. Unified Monolith vs. Distributed Microservices
- **Choice Made**: Modular monolith (FastAPI serving API and mounting React SPA).
- **Alternative**: Separate frontend service, backend API gateway, ML inference microservice, and database container.
- **Why Our Choice Makes Sense**: Microservices introduce distributed network latency, container orchestration complexity (Kubernetes), and failure modes across network boundaries. For rural clinics operating with limited IT support or offline laptops, a single unified process running on one port guarantees zero network orchestration overhead.

### 2. SQLite vs. PostgreSQL
- **Choice Made**: Embedded SQLite database with SQLAlchemy ORM.
- **Alternative**: Dedicated PostgreSQL or MySQL database server.
- **Why Our Choice Makes Sense**: PostgreSQL requires a running daemon, user authentication, and network port management. SQLite requires zero configuration, has zero network overhead, and stores all records in a single portable file (`health_screening.db`). Because SQLAlchemy 2.0 ORM is used, migrating to PostgreSQL in a multi-hospital cloud deployment requires changing only one line (`DATABASE_URL`).

### 3. Base64 in JSON vs. Binary Image Streaming
- **Choice Made**: Dual delivery (Base64 data URI in JSON payload + server URL `/uploads/gradcam/*.png`).
- **Alternative**: Strictly returning image URLs and requiring the client to issue a secondary HTTP GET request.
- **Why Our Choice Makes Sense**: The Base64 string allows the React UI to render the Grad-CAM heatmap instantly upon receiving the classification response, eliminating an extra HTTP round-trip latency. The URL fallback ensures that large images can be linked, cached, or embedded into diagnostic PDF reports.

### 4. Component Local State vs. Global Redux Store
- **Choice Made**: Native React hooks (`useState`, `useRef`, `useMemo`) with prop passing.
- **Alternative**: Global state management with Redux Toolkit or Zustand.
- **Why Our Choice Makes Sense**: The application has two distinct visual domains: Retinal Eye Assessment and Physiological Vitals Assessment. The data flows unidirectionally from inputs to the unified screening handler. Adding Redux would introduce hundreds of lines of boilerplate actions, reducers, and selectors without providing any architectural benefit.

### 5. In-Process PyTorch Inference vs. Dedicated Triton / TorchServe Server
- **Choice Made**: In-process PyTorch model evaluation wrapped in `MLService`.
- **Alternative**: Triton Inference Server or TorchServe communicating via gRPC.
- **Why Our Choice Makes Sense**: Triton and TorchServe are designed for high-concurrency cloud clusters handling thousands of requests per second. In our clinical screening setting, requests are dispatched sequentially per patient encounter. In-process execution eliminates gRPC serialization overhead, simplifies memory management, and avoids running a secondary Java or C++ daemon.

### 6. Tailwind CSS vs. Pre-Styled Component Libraries (MUI / Ant Design)
- **Choice Made**: Tailwind CSS utility-first styling.
- **Alternative**: Material UI (MUI), Ant Design, or Chakra UI.
- **Why Our Choice Makes Sense**: Medical diagnostic interfaces require high-contrast, dark-mode styling with specialized Reticle HUD overlays and millimeter-precise layouts. Pre-built component libraries force opinionated themes that require extensive CSS overrides. Tailwind delivers a custom, clinical aesthetic while purging unused styles down to <25 KB.

### 7. Modern Vite vs. Legacy Create-React-App (Webpack)
- **Choice Made**: Vite 8.3.4.
- **Alternative**: Create-React-App (CRA) or custom Webpack configuration.
- **Why Our Choice Makes Sense**: CRA is officially deprecated by the React team. Webpack requires re-bundling the entire application on code changes. Vite leverages browser-native ES Modules during development to provide instant Hot Module Replacement (<50ms) and uses Rollup for highly optimized tree-shaken production builds.

### 8. Synchronous Inference Route vs. Asynchronous Job Queue (Celery + Redis)
- **Choice Made**: Synchronous HTTP POST request/response cycle.
- **Alternative**: Asynchronous task queue where requests return a `task_id` and poll `/task/{id}`.
- **Why Our Choice Makes Sense**: Because DenseNet-201 inference executes in 3.4ms on GPU (and <85ms on CPU), the entire request lifecycle completes in under 120ms. Introducing Celery, Redis brokers, worker processes, and client polling would add significant architectural complexity for a task that finishes almost instantaneously.

### 9. Backend ReportLab PDF Compilation vs. Client-Side `window.print()` / jsPDF
- **Choice Made**: Server-side ReportLab 5.0 vector PDF compilation.
- **Alternative**: Client-side PDF generation using `window.print()` or jsPDF.
- **Why Our Choice Makes Sense**: `window.print()` produces inconsistent outputs across different operating systems, screen resolutions, and browsers. ReportLab compiles pixel-perfect, tamper-resistant vector documents with consistent typography, clinical tables, and embedded high-resolution fundus images regardless of the client machine.

### 10. In-Memory Sliding-Window Rate Limiter vs. Distributed Redis Token Bucket
- **Choice Made**: In-memory sliding-window deque limiter in `security.py`.
- **Alternative**: Distributed Redis instance running a Token Bucket algorithm.
- **Why Our Choice Makes Sense**: For single-instance clinic deployments, an in-memory Python deque tracks timestamps with microsecond latency and zero external dependencies. In a horizontally scaled multi-node cloud deployment, this middleware can be swapped for Redis without altering route logic.

---

# CHAPTER 10: FULL-STACK SOFTWARE ENGINEERING VIVA SUITE (75 QUESTIONS)

---

## PART A: CORE FULL-STACK & WEB TECHNOLOGIES (40 QUESTIONS)

#### Q1: What is the Virtual DOM in React and how does reconciliation work?
**Answer**: The Virtual DOM is a lightweight in-memory JavaScript representation of the real DOM tree. When component state changes, React creates a new Virtual DOM tree, performs a diffing comparison against the previous Virtual DOM tree (the reconciliation algorithm), and computes the minimum set of mutations needed. It then applies these batch updates to the real browser DOM in a single pass, avoiding costly full-page layout reflows and repaints.

#### Q2: How does React 19 differ from previous versions?
**Answer**: React 19 introduces the React Server Components paradigm, improved asset loading, built-in Actions for form handling, direct support for document metadata (`<title>`, `<meta>`), and deprecation of legacy lifecycle methods. It optimizes compiler-driven memoization, reducing the need for manual `useMemo` and `useCallback` annotations.

#### Q3: Why is Hot Module Replacement (HMR) in Vite significantly faster than Webpack?
**Answer**: Webpack bundles the entire application's dependency graph into a bundle file before serving, meaning any code edit requires rebuilding parts of that bundle. Vite leverages native ES Modules (ESM) supported by modern browsers. During development, Vite only transforms the specific file that changed and sends it over native WebSocket HMR, achieving updates in under 50 milliseconds regardless of application size.

#### Q4: What is the difference between ASGI and WSGI in Python web servers?
**Answer**: WSGI (Web Server Gateway Interface, e.g., Flask/Django with Gunicorn) is a synchronous, blocking standard where each incoming request occupies an entire worker thread until completion. ASGI (Asynchronous Server Gateway Interface, e.g., FastAPI with Uvicorn) supports asynchronous coroutines (`async`/`await`), allowing a single thread to handle thousands of concurrent connections using an asynchronous event loop (`asyncio`/`uvloop`).

#### Q5: What role does Pydantic play in FastAPI request validation?
**Answer**: Pydantic parses and validates incoming HTTP request bodies against Python type annotations. If an incoming JSON payload violates the schema (e.g., passing a string for heart rate or omitting a required field), Pydantic automatically rejects the request with HTTP `422 Unprocessable Entity` and returns a structured JSON error indicating the exact field and validation violation before application logic runs.

#### Q6: Why did you choose Axios over the browser's native `fetch()` API?
**Answer**: While `fetch()` is built-in, Axios provides several operational advantages:
1. Automatic JSON data serialization and parsing (fetch requires explicit `response.json()`).
2. Built-in request and response interceptors for global logging and error handling.
3. Native upload progress tracking for large file uploads.
4. Automatic rejection of promises on HTTP error codes like 4xx/5xx (fetch resolves successfully unless a network failure occurs).

#### Q7: What is Cross-Origin Resource Sharing (CORS) and why does it occur?
**Answer**: CORS is a browser security mechanism that restricts web pages from making HTTP requests to a domain, port, or protocol different from the one that served the page. When the React dev server runs on `localhost:5173` and calls the backend on `localhost:8000`, the browser issues an `OPTIONS` preflight request. The backend's `CORSMiddleware` must respond with `Access-Control-Allow-Origin: http://localhost:5173` to allow the browser to proceed.

#### Q8: What is Content Security Policy (CSP) and how does our backend implement it?
**Answer**: CSP is an HTTP response header that prevents Cross-Site Scripting (XSS) and data injection attacks by restricting the origins from which scripts, styles, images, and fonts can be loaded. Our `SecurityHeadersMiddleware` sets `Content-Security-Policy` to permit Google Fonts and self-hosted assets while blocking inline executable scripts and unauthorized external script execution.

#### Q9: What is the purpose of the `X-Request-ID` correlation ID?
**Answer**: In distributed systems, a single user click may trigger multiple log lines across middleware, routers, database calls, and ML pipelines. The `CorrelationIdMiddleware` assigns a unique UUID to every request, attaches it to the response header, and injects it into every log line. This allows developers to isolate and trace the complete lifecycle of a failed request in production logs.

#### Q10: How does the sliding-window rate limiter protect the backend?
**Answer**: It records timestamps of requests per client IP inside a double-ended queue (`collections.deque`). On each incoming request, timestamps older than 60 seconds are purged. If the remaining count exceeds the threshold (60 requests), the request is rejected with `429 Too Many Requests`. This prevents brute-force attacks and denial-of-service abuse.

#### Q11: Explain the difference between `application/json` and `multipart/form-data`.
**Answer**: `application/json` encodes structured text data as key-value pairs; binary data must be Base64-encoded, increasing size by ~33%. `multipart/form-data` divides the payload into separate body parts separated by boundary strings, allowing raw binary image files to stream directly alongside form fields without encoding overhead.

#### Q12: What is the purpose of `torch.no_grad()` during inference?
**Answer**: During training, PyTorch constructs a dynamic computation graph to track gradients for backpropagation, consuming significant VRAM and CPU cycles. Wrapping inference inside `with torch.no_grad():` disables this tracking mechanism, cutting memory usage by >50% and speeding up forward inference.

#### Q13: What happens during model pre-warming on startup?
**Answer**: On startup, the backend calls `ml_service.prewarm()`. This loads the DenseNet-201 weights into memory and executes a dummy forward pass through the network using a synthetic tensor. This initializes CUDA memory pools, loads driver libraries, and compiles execution kernels, eliminating first-request latency ("cold start") for patients.

#### Q14: How does SQLAlchemy prevent SQL injection attacks?
**Answer**: SQLAlchemy uses parameterized queries. Instead of concatenating user input directly into SQL strings (`"SELECT * FROM patients WHERE name = '" + user_input + "'"`), SQLAlchemy passes input parameters separately from the SQL statement template. The database driver treats parameters strictly as literal values, rendering SQL injection impossible.

#### Q15: What is SQLite Write-Ahead Logging (WAL) mode?
**Answer**: In traditional rollback journal mode, writing to SQLite requires locking the entire database, blocking readers. In WAL mode, changes are appended to a separate `.wal` file. Readers can query the main database file concurrently without blocking or being blocked by active writes, improving concurrent clinical reading throughput.

#### Q16: How does ReportLab compile PDF documents programmatically?
**Answer**: ReportLab uses a canvas model with flowable elements (paragraphs, tables, images, spacers). A `SimpleDocTemplate` manages page layout, margins, and page breaking. When `doc.build(story)` is called, ReportLab processes flowable elements, executes text wrapping and table layout algorithms, embeds image bytes, and writes valid vector PDF primitives to an in-memory buffer (`io.BytesIO`).

#### Q17: What is an HTTP 422 Unprocessable Entity error?
**Answer**: HTTP 422 indicates that the server understands the content type and syntax of the request entity, but was unable to process the contained instructions due to semantic validation errors. For example, sending a valid JSON payload where `spo2_percent` is `150.0` triggers a 422 because oxygen saturation cannot exceed 100%.

#### Q18: What is an HTTP 413 Payload Too Large error?
**Answer**: HTTP 413 is returned when the request body exceeds the maximum size allowed by the server. Our `RequestSizeLimitMiddleware` inspects the `Content-Length` header and rejects requests over 30 MB to prevent memory exhaustion attacks.

#### Q19: What is the difference between state and props in React?
**Answer**: **Props** (properties) are read-only inputs passed down from parent components to child components to configure them. **State** is mutable data managed locally within a component that can change over time in response to user actions, triggering re-rendering when updated via `useState`.

#### Q20: What is the purpose of the `useEffect` hook in React?
**Answer**: `useEffect` lets functional components perform side effects, such as fetching data from an API, subscribing to browser events, or interacting directly with the DOM. It runs after the component renders, and its dependency array controls whether it runs once on mount (`[]`), on every render (no array), or only when specific variables change (`[dep1]`).

#### Q21: What is the purpose of `useRef` and how does it differ from `useState`?
**Answer**: `useRef` returns a mutable object whose `.current` property persists across renders. Updating a `ref` does **not** trigger a component re-render, making it ideal for storing direct references to DOM nodes (like an HTML5 Canvas or file input element) or keeping timer IDs. Updating `useState` always triggers a component re-render.

#### Q22: What is the role of `useMemo` in React?
**Answer**: `useMemo` caches the calculated result of an expensive calculation between renders. It only recomputes the memoized value when one of its dependencies changes, preventing redundant CPU-intensive computations on every render.

#### Q23: What is the role of `useCallback` in React?
**Answer**: `useCallback` caches a function definition between renders. Passing un-memoized inline functions to child components causes those children to re-render unnecessarily because function references change on every parent render. `useCallback` preserves the function reference across renders.

#### Q24: What is tree-shaking in modern bundlers?
**Answer**: Tree-shaking is a dead-code elimination technique used by Rollup/Vite. Because ES Modules use static `import` and `export` statements, the bundler can determine which functions are actually imported and executed, discarding unused exports from third-party libraries and reducing production bundle sizes.

#### Q25: How does Tailwind CSS reduce final CSS bundle size?
**Answer**: Tailwind v4 uses a high-performance compiler that scans source files for utility class names. It generates CSS rules *only* for the specific utility classes actually used in the code, purging everything else. As a result, production CSS files often compress to under 25 KB.

#### Q26: What is a Single Page Application (SPA)?
**Answer**: An SPA is a web application that loads a single HTML shell (`index.html`) on initial visit. Subsequent page navigation is intercepted client-side by a router (e.g., React Router) which dynamically swaps components in the DOM and fetches data via AJAX/Axios without reloading the full HTML page from the server.

#### Q27: How does client-side routing work without server redirects?
**Answer**: React Router uses the HTML5 History API (`window.history.pushState` and `popstate` events) to update the browser URL bar and manage history stack entries without triggering a full page reload. In production, the backend server must be configured to return `index.html` for any unrecognized route so React Router can handle routing.

#### Q28: What is an Error Boundary in React?
**Answer**: An Error Boundary is a component that catches JavaScript errors anywhere in its child component tree, logs the error, and displays a fallback UI instead of crashing the entire application. In our project, the Grad-CAM image viewer implements a fallback error alert when image decoding fails.

#### Q29: What is dependency injection in FastAPI?
**Answer**: Dependency injection allows functions or route handlers to declare dependencies (such as database sessions, authentication credentials, or configuration settings) as parameters using `Depends()`. FastAPI resolves, creates, and injects these dependencies at runtime, and handles cleanup (like closing database sessions) automatically when the request completes.

#### Q30: What is a database migration and why is it needed?
**Answer**: A database migration is a version-controlled script that modifies a database schema (adding columns, renaming tables, modifying constraints) over time without losing existing data. Tools like Alembic manage migrations in SQLAlchemy applications.

#### Q31: What is the purpose of foreign key cascade delete rules?
**Answer**: Cascade delete rules (`ON DELETE CASCADE`) ensure that when a parent record (e.g., a patient in `patients`) is deleted, all related child records (e.g., all associated rows in `sessions`, `eye_data`, and `physiological_data`) are automatically deleted by the database engine. This prevents orphaned records and maintains relational integrity.

#### Q32: What is an in-memory stream (`io.BytesIO`) in Python?
**Answer**: `io.BytesIO` is an in-memory binary stream that behaves like a file object without writing bytes to physical disk storage. Our ReportLab PDF engine compiles reports directly into a `BytesIO` buffer, which is returned over HTTP as a streaming response, minimizing disk I/O.

#### Q33: What is the purpose of HTTP `Content-Disposition` header?
**Answer**: The `Content-Disposition` header tells the browser how to present the response payload. Setting it to `inline` renders the file inside the browser tab; setting it to `attachment; filename="report.pdf"` forces the browser to prompt the user with a download dialog using the specified filename.

#### Q34: What is the difference between synchronous and asynchronous code in Python?
**Answer**: Synchronous code executes sequentially line-by-line; if a line performs a slow I/O operation (like reading from disk or waiting for a remote server), the entire thread blocks. Asynchronous code (`async`/`await`) yields control back to the event loop while waiting for I/O, allowing the thread to process other incoming requests in the meantime.

#### Q35: How does the backend handle static file serving?
**Answer**: Starlette's `StaticFiles` middleware maps a physical directory on disk (e.g., `backend/uploads/` or `frontend/dist/`) to a URL path (e.g., `/uploads/` or `/`). It handles HTTP range requests, MIME-type resolution, and browser cache headers (`ETag`, `Last-Modified`).

#### Q36: What is a UUID and why is it used for file uploads?
**Answer**: A UUID (Universally Unique Identifier) is a 128-bit label generated using cryptographic randomness (UUID v4). Uploaded images are saved with filenames like `fundus_20261012_183500_a9f2.jpg` to prevent name collisions and prevent malicious users from guessing file paths on the server.

#### Q37: What is an E2E (End-to-End) test?
**Answer**: An E2E test tests the entire software stack from the user interface down to the database and ML model. Tools like Puppeteer launch a real headless browser, simulate user clicks, file uploads, and form inputs, and verify that the UI renders expected elements, confirming that frontend, backend, and database interact correctly.

#### Q38: What is the difference between unit testing and integration testing?
**Answer**: Unit testing tests individual functions, classes, or modules in isolation (e.g., testing the Laplacian blur formula with a synthetic array). Integration testing tests the interaction between multiple interconnected modules (e.g., testing that an API endpoint receives an upload, runs ML inference, saves to SQLite, and returns JSON).

#### Q39: What is `torchvision.transforms.Normalize` doing mathematically?
**Answer**: It normalizes an input tensor using the mean ($\mu$) and standard deviation ($\sigma$) of the ImageNet dataset:
$$\text{tensor}_{\text{norm}} = \frac{\text{tensor} - \mu}{\sigma}$$
For RGB channels, $\mu = [0.485, 0.456, 0.406]$ and $\sigma = [0.229, 0.224, 0.225]$. This ensures the input distribution matches what the pre-trained DenseNet-201 weights expect.

#### Q40: What is Laplacian variance and how does it detect blurry images?
**Answer**: The Laplacian operator is a second-order derivative mask that highlights rapid intensity changes (edges). In a sharp image, edges are crisp, resulting in high variance across Laplacian pixel values. In a blurry image, edges are smoothed out, causing low Laplacian variance. An image with variance below a calibrated threshold (e.g., $<100$) is flagged as blurry.

---

## PART B: SYSTEM DESIGN & OPERATIONAL SCENARIOS (15 QUESTIONS)

#### Q41: How would you scale this architecture to support 10,000 concurrent clinic users across 50 regional hospitals?
**Answer**:
1. **Frontend**: Deploy the static Vite build to a global Content Delivery Network (Cloudflare or AWS CloudFront) for edge caching.
2. **Backend**: Containerize FastAPI in Docker and deploy across a Kubernetes cluster with Horizontal Pod Autoscaling (HPA) based on CPU and request latency.
3. **ML Inference**: Decouple ML inference from the API server. Move DenseNet-201 onto a Triton Inference Server cluster with GPU autoscaling, routing inference requests via high-speed gRPC.
4. **Database**: Migrate from SQLite to an Amazon RDS PostgreSQL cluster with multi-AZ replication, connection pooling (PgBouncer), and read replicas for clinical reporting queries.
5. **Asset Storage**: Move uploaded images from local disk to an S3 object store with pre-signed upload URLs.

#### Q42: How does the system function in an offline rural clinic with no internet connectivity?
**Answer**:
The entire platform was engineered with zero external cloud dependencies:
1. DenseNet-201 weights (`70.34 MB`) are stored locally on the laptop/edge device.
2. SQLite stores patient records and screening sessions locally in a single file (`health_screening.db`).
3. ReportLab compiles diagnostic PDFs locally in memory.
4. The React SPA is bundled and served locally by FastAPI over `localhost`.
Health workers can run complete screening sessions and generate printable diagnostic reports in remote areas without internet access.

#### Q43: How would you ensure HIPAA and GDPR compliance for patient medical data?
**Answer**:
1. **Data at Rest**: Encrypt SQLite databases and filesystem upload directories using AES-256.
2. **Data in Transit**: Enforce TLS 1.3 encryption across all network communication.
3. **De-Identification**: Strip EXIF metadata from uploaded retinal images before saving to disk.
4. **Access Control**: Implement Role-Based Access Control (RBAC) with JWT authentication, ensuring only authorized physicians can view patient records.
5. **Audit Trails**: Maintain append-only audit logs recording every access, view, and export of patient data.

#### Q44: How would you handle a sudden surge of 500 simultaneous image uploads without crashing the server?
**Answer**:
1. **Rate Limiting**: The sliding-window rate limiter rejects clients sending more than 60 requests/minute with HTTP 429.
2. **Asynchronous Task Queue**: In production, introduce a Celery/Redis task queue. Incoming requests write the image to object storage, push a task to Redis, and return an HTTP `202 Accepted` with a task ID.
3. **Worker Throttling**: A fixed pool of GPU workers processes tasks sequentially from the queue, preventing GPU Out-Of-Memory (OOM) crashes by limiting concurrent model forward passes.

#### Q45: What happens if the server crashes or loses power during an active screening session?
**Answer**:
1. **Database Atomicity**: SQLite uses Write-Ahead Logging (WAL). Uncommitted transactions are automatically rolled back upon restart, preventing database corruption.
2. **Session Persistence**: The frontend caches draft vitals inputs in browser `sessionStorage`, allowing the clinician to restore their session upon reconnection.
3. **Health Probes**: Container orchestrators (Docker/systemd) monitor `/live` and restart the Uvicorn process automatically.

#### Q46: How would you integrate direct retinal camera hardware via USB or WebRTC?
**Answer**:
1. Use the HTML5 MediaDevices API (`navigator.mediaDevices.getUserMedia()`) to stream live video directly from a USB-connected fundus camera into an HTML5 `<video>` element.
2. Render an inspection reticle guide over the video feed to assist the operator in centering the optic disc.
3. Implement a client-side frame analyzer using HTML5 Canvas that calculates image brightness and blur in real-time, automatically capturing the frame when optimal focus and illumination are achieved.

#### Q47: How do you prevent sensitive patient information from leaking through browser caching?
**Answer**:
The `SecurityHeadersMiddleware` injects strict cache-control headers on all clinical API responses:
```http
Cache-Control: no-store, no-cache, must-revalidate, max-age=0
Pragma: no-cache
Expires: 0
```
This instructs the browser not to cache patient diagnostic data, images, or PDF reports in local disk cache, preventing data exposure on shared clinic computers.

#### Q48: How would you implement zero-downtime deployments for new ML model weights?
**Answer**:
1. **Blue-Green Deployment**: Run two identical backend environments ("Blue" active, "Green" idle).
2. Deploy the updated model checkpoint to the "Green" environment and run health probes (`/ready`) to pre-warm the model and verify inference accuracy.
3. Once verified, switch the reverse proxy (Nginx or AWS ALB) to route live traffic to "Green".
4. If anomalies occur, traffic can be switched back to "Blue" in under one second.

#### Q49: How would you handle multilingual support for community health workers?
**Answer**:
1. Implement internationalization (i18n) on the frontend using `react-i18next`.
2. Extract all UI labels, clinical instructions, and risk warnings into JSON translation files (e.g., `en.json`, `hi.json`, `es.json`).
3. Allow health workers to toggle their preferred language from the navigation bar, switching the interface dynamically without reloading the page.
4. Pass the selected language code to the ReportLab PDF generator to compile diagnostic reports in the patient's native language.

#### Q50: How do you verify that the database schema is consistent between developer laptops and production?
**Answer**:
Use Alembic, SQLAlchemy's migration framework:
1. Migrations are stored as version-controlled Python scripts in `backend/alembic/versions/`.
2. On deployment startup, the system executes `alembic upgrade head`.
3. Alembic inspects the `alembic_version` table in the database and applies any pending migrations sequentially, ensuring schema consistency across environments.

#### Q51: How would you monitor system performance and detect model accuracy drift in production?
**Answer**:
1. **Application Metrics**: Expose Prometheus metrics from FastAPI tracking request throughput, latency (p50, p95, p99), error rates (4xx/5xx), and GPU memory usage.
2. **Prediction Distribution Logging**: Log the predicted class distribution over time (e.g., percentage of Cataract vs Normal cases).
3. **Drift Detection**: If the proportion of a disease class shifts significantly from historical baselines, alert clinicians to investigate potential camera calibration issues or population shifts.

#### Q52: What is the risk of allowing arbitrary image file uploads and how do we prevent it?
**Answer**:
Attackers may attempt to upload executable scripts (e.g., PHP, shell scripts) disguised as images, or exploit buffer overflow vulnerabilities in image decoding libraries. We mitigate this through defense in depth:
1. **MIME-Type & Extension Whitelist**: Reject any file without approved image extensions (`.jpg`, `.jpeg`, `.png`).
2. **Magic Byte Inspection**: Inspect the initial bytes of the file stream to verify valid JPEG (`FF D8 FF`) or PNG (`89 50 4E 47`) signatures.
3. **Image Re-Encoding**: Decode the uploaded file through Pillow and re-encode it to clean pixels, stripping any malicious embedded metadata or polyglot payloads.
4. **Execution Prevention**: Store uploaded files outside the web root in a directory with execution permissions disabled.

#### Q53: How would you implement multi-tenant data isolation for different hospital networks?
**Answer**:
1. **Row-Level Security (RLS)**: Add a `hospital_id` column to every table (`patients`, `sessions`, `eye_data`).
2. **Tenant Scoping**: Extract the `hospital_id` from the authenticated user's JWT token in a FastAPI dependency, automatically scoping all database queries (`db.query(Patient).filter(Patient.hospital_id == current_user.hospital_id)`).
3. **Dedicated Schemas**: For enterprise healthcare clients requiring physical separation, configure SQLAlchemy to route queries to tenant-specific database schemas.

#### Q54: How would you optimize image loading performance on low-bandwidth mobile connections?
**Answer**:
1. **Modern Formats**: Convert uploaded JPEG/PNG images to modern WebP format on the backend, reducing file size by 30–50% at equivalent visual quality.
2. **Responsive Thumbnails**: Generate low-resolution thumbnails ($128 \times 128$) for patient history lists, loading full-resolution fundus images ($512 \times 512$) only when drilling down into detailed analysis.
3. **Client-Side Caching**: Cache static UI assets and sample images using browser Service Workers and CacheStorage.

#### Q55: How do you handle database backup and disaster recovery for SQLite?
**Answer**:
1. **Online Backup API**: Use SQLite's native backup API (`sqlite3_backup_step`) or Python's `connection.backup()` method to create consistent database snapshots without stopping the server.
2. **Automated Cron Jobs**: Schedule periodic snapshots (e.g., hourly) to a secondary physical drive or encrypted cloud backup.
3. **Point-in-Time Recovery**: Retain WAL files alongside database snapshots to allow point-in-time recovery up to the moment of failure.

---

## PART C: LIVE DEBUGGING & INCIDENT RESPONSE (10 QUESTIONS)

#### Q56: Incident: Clinicians report that the frontend is completely blank and the browser console shows `Uncaught TypeError: Cannot read properties of undefined (reading 'map')`. How do you diagnose and fix this?
**Answer**:
1. **Diagnosis**: This error occurs when a React component attempts to call `.map()` on an array variable that is `undefined` or `null` (typically before an API call has completed or when an API returns an error response).
2. **Trace**: Check the component stack in the console to identify the failing component (e.g., `PatientsPage.jsx` trying to render `patients.map(...)`).
3. **Fix**:
   - Initialize the state variable with an empty array: `const [patients, setPatients] = useState([]);`.
   - Add optional chaining and fallback guards in JSX: `patients?.map(...) || <p>No records found.</p>`.
   - Verify that the API error handler sets state cleanly rather than leaving it undefined.

#### Q57: Incident: The backend is returning HTTP 504 Gateway Timeout on image upload requests. What is your troubleshooting sequence?
**Answer**:
1. Check backend CPU and GPU utilization (`nvidia-smi`, `htop`). If GPU VRAM is 100% full, PyTorch may be deadlocked.
2. Check Uvicorn process logs using the correlation ID (`X-Request-ID`) to see where the request stalled.
3. Determine if the bottleneck is in file I/O (slow network upload), image quality checks (unoptimized OpenCV loops), or ML inference.
4. Verify if Uvicorn worker threads are blocked by a synchronous computation that should be offloaded.
5. Check if the reverse proxy (Nginx) has a timeout setting (e.g., `proxy_read_timeout 60s`) that is too short for large file uploads.

#### Q58: Incident: The browser console shows `Access to XMLHttpRequest at 'http://localhost:8000/api/v1/predict_eye' from origin 'http://localhost:5173' has been blocked by CORS policy`. How do you resolve this?
**Answer**:
1. Inspect the incoming request headers to determine the client origin (`Origin: http://localhost:5173`).
2. Check `backend/config.py` and `backend/main.py` to confirm that `http://localhost:5173` is included in the `CORS_ORIGINS` list.
3. Verify that the browser's `OPTIONS` preflight request is returning HTTP 200 with appropriate headers (`Access-Control-Allow-Origin`, `Access-Control-Allow-Methods`, `Access-Control-Allow-Headers`).
4. Ensure that any custom middleware running before `CORSMiddleware` does not raise an unhandled exception on `OPTIONS` requests.

#### Q59: Incident: PyTorch crashes with `CUDA out of memory. Tried to allocate 256.00 MiB`. What caused this and how do you prevent it?
**Answer**:
1. **Cause**:
   - Gradients may be accumulating during inference because `torch.no_grad()` was omitted.
   - Activations or tensors may be staying in GPU memory across requests without being dereferenced.
   - Multiple concurrent inference requests may be overwhelming GPU memory.
2. **Resolution**:
   - Ensure all inference routines are wrapped in `with torch.no_grad():` (except Grad-CAM backpropagation).
   - In the Grad-CAM module, detach intermediate tensors from the computation graph immediately after computing weights: `activations.detach().cpu()`.
   - Call `torch.cuda.empty_cache()` inside error handling blocks to release unreferenced cached memory.

#### Q60: Incident: The frontend displays a broken image icon when switching to Grad-CAM mode (The Part 14 Bug). Walk through the exact debugging steps.
**Answer**:
1. Open Chrome DevTools -> Inspect the broken `<img>` element.
2. Check the `src` attribute value. Discover it is `data:image/png;base64,undefined`.
3. Trace the source of `undefined`: Check the React state (`eyeResult.gradcam_image`).
4. Trace state assignment: Look at how `eyeResult` was set when clicking demo presets. Discover that the preset object in `frontend/src/api.js` lacked the `gradcamHeatmap` property.
5. Fix by providing valid heatmap assets for presets and implementing a safe multi-source resolver in `EyeAssessment.jsx` that validates Base64 length and falls back to a clean alert UI if resolution fails.

#### Q61: Incident: A physician reports that submitting vitals returns HTTP 422 Unprocessable Entity. How do you identify which field failed validation?
**Answer**:
1. Inspect the response body in the DevTools Network tab.
2. Pydantic v2 returns a detailed JSON error payload:
   ```json
   {
     "detail": [
       {
         "type": "greater_than_equal",
         "loc": ["body", "spo2_percent"],
         "msg": "Input should be greater than or equal to 50",
         "input": 42.0
       }
     ]
   }
   ```
3. The `loc` field points directly to `body -> spo2_percent`, and `msg` explains that the entered value (`42.0`) was below the minimum clinical boundary of `50.0%`.

#### Q62: Incident: The diagnostic PDF report downloads with missing patient images. How do you diagnose and fix this?
**Answer**:
1. Check the server logs for `backend/pdf_service.py`.
2. Determine whether the image file path stored in `eye_data.image_path` exists on disk.
3. If an absolute file path was stored and the application was moved to a new directory, relative path resolution will fail.
4. **Fix**: In `pdf_service.py`, implement an image resolver that checks:
   - Whether the file exists on the filesystem.
   - If not, check fallback directories (`backend/uploads/eye_images/`).
   - If still missing, render a placeholder clinical notice box in the PDF rather than letting the entire document build crash.

#### Q63: Incident: Database queries on the patient history page become noticeably sluggish after 5,000 screening sessions are logged. How do you optimize them?
**Answer**:
1. Use `EXPLAIN QUERY PLAN` on the slow SQL query to check if SQLite is performing a full table scan.
2. Identify missing indexes: If querying `sessions` by `patient_id` or ordering by `timestamp`, ensure indexes exist on foreign keys and sort columns:
   ```python
   Index('ix_sessions_patient_id', SessionModel.patient_id)
   Index('ix_sessions_timestamp', SessionModel.timestamp)
   ```
3. Use pagination (`LIMIT 20 OFFSET :page`) in the API route rather than returning all 5,000 records in a single payload.
4. Use SQLAlchemy eager loading (`joinedload`) to fetch patient sessions in a single `JOIN` query, eliminating the $N+1$ query problem.

#### Q64: Incident: The frontend build fails during `npm run build` with `Rollup failed to resolve import`. How do you fix this?
**Answer**:
1. Inspect the Rollup error log to identify the unresolved file path and calling module.
2. Check for case-sensitivity issues: Windows filesystems are case-insensitive, while Linux is case-sensitive. Importing `./Components/Navbar` when the file is named `./components/Navbar.jsx` works on Windows dev but fails in Linux CI/CD.
3. Check for circular imports between components.
4. Verify that dependencies listed in `package.json` match imported package names.

#### Q65: Incident: After deploying updates, users report seeing the old version of the website. How do you resolve this?
**Answer**:
1. **Cause**: The browser has aggressively cached the old `index.html` or static JavaScript bundles.
2. **Resolution**:
   - Configure Vite to include content hashes in output filenames (e.g., `index-C8a9f2.js`). When code changes, the hash changes, forcing the browser to fetch the new file (cache-busting).
   - Configure the backend server to serve `index.html` with `Cache-Control: no-cache, no-store, must-revalidate`, ensuring the browser always checks for the latest HTML shell.

---

## PART D: ARCHITECTURAL DEFENSE & PANEL CHALLENGES (10 QUESTIONS)

#### Q66: Panel Challenge: "Why did you build this as a full-stack web application instead of a standalone desktop app using Python Tkinter or PyQt?"
**Answer**:
A web application provides significant advantages in a healthcare environment:
1. **Cross-Platform Access**: Works on any device with a modern browser (tablets, laptops, clinic workstations) without requiring platform-specific installation or OS dependencies.
2. **Centralized Updates**: Model updates, security patches, and clinical guideline changes are deployed to the server once, updating instantly for all users without manual software reinstalls.
3. **Decoupled Architecture**: The frontend UI is cleanly separated from the backend ML engine via REST APIs, allowing the user interface to be updated without touching clinical inference logic.

#### Q67: Panel Challenge: "Why did you build your own custom CSS design using Tailwind instead of using a standard medical UI kit?"
**Answer**:
Standard UI kits are designed for generic business dashboards and lack the specialized visual tools required for ophthalmic screening:
1. We needed a high-contrast dark clinical theme to minimize visual fatigue for ophthalmologists reviewing fundus scans.
2. We required custom canvas-based Reticle HUD inspection overlays and real-time opacity sliders to blend Grad-CAM heatmaps with original photographs.
3. Tailwind allowed us to build these specialized medical widgets while generating a production stylesheet under 25 KB, avoiding the bundle bloat of heavy component libraries.

#### Q68: Panel Challenge: "Isn't SQLite too toy-like for a medical system? Shouldn't you have used PostgreSQL?"
**Answer**:
SQLite is not a toy database—it is one of the most widely deployed, thoroughly tested database engines in the world, trusted in aircraft flight software and smartphones:
1. In a rural clinic or community health center, the platform often runs on a standalone laptop without dedicated IT staff. SQLite requires zero installation, zero configuration, and stores all records in a single portable file.
2. SQLite handles thousands of reads and writes per second, easily exceeding the concurrency requirements of a local clinic.
3. Because we use SQLAlchemy 2.0 ORM, our data layer is database-agnostic. Migrating to PostgreSQL for a regional hospital deployment requires changing only the `DATABASE_URL` connection string without modifying any application code.

#### Q69: Panel Challenge: "Why return the Grad-CAM image as a Base64 string in JSON rather than just saving it to disk and returning a URL?"
**Answer**:
We implemented a dual-delivery architecture:
1. Returning the Base64 data URI allows the React frontend to render the heatmap immediately upon receiving the classification response, eliminating an extra HTTP GET round-trip and avoiding UI loading flickers.
2. Simultaneously saving the image to disk (`/uploads/gradcam/`) provides a persistent file reference for embedding into diagnostic PDF reports and maintaining historical audit logs.

#### Q70: Panel Challenge: "Why did you use synchronous inference in FastAPI rather than offloading it to a background worker like Celery?"
**Answer**:
DenseNet-201 inference executes in under 4 milliseconds on GPU and under 85 milliseconds on CPU. Offloading a sub-100ms task to Celery would add significant architectural overhead: running a Redis broker, managing worker processes, and requiring the frontend to poll for results. Synchronous execution delivers the complete diagnostic prediction, quality check, and Grad-CAM visualization in under 120ms total round-trip time.

#### Q71: Panel Challenge: "How do you justify keeping the physiological vitals triage separate from the eye disease ML model rather than training a multimodal neural network?"
**Answer**:
This separation is clinically and architecturally deliberate:
1. **Clinical Independence**: Retinal diseases like Cataract or Glaucoma have distinct physical manifestations in the eye. Allowing vital sign abnormalities (like a fever) to influence image classification could cause high-confidence false positives or false negatives.
2. **Data Availability**: In many screening scenarios, a patient may provide a fundus scan without immediate vital signs (or vice versa). Decoupled systems allow each modality to function independently.
3. **Interpretability**: The physiological module uses transparent, deterministic clinical thresholds ($\text{SpO}_2 < 95\%$, $\text{Heart Rate} > 100\text{ BPM}$), ensuring auditability for medical review.

#### Q72: Panel Challenge: "Why use ReportLab on the backend for PDF generation instead of letting the user print the webpage with `window.print()`?"
**Answer**:
`window.print()` is unsuitable for legal medical records:
1. It relies on the client's browser rendering engine, resulting in inconsistent page breaks, fonts, and layouts across different operating systems.
2. ReportLab compiles standardized, pixel-perfect vector documents with locked clinical layouts, institutional headers, exact color palettes, and embedded high-resolution imagery.
3. The backend can digitally sign the generated PDF or store a hash of the document in the database, ensuring auditability and tamper-resistance for medical records.

#### Q73: Panel Challenge: "Why didn't you use WebSockets instead of REST APIs?"
**Answer**:
WebSockets are designed for bidirectional, full-duplex communication (like chat apps or financial tickers). Retinal screening follows a request-response pattern: the user submits an image, the server processes it, and returns the result. REST over HTTP/1.1 or HTTP/2 is simpler, stateless, easily cacheable, works seamlessly through enterprise hospital firewalls, and avoids the memory overhead of maintaining open persistent TCP socket connections.

#### Q74: Panel Challenge: "What happens to patient privacy if someone intercepts the network traffic between the frontend and backend?"
**Answer**:
In production, all network traffic is secured via TLS 1.3 (HTTPS), encrypting data in transit. Furthermore:
1. The backend implements defensive security headers (CSP, HSTS, X-Content-Type-Options).
2. API responses include strict `Cache-Control: no-store` headers to prevent browser caching of sensitive clinical data.
3. Image uploads are stored on disk using randomized UUIDs, preventing unauthorized directory enumeration.

#### Q75: Panel Challenge: "If you had another month to work on this project, what single architectural improvement would you prioritize?"
**Answer**:
I would prioritize implementing an **asynchronous offline sync engine with IndexedDB and Service Workers (Progressive Web App)**:
1. This would allow community health workers to capture fundus images and vitals on mobile devices completely offline in remote villages.
2. Images and clinical data would queue locally in browser IndexedDB storage.
3. When the device reconnects to Wi-Fi or cellular service, a background synchronization worker would batch-upload records to the hospital server, reconcile session IDs, and download completed diagnostic reports automatically.

---

# CHAPTER 11: 1-PAGE FULL-STACK CHEAT SHEET / QUICK DEFENSE CARD

Print or review this page 10 minutes before your panel presentation:

```
====================================================================================================
                        MULTIMODAL HEALTH SCREENING PLATFORM: ARCHITECTURE CHEAT SHEET
====================================================================================================

1. THE ELEVATOR PITCH
   "Our platform is a clinical decision-support system that screens retinal fundus images using a 
   201-layer DenseNet deep learning model and performs deterministic physiological vital-sign triage. 
   It features automated pre-inference image quality checks, sub-50ms inference, mathematical visual 
   explainability via Grad-CAM, and generates pixel-perfect PDF diagnostic reports."

2. KEY SYSTEM NUMBERS
   - Model Architecture     : DenseNet-201 (20.2 million parameters, 201 layers deep)
   - Model Checkpoint Size  : 70.34 MB (PyTorch state_dict)
   - Inference Latency      : 3.4 ms on CUDA GPU / ~80 ms on Intel i7 CPU
   - Image Input Resolution : 224 x 224 x 3 (normalized to ImageNet mean/std)
   - Quality Checks Enforced: Laplacian Blur (>100.0), Brightness, Contrast, Retinal Geometry
   - Automated Test Suite   : 638 tests passing (575 ML unit/integration + 63 backend tests)
   - Security Ceilings      : 30 MB max upload, 60 requests/min rate limit, CSP hardened

3. FRONTEND STACK SUMMARY
   - Framework & Version    : React 19.2.8 + Vite 8.3.4 (Single Page Application)
   - Styling Engine         : Tailwind CSS 4.3.3 (Utility-first, dark clinical theme, <25 KB build)
   - Routing & HTTP Client  : React Router DOM 7.18.4 + Axios 1.20.0 (dynamic base URL resolution)
   - Data Visualization     : Chart.js 4.5.1 on HTML5 Canvas (biometric gauges & sparklines)

4. BACKEND STACK SUMMARY
   - Framework & ASGI Server: FastAPI 0.143.0 + Uvicorn 0.54.0 (asynchronous event loop)
   - Schema Validation      : Pydantic 2.14.0 (Rust-based validation, strict type enforcement)
   - Database & ORM         : SQLite 3.x in WAL mode + SQLAlchemy 2.1.4 ORM (4 relational tables)
   - Report Generation      : ReportLab 5.0.1 (vector PDF compilation in memory via BytesIO)

5. CRITICAL DATA FLOW SUMMARY
   [User Image] -> [React 19] -> [Axios FormData] -> [Uvicorn :8000] -> [Middleware Pipeline]
   -> [FastAPI Route] -> [Image Quality Gate] -> [DenseNet-201 Inference] -> [Grad-CAM Hook at dense_block4]
   -> [Disk & SQLite Save] -> [Pydantic JSON Response] -> [React Reticle HUD & Heatmap Render]

6. THE PART 14 DEFECT & FIX (BE READY TO EXPLAIN THIS)
   - Symptom   : Broken image icon on clicking "Grad-CAM Heatmap" for demo presets.
   - Root Cause: Missing 'gradcamHeatmap' property in api.js demo preset -> 'undefined' Base64 string.
   - Repair    : Generated valid preset PNGs, updated api.js, implemented multi-source resolver with
                 strict string validation in EyeAssessment.jsx, and added diagnostic fallback UI.

7. ARCHITECTURAL SAFEGUARDS
   - CORS Protection        : Restricts API calls to authorized origins.
   - CSP & Security Headers : Blocks unauthorized scripts, clickjacking, and MIME sniffing.
   - Rate Limiter           : 60 req/min sliding-window limiter prevents DoS attacks.
   - Size Limiter           : Rejects payloads >30 MB with HTTP 413 before memory allocation.
   - Pre-Warming            : Model pre-warms on startup, eliminating first-request cold-start latency.
====================================================================================================
```
