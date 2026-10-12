# Developer Handoff Guide: ML/AI Screening Module
**Multimodal Health Screening & Analysis System**  
**Component:** Machine Learning & Artificial Intelligence Core Layer (`ml/`)  
**Target Audience:** Backend Developers, ML Engineers, Integration Architects  

> [!NOTE]
> **Authoritative Release Readiness Report:**  
> For the complete final release-readiness verification, audit evidence table, benchmark metrics, and formal acceptance verdict, please consult the root [HANDOFF.md](../HANDOFF.md).  
> This guide serves as the technical integration specification and API reference for developers integrating the `ml` engine.

---

## 1. Project Scope & Architecture

### Scope Definition
This project currently contains **strictly the ML/AI layer**. There is **no frontend** and **no backend REST API** implemented in this repository. All capabilities are packaged as a self-contained, standalone Python library (`ml`) that can be installed, imported, and executed independently.

### Architecture Overview
The ML system provides multimodal clinical triage by combining computer vision analysis of retinal fundus images with physiological vital-sign assessment.

```
                              [Incoming Screening Request]
                                           |
                   +-----------------------+-----------------------+
                   |                                               |
        [Retinal Fundus Image]                          [Vital Sign Readings]
                   |                                    (HR, SpO2, Temperature)
                   v                                               |
       +-----------------------+                                   v
       | Automated Quality Gate|                         +-------------------+
       |   (ml.quality_gate)   |                         | Physiological     |
       +-----------------------+                         | Triage Engine     |
         | Pass            | Fail                        | (ml.physio_model) |
         v                 v                             +-------------------+
    +------------+   [Reject Early]                                |
    | Standard   |   (HTTP 422 /                                   v
    | Preprocess |    actionable reason)                   [Vitals Risk Status]
    +------------+                                         (LOW, MODERATE, HIGH)
         |                                                         |
         v                                                         |
    +---------------------------+                                  |
    | DenseNet201 Classifier    |                                  |
    | (18.1M params, 4 classes) |                                  |
    +---------------------------+                                  |
         |                                                         |
         +-----------------------+                                 |
         |                       |                                 |
         v                       v                                 |
    [Predictions &          [Grad-CAM Heatmap]                     |
     Confidence]             (features.denseblock4)                |
         |                       |                                 |
         +-----------------------+                                 |
                   |                                               |
                   v                                               v
        [Eye Disease Risk]                               [Vitals Risk]
                   |                                               |
                   +-----------------------+-----------------------+
                                           |
                                           v
                             +---------------------------+
                             | Unified Screening Report  |
                             |  (ml.screening_pipeline)  |
                             |  - Overall Risk Level     |
                             |  - Modality Independence  |
                             |  - Clinical Disclaimer    |
                             +---------------------------+
```

### Decoupling & Modality Independence
- **Eye Disease Screening** and **Vital-Sign Analysis** are completely decoupled.
- If only vital signs are supplied (`image_path=None`), the pipeline returns `status: "partial"`, evaluates vitals, and sets `eye_screening: {"status": "skipped"}`.
- If only a retinal image is supplied, the pipeline evaluates the image and sets `vitals_screening: {"status": "skipped"}`.
- An issue in one modality (e.g., an unreadable fundus photo) never invalidates the physiological triage of valid vital signs.

---

## 2. Repository Structure & Module Responsibilities

| Path | Primary Responsibility | Key Public Symbols |
| :--- | :--- | :--- |
| [`ml/__init__.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/__init__.py) | Package initialization, contract invariants, public exports | `CLASS_NAMES`, `NUM_CLASSES`, `check_image_quality`, `run_screening`, etc. |
| [`ml/eye_model.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/eye_model.py) | DenseNet201 architecture, safe checkpoint loading, inference cache | `build_densenet201_model`, `load_eye_model_checkpoint`, `predict_eye_disease`, `get_model` |
| [`ml/quality_gate.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/quality_gate.py) | Pre-inference image quality gate (blur, lighting, contrast, circularity) | `check_image_quality`, `calculate_laplacian_blur`, `check_illumination_and_contrast` |
| [`ml/preprocessing.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/preprocessing.py) | Input format resolution, resizing ($224 \times 224$), ImageNet normalization | `preprocess_image`, `load_image_safely`, `validate_image_tensor` |
| [`ml/gradcam.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/gradcam.py) | Visual explainability heatmaps from `features.denseblock4` | `generate_gradcam`, `get_target_layer` |
| [`ml/physio_model.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/physio_model.py) | Rule-based clinical triage for Heart Rate, SpO2, and Body Temperature | `analyze_vitals`, `assess_vital_risk` |
| [`ml/screening_pipeline.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/screening_pipeline.py) | Unified standalone screening entry point & CLI (`ml-screening`) | `run_screening`, `main` |
| [`ml/dataset_loader.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/dataset_loader.py) | Dataset split manifest parser and PyTorch Dataset | `EyeDiseaseDataset`, `load_split_manifest` |
| [`ml/data_integrity.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/data_integrity.py) | Cryptographic SHA-256 hash audit and cross-split leakage auditor | `run_data_integrity_audit`, `audit_split_leakage` |
| [`ml/train_eye.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/train_eye.py) | DenseNet201 training pipeline with atomic checkpointing & class weighting | `train_eye_model`, `TrainConfig`, `compute_class_weights` |
| [`ml/evaluate.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/evaluate.py) | Held-out test evaluation, metric calculation, confusion matrix exporter | `evaluate_model_checkpoint`, `calculate_classification_metrics` |
| [`ml/benchmark.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/benchmark.py) | Latency, memory profiling, and batched throughput benchmarks | `run_benchmark_suite` |
| [`ml/monitoring.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/monitoring.py) | Telemetry logging, path sanitization, and PII masking | `log_ml_event`, `measure_ml_latency`, `sanitize_path` |
| [`ml/smoke_test.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/smoke_test.py) | 4-stage reproducible packaging & real inference smoke test | `run_smoke_test` |
| [`ml/weights/eye_model.pth`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/weights/eye_model.pth) | Validated champion checkpoint (70.34 MB, Epoch 6) | Primary production model weights |
| [`ml/dataset/split_manifest.json`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/dataset/split_manifest.json) | Patient-stratified split manifest (5,889 images, 0% leakage) | Ground truth split definition |

---

## 3. Installation & Environment Setup

### Prerequisites
- **Python:** `>= 3.10` (Tested on Python 3.12.3 on Windows 11).
- **Compute Device:**
  - **GPU (Recommended):** NVIDIA GPU with CUDA 12.x support (verified on NVIDIA GeForce RTX 4060 Laptop GPU).
  - **CPU:** Fully supported fallback when CUDA is not present. Device selection is automated (`torch.cuda.is_available()`) or can be forced via `device="cpu"`.

### Installation Steps

1. **Activate Virtual Environment:**
   ```powershell
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install PyTorch (CUDA GPU or CPU fallback):**
   ```bash
   # Option A: NVIDIA GPU with CUDA 12.4
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

   # Option B: CPU-only execution
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install ML Dependencies & Package in Editable Mode:**
   ```bash
   # Install runtime requirements:
   pip install -r requirements.txt

   # (Optional) Install development and testing dependencies:
   pip install -r requirements-dev.txt

   # Register package in editable mode:
   pip install -e . --no-deps
   ```
   *Alternatively, install the fully pinned reproducible lockfile: `pip install -r requirements-lock.txt && pip install -e . --no-deps`.*

4. **Verify Installation:**
   ```bash
   python -c "import ml; print('Package imported successfully! Classes:', ml.CLASS_NAMES)"
   python -m ml.diagnostics
   ```

---

## 4. Dataset Location & Split-Manifest Requirements

### Ground Truth Dataset
- **Raw Image Root:** `C:\Users\Aryan\.gemini\antigravity\scratch\Custom_Eye_Disease_Model\dataset`
- **Canonical Manifest:** [`ml/dataset/split_manifest.json`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/dataset/split_manifest.json)

### Split Distribution (5,889 Images Total)
| Split | Total Count | Cataract | Diabetic Retinopathy | Glaucoma | Normal | Percentage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 4,130 | 975 | 1,078 | 958 | 1,119 | 70.1% |
| **Validation** | 865 | 207 | 218 | 198 | 242 | 14.7% |
| **Held-out Test** | 894 | 210 | 233 | 203 | 248 | 15.2% |
| **Total** | **5,889** | **1,392** | **1,529** | **1,359** | **1,609** | **100.0%** |

### Canonical Class Mapping (0-Indexed)
```python
CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
CLASS_TO_IDX = {"cataract": 0, "diabetic_retinopathy": 1, "glaucoma": 2, "normal": 3}
```

### Data Integrity & Leakage Verification
The dataset was audited by [`ml/data_integrity.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/data_integrity.py):
- **Cross-Split Hash Overlap:** **0** (verified via SHA-256).
- **Patient ID Leakage:** **0** (patient-stratified isolation).
- **Label Contradictions:** **0**.
- **Unassigned Files on Disk:** 4 files exist in raw dataset directories that are excluded from the manifest due to multi-label ambiguity. They do not enter training or evaluation.

---

## 5. Training Commands & Reproducibility Parameters

DenseNet201 training is fully reproducible via [`ml/train_eye.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/train_eye.py).

### Training Execution Command
```bash
python -m ml.train_eye --manifest ml/dataset/split_manifest.json --epochs 6 --batch-size 8 --lr 0.0005 --patience 3 --seed 42
```

### Resuming Training from Recovery Checkpoint
```bash
python -m ml.train_eye --manifest ml/dataset/split_manifest.json --resume --epochs 10 --batch-size 8 --lr 0.0001
```

### Hyperparameters & Reproducibility Configuration
- **Backbone:** Torchvision `DenseNet201` initialized from ImageNet pretrained weights (`densenet201-c1103571.pth`).
- **Classifier Head:** Linear layer: `nn.Linear(in_features=1920, out_features=4)`.
- **Random Seed:** `42` applied across Python `random`, `numpy`, PyTorch `torch.manual_seed()`, and CUDA `torch.cuda.manual_seed_all()`.
- **Optimizer:** `AdamW` ($\text{lr} = 5 \times 10^{-4}$, $\text{weight\_decay} = 1 \times 10^{-4}$).
- **Learning Rate Scheduler:** `ReduceLROnPlateau` ($\text{mode}=\text{'max'}$, $\text{factor}=0.5$, $\text{patience}=2$).
- **Loss Function:** `CrossEntropyLoss` with inverse frequency class weighting normalized to mean 1.0.
- **Batch Size:** 8 (optimized for 8GB VRAM).
- **Checkpointing:** Atomic write to temporary file before rename, preventing corrupt files if interrupted.

### Hardware Nondeterminism Note
> [!NOTE]
> Setting deterministic seeds (`seed=42`, `cudnn.deterministic=True`) ensures reproducible dataset shuffling and parameter initialization. However, atomic operations in cuDNN GPU convolution backward kernels may introduce minute floating-point variances across different GPU architectures.

---

## 6. Evaluation Commands & Verified Metrics

The model evaluation pipeline in [`ml/evaluate.py`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/evaluate.py) strictly evaluates against the **894 held-out test images**. It never tunes parameters on the test set.

### Evaluation Execution Command
```bash
python -m ml.evaluate --checkpoint ml/weights/eye_model.pth --manifest ml/dataset/split_manifest.json --batch-size 16 --device cuda
```

### Verified Held-Out Test Evaluation Results
*Evaluated on NVIDIA GeForce RTX 4060 Laptop GPU in 30.31 seconds:*

- **Total Test Images:** 894
- **Evaluated Images:** 894 (0 failed / 0 excluded)
- **Overall Accuracy:** **87.81%**
- **Macro Precision:** **87.97%**
- **Macro Recall (Sensitivity):** **87.74%**
- **Macro Specificity:** **95.95%**
- **Macro F1-Score:** **87.78%**
- **Weighted F1-Score:** **87.91%**

#### Per-Class Performance Breakdown
| Class Name | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cataract** | 90.00% | 85.71% | 97.08% | **87.80%** | 210 |
| **Diabetic Retinopathy** | 97.27% | 91.85% | 99.09% | **94.48%** | 233 |
| **Glaucoma** | 79.28% | 86.70% | 93.34% | **82.82%** | 203 |
| **Normal** | 85.32% | 86.69% | 94.27% | **86.00%** | 248 |

#### 4×4 Confusion Matrix
*Rows represent True Ground Truth labels; Columns represent Model Predictions:*

$$\begin{pmatrix} & \textbf{Cataract} & \textbf{Diab. Ret.} & \textbf{Glaucoma} & \textbf{Normal} \\
\textbf{Cataract} & 180 & 2 & 15 & 13 \\
\textbf{Diab. Ret.} & 8 & 214 & 2 & 9 \\
\textbf{Glaucoma} & 8 & 4 & 176 & 15 \\
\textbf{Normal} & 4 & 0 & 29 & 215 \end{pmatrix}$$

#### Inference Latency Metrics (Steady State)
- **Mean Latency:** 3.60 ms per image
- **Median Latency:** 2.77 ms per image
- **95th Percentile:** 5.87 ms per image
- **Peak Batched Throughput ($B=16$):** 404.3 images / second

---

## 7. Model Checkpoint Specifications

- **File Path:** [`ml/weights/eye_model.pth`](file:///C:/Users/Aryan/.gemini/antigravity/scratch/multimodal-health-screening/ml/weights/eye_model.pth)
- **File Size:** **70.34 MB** (73,760,256 bytes)
- **Architecture:** PyTorch DenseNet201 (18,100,612 parameters)
- **Training Epoch:** 6
- **Selection Criteria:** Champion model selected based on highest validation accuracy (**90.29%** on validation split, achieving **87.81%** on held-out test split).
- **Deserialization Security:** `torch.load` enforces `weights_only=True` to block arbitrary Python object execution exploits. Architecture metadata and 4-class output head are strictly verified upon loading.

---

## 8. Public API Signatures & Output Contracts

### 1. `ml.run_screening` (Unified Screening Pipeline)
```python
def run_screening(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    heart_rate: Optional[float] = None,
    spo2: Optional[float] = None,
    temperature: Optional[float] = None,
    weights_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    include_gradcam: bool = True,
    use_cache: bool = True,
    request_id: Optional[str] = None,
) -> Dict[str, Any]
```

#### Successful JSON Response Schema (Both Modalities Present)
```json
{
  "request_id": "c1f7a188-df05-4c91-9c88-e565ad0c3132",
  "status": "success",
  "timestamp_utc": "2026-10-11T18:43:13.123456+00:00",
  "latency_ms": 594.02,
  "eye_screening": {
    "status": "success",
    "prediction": "glaucoma",
    "class_index": 2,
    "confidence": 0.9286,
    "risk_level": "HIGH",
    "all_probabilities": {
      "cataract": 0.0039,
      "diabetic_retinopathy": 0.0015,
      "glaucoma": 0.9286,
      "normal": 0.0660
    },
    "quality_checks": {
      "overall_pass": true,
      "failed_check": null,
      "metrics": {
        "blur_score": 146.42,
        "brightness_mean": 88.51,
        "contrast_std": 52.13,
        "width": 512,
        "height": 512,
        "circularity": 0.88
      }
    },
    "gradcam_image": "iVBORw0KGgoAAAANSUhEUgAA..."
  },
  "vitals_screening": {
    "status": "success",
    "heart_rate": 75.0,
    "spo2": 98.0,
    "temperature": 36.8,
    "overall_status": "Normal",
    "risk_level": "LOW",
    "is_abnormal": false,
    "summary": "All vital signs are within normal clinical reference ranges.",
    "hr_status": "Normal",
    "spo2_status": "Normal",
    "temp_status": "Normal"
  },
  "overall_risk_level": "HIGH",
  "disclaimer": "This system is an automated multimodal health screening prototype..."
}
```

#### Quality Gate Rejection Schema
If the fundus photo is excessively blurry or improperly illuminated:
```json
{
  "request_id": "7fca5e4c-fc20-411a-934d-c10429fc0843",
  "status": "partial",
  "eye_screening": {
    "status": "rejected",
    "reason": "Excessive image blur (blur score 13.98 < threshold 100.0).",
    "quality_checks": {
      "overall_pass": false,
      "failed_check": "blur",
      "metrics": { "blur_score": 13.98, "brightness_mean": 43.2, "contrast_std": 43.34 }
    }
  },
  "vitals_screening": {
    "status": "success",
    "overall_status": "Normal",
    "risk_level": "LOW"
  },
  "overall_risk_level": "LOW"
}
```

### 2. `ml.check_image_quality`
```python
def check_image_quality(
    image_input: Union[str, bytes, np.ndarray, Image.Image],
    min_blur: float = 100.0,
    min_brightness: float = 40.0,
    max_brightness: float = 220.0,
    min_contrast: float = 30.0,
    min_resolution: Tuple[int, int] = (200, 200),
    min_circularity: float = 0.50,
) -> Dict[str, Any]
```

### 3. `ml.analyze_vitals`
```python
def analyze_vitals(
    heart_rate: float,
    spo2: float,
    temperature: float,
    raise_errors: bool = True,
) -> Dict[str, Any]
```
- **Plausible Clinical Boundaries:**
  - Heart Rate: $20.0 \le \text{HR} \le 300.0$ BPM
  - Blood Oxygen: $0.0 \le \text{SpO}_2 \le 100.0$ %
  - Body Temperature: $25.0 \le \text{Temp} \le 45.0$ °C
- **Type Checking:** Python booleans (`True`, `False`) are strictly rejected (`isinstance(v, (bool, np.bool_))`) to prevent boolean-as-integer leakage.

### 4. `ml.generate_gradcam`
```python
def generate_gradcam(
    image_path: str,
    predicted_class_idx: int,
    model: Optional[nn.Module] = None,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    target_layer: Optional[Union[str, nn.Module]] = None,
    alpha: float = 0.4,
) -> str
```
- Returns a Base64-encoded PNG string.
- Hooks are managed via `try ... finally` blocks to prevent memory leaks.
- Target layer defaults to `features.denseblock4`.

---

## 9. CLI Usage Examples

The package installs a command-line script `ml-screening`:

```bash
# 1. Full multimodal screening:
ml-screening --image "path/to/retina.jpg" --hr 75 --spo2 98 --temp 36.8

# 2. Retinal fundus image only:
ml-screening --image "path/to/retina.jpg"

# 3. Vital signs only:
ml-screening --hr 115 --spo2 91 --temp 38.6

# 4. Save structured JSON result to disk:
ml-screening --image "path/to/retina.jpg" --hr 75 --spo2 98 --temp 36.8 --output report.json
```

---

## 10. Automated Test Commands & Verification Results

### Summary of Automated Test Runs
All tests have been executed and verified in the active environment:

| Test Scope | Command | Passed | Failed | Total Time | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Reproducible Smoke Test** | `python ml/smoke_test.py` | 4/4 Stages | 0 | 4.8s | **PASS** |
| **Pipeline Interface Suite** | `pytest ml/tests/test_screening_pipeline.py` | 19 | 0 | 16.14s | **PASS** |
| **Complete Unit & Integration Suite** | `pytest ml/tests/` | 306 | 0 | 150.94s | **PASS** |
| **Full Acceptance Suite** | `python -m ml.acceptance_test` | 9/9 Checks | 0 | 32.5s | **PASS** |

### Distinguishing Mocks vs. Real Inference Tests
- **Unit Tests (`ml/tests/`):** Use synthetic test tensors, synthetic color ramps, and temporary directories to verify edge cases, type errors, boundary conditions, and isolation without requiring GPU overhead.
- **Integration & Smoke Tests (`ml/smoke_test.py`, `ml.acceptance_test`, `ml.evaluate`):** Load the **real 70.34 MB checkpoint weights** onto CUDA, process **real patient retinal images**, execute actual backward passes for Grad-CAM, and compute real classification metrics against ground truth.

---

## 11. Known Limitations & Outstanding Work

1. **Dataset Origin & External Generalization:**
   - The current dataset (5,889 images) is drawn from combined Kaggle and RFMiD fundus collections. While patient-stratified and leakage-free, it has not yet been validated on external multi-center clinical cohorts.
2. **Quality Gate Threshold on Low-Resolution Capture:**
   - The blur threshold (Laplacian variance 100.0) correctly rejects out-of-focus or motion-blurred captures. Highly compressed legacy web images (e.g., low-resolution RFMiD thumbnails with blur scores ~14.0) will be rejected by design. Upstream hardware must supply clean fundus captures ($>512 \times 512$).
3. **Concurrency & Memory Management in Multi-Worker Backends:**
   - In-memory model caching uses threading locks (`threading.Lock()`), making it thread-safe within a single process.
   - If deploying with multi-process web servers (e.g., Gunicorn with multiple worker processes), each worker process will initialize its own DenseNet201 instance in VRAM (~70 MB weights + ~400 MB PyTorch context). For high-concurrency production deployments, use a dedicated model worker pool or an async single-process FastAPI service.

---

## 12. Instructions for Future Backend Developers (FastAPI Integration)

A backend developer integrating this package into a FastAPI or Flask microservice should follow these implementation steps:

### Example FastAPI Service Implementation
```python
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import ml

app = FastAPI(title="Multimodal Health Screening API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    # Warm up model cache in memory during startup (takes ~700 ms once)
    ml.get_cached_model(device="cuda")

@app.post("/api/v1/screen")
async def screen_patient(
    image: UploadFile = File(None),
    heart_rate: float = Form(None),
    spo2: float = Form(None),
    temperature: float = Form(None),
):
    image_bytes = None
    if image is not None:
        image_bytes = await image.read()

    result = ml.run_screening(
        image_bytes=image_bytes,
        heart_rate=heart_rate,
        spo2=spo2,
        temperature=temperature,
        include_gradcam=True,
        use_cache=True,
    )

    # Return HTTP 422 if an uploaded image is explicitly rejected by the quality gate
    if result.get("eye_screening", {}).get("status") == "rejected":
        return JSONResponse(
            status_code=422,
            content=result
        )

    return result
```

### Rendering Grad-CAM in the Frontend
When `include_gradcam=True`, the response contains a Base64 string at `result["eye_screening"]["gradcam_image"]`. The frontend can render it directly:
```html
<img src="data:image/png;base64,{{ result.eye_screening.gradcam_image }}" alt="Grad-CAM Activation Map" />
```

---

## 13. Clinical Decision-Support Notice

> [!WARNING]
> **Clinical Research & Decision Support Notice:**  
> This software is an automated multimodal health screening prototype engineered strictly for clinical triage research and decision support. It does **not** provide definitive autonomous medical diagnoses. All screening outputs, visual activation maps, and vital-sign classifications must be reviewed and interpreted by qualified healthcare professionals prior to making clinical decisions.
