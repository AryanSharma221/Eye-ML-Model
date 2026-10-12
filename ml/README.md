# Multimodal Health Screening ML Package
**Machine Learning Core: Retinal Fundus AI & Physiological Triage**  
*DenseNet201 Four-Class Eye Disease Classifier, Grad-CAM Explainability, and Physiological Vital-Sign Analysis*

---

## 1. Package Overview & Architecture

The `ml` package is a self-contained, production-grade Python package delivering the core AI/ML capabilities for multimodal health screening:

1. **Automated Retinal Quality Gate (`ml.quality_gate`):**  
   Validates incoming fundus images across blur (Laplacian variance), illumination (brightness mean), dynamic range (contrast standard deviation), spatial resolution, and retinal circular contour prior to neural inference.
2. **Standardized Preprocessing Pipeline (`ml.preprocessing`):**  
   Converts raw image inputs (filepaths, PIL Images, NumPy arrays, or raw byte streams) into standardized ImageNet-normalized PyTorch tensors ($3 \times 224 \times 224$).
3. **Deep Learning Retinal Classifier (`ml.eye_model`):**  
   PyTorch `DenseNet201` architecture trained on 5,889 patient-stratified retinal images across 4 canonical classes:
   `["cataract", "diabetic_retinopathy", "glaucoma", "normal"]`.
4. **Visual Explainability via Grad-CAM (`ml.gradcam`):**  
   Computes gradient-weighted class activation maps from the final convolutional block (`features.denseblock4`), superimposes them onto the native fundus image, and produces Base64-encoded PNG heatmaps.
5. **Physiological Vital Signs Engine (`ml.physio_model`):**  
   Rule-based clinical triage engine for Heart Rate (BPM), Blood Oxygen Saturation ($\text{SpO}_2$ %), and Body Temperature (°C), calculating independent and composite risk levels (`LOW`, `MEDIUM`, `HIGH`).
6. **Unified Screening Pipeline (`ml.screening_pipeline`):**  
   Standalone Python entry point coordinating quality gating, deep inference, visual explainability, and vital-signs triage into a consistent screening contract.
7. **Zero-Overhead Lazy Loading:**  
   Importing `ml` (`import ml`) takes ~2.7s and initializes **0** models in memory. Weights are only loaded when explicitly called or during screening inference.

---

## 2. Environment & System Requirements

- **Python Version:** `>= 3.10` (verified on Python 3.12).
- **Hardware Support:**
  - **CUDA (Recommended):** NVIDIA GPU with CUDA 12.x support (e.g., NVIDIA GeForce RTX 4060 Laptop GPU).
  - **CPU:** Fully supported fallback when CUDA is not present or when `device="cpu"` is specified.
- **Key Runtime Dependencies:**
  - `torch>=2.4.0`
  - `torchvision>=0.19.0`
  - `numpy>=1.26.0`
  - `pillow>=10.2.0`
  - `opencv-python>=4.9.0`
  - `scikit-learn>=1.4.0`
- **Development & Testing Dependencies:**
  - `pytest>=8.0.0`

---

## 3. Installation & Developer Setup Guide

### 3.1 Prerequisites
- **Python:** Python 3.10 to 3.12 (verified on Python 3.12.4).
- **Operating System:** Windows 10/11 x86_64, Linux x86_64, or macOS (CPU only).
- **GPU (Optional):** NVIDIA GPU supporting CUDA 12.x (e.g. RTX 4060) or CPU fallback.

### 3.2 Create & Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On Linux / macOS (Bash):**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3.3 Install Dependencies

Choose either GPU acceleration or CPU-only runtime:

#### Option A: NVIDIA GPU Acceleration (CUDA 12.4 - Recommended)
```bash
# 1. Install PyTorch with CUDA 12.4 wheels:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Install ML runtime requirements:
pip install -r requirements.txt

# 3. (Optional) Install development and testing dependencies:
pip install -r requirements-dev.txt

# 4. Register package in editable mode without reinstalling dependencies:
pip install -e . --no-deps
```

#### Option B: CPU-Only Execution
```bash
# 1. Install CPU-optimized PyTorch wheels:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install ML runtime requirements:
pip install -r requirements.txt

# 3. (Optional) Install development and testing dependencies:
pip install -r requirements-dev.txt

# 4. Register package in editable mode:
pip install -e . --no-deps
```

#### Option C: Deterministic Environment Reproduction (Lockfile)
To reproduce the exact verified environment down to patch versions:
```bash
# Install exact pinned lockfile:
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

### 3.4 Configuration Instructions
All pipeline parameters are managed via `ml/config.py` with zero hardcoded machine-specific paths. You can optionally override defaults via environment variables:

| Environment Variable | Default Path / Value | Purpose |
| :--- | :--- | :--- |
| `ML_CHECKPOINT_PATH` | `ml/weights/eye_model.pth` | Active DenseNet201 checkpoint binary |
| `ML_DATASET_DIR` | `ml/dataset/` | Root directory containing retinal images |
| `ML_MANIFEST_PATH` | `ml/dataset/split_manifest.json` | 5,889 image split manifest |
| `ML_DEVICE` | `auto` | Execution target: `auto`, `cuda`, or `cpu` |
| `ML_LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `ML_EVAL_DIR` | `ml/evaluation_results/` | Output directory for evaluation reports |
| `ML_DEMO_DIR` | `ml/demo_results/` | Output directory for demonstration artifacts |

### 3.5 Checkpoint & Dataset Requirements
- **Inference Only:** Requires only `ml/weights/eye_model.pth`. Does **not** require raw training images or manifest.
- **Training & Evaluation:** Requires `ml/dataset/split_manifest.json` and image directories in `ml/dataset/`.

### 3.6 First-Run Diagnostic Command
Run environment diagnostics to verify the setup in seconds:
```bash
# Lightweight general diagnostic (checks Python, dependencies, imports, config):
ml-diagnostics
# or: python -m ml.diagnostics

# Strict validation requiring checkpoint for real inference:
ml-diagnostics --check-inference

# Full validation requiring dataset & manifest for training:
ml-diagnostics --check-dataset
```

---

## 4. Package Import Hierarchy & Public Interface

All core functionality is exported directly from the top-level `ml` namespace:

```python
import ml

# Canonical Class Configuration
print(ml.CLASS_NAMES)  # ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
print(ml.NUM_CLASSES)   # 4

# Core Public Functions:
# - ml.check_image_quality
# - ml.preprocess_image
# - ml.predict_eye_disease
# - ml.generate_gradcam
# - ml.analyze_vitals
# - ml.run_screening
# - ml.build_densenet201_model
# - ml.load_eye_model_checkpoint
# - ml.get_model
# - ml.get_cached_model
```

### Clean Import Guarantees
- The package can be imported from the repository root, from any subfolder, or when installed in a virtual environment via `pip install -e .`.
- No absolute machine paths are hardcoded; all default artifact locations resolve dynamically relative to `ml/`.

---

## 5. Dataset & Trained Checkpoint Specifications

### Checkpoint Storage & Performance
- **Primary Checkpoint:** `ml/weights/eye_model.pth` (70.34 MB)
- **Model Architecture:** DenseNet201 (18,100,612 parameters)
- **Validation Accuracy:** **90.29%**
- **Held-Out Test Accuracy:** **87.81%** (evaluated across 894 held-out test images)
- **Automatic Checkpoint Discovery:** `ml.eye_model.DEFAULT_CHECKPOINT_PATH` points directly to `ml/weights/eye_model.pth`.

### Dataset & Split Manifest
- **Manifest Location:** `ml/dataset/split_manifest.json`
- **Total Images:** 5,889 patient-stratified retinal images.
- **Split Distribution:**
  - Training: 4,117 images (70%)
  - Validation: 878 images (15%)
  - Held-out Test: 894 images (15%)
- **Data Leakage:** 0% cross-split leakage verified via cryptographic SHA-256 auditing.
- **Target Classes (Canonical 4-Class Order):**
  1. `cataract` (Index 0)
  2. `diabetic_retinopathy` (Index 1)
  3. `glaucoma` (Index 2)
  4. `normal` (Index 3)

---

## 6. Public API Reference

### 1. `check_image_quality`
Validates whether a retinal fundus image meets clinical quality criteria.

```python
def check_image_quality(
    image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
    min_blur: float = 100.0,
    min_brightness: float = 40.0,
    max_brightness: float = 220.0,
    min_contrast: float = 30.0,
    min_resolution: Tuple[int, int] = (200, 200),
    min_circularity: float = 0.50,
) -> Dict[str, Any]
```
- **Returns:**
  ```python
  {
      "overall_pass": True,  # or False
      "failed_check": None,  # or "blur", "brightness", "contrast", etc.
      "reason": "All quality criteria satisfied.",
      "metrics": {
          "blur_score": 146.42,
          "brightness_mean": 88.51,
          "contrast_std": 52.13,
          "width": 512,
          "height": 512,
          "circularity": 0.88
      }
  }
  ```

### 2. `preprocess_image`
Standardizes an input image into a normalized PyTorch tensor.

```python
def preprocess_image(
    image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor
```
- **Returns:** 4D FloatTensor of shape `(1, 3, 224, 224)` ready for DenseNet201 forward pass.

### 3. `predict_eye_disease`
Runs inference on a preprocessed image tensor.

```python
def predict_eye_disease(
    tensor: torch.Tensor,
    model: Optional[torch.nn.Module] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]
```
- **Returns:**
  ```python
  {
      "prediction": "glaucoma",
      "class_index": 2,
      "confidence": 0.9286,
      "all_probabilities": {
          "cataract": 0.0125,
          "diabetic_retinopathy": 0.0341,
          "glaucoma": 0.9286,
          "normal": 0.0248
      },
      "risk_level": "HIGH"
  }
  ```

### 4. `generate_gradcam`
Generates a gradient-weighted class activation map visualization.

```python
def generate_gradcam(
    image_path: Union[str, Path],
    predicted_class_idx: int,
    model: Optional[torch.nn.Module] = None,
    target_layer_name: str = "features.denseblock4",
    device: Optional[str] = None,
) -> str
```
- **Returns:** Base64-encoded PNG image string containing the jet heatmap blended over the original image.

### 5. `analyze_vitals`
Performs physiological rule-based triage on vital sign inputs.

```python
def analyze_vitals(
    heart_rate: Optional[float] = None,
    spo2: Optional[float] = None,
    temperature: Optional[float] = None,
    raise_errors: bool = True,
) -> Dict[str, Any]
```
- **Parameters:**
  - `heart_rate`: Heart rate in beats per minute (plausible: 20–300 BPM).
  - `spo2`: Blood oxygen saturation percentage (plausible: 0–100%).
  - `temperature`: Core body temperature in °C (plausible: 25–45 °C).
  - `raise_errors`: If `True`, raises `ValueError` on impossible/invalid values; if `False`, returns structured rejection.
- **Returns:**
  ```python
  {
      "heart_rate": {"value": 72.0, "status": "Normal", "unit": "BPM"},
      "spo2": {"value": 98.0, "status": "Normal", "unit": "%"},
      "temperature": {"value": 36.8, "status": "Normal", "unit": "°C"},
      "hr_status": "Normal",
      "spo2_status": "Normal",
      "temp_status": "Normal",
      "overall_status": "Normal",
      "risk_level": "LOW",
      "is_abnormal": False,
      "summary": "All recorded vital signs are within normal ranges.",
      "disclaimer": "Preliminary screening assessment..."
  }
  ```

### 6. `run_screening`
Canonical end-to-end multimodal health screening pipeline function.

```python
def run_screening(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    heart_rate: Optional[float] = None,
    spo2: Optional[float] = None,
    temperature: Optional[float] = None,
    include_gradcam: bool = True,
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]
```
- Coordinates modality inputs independently: if only vitals or only image is provided, processes available modalities gracefully.
- Rejects unsuitable images with clear actionable feedback without throwing.
- Combines eye disease findings and vital status into a unified `overall_risk_level` (`LOW`, `MEDIUM`, `HIGH`).

---

## 7. Error Handling & Safety Guardrails

The package enforces strict enterprise security and medical data safety guardrails:

1. **Safe Model Deserialization:**
   - Enforces `weights_only=True` on all `torch.load` operations.
   - Validates architecture tag (`DenseNet201`), number of output logits ($4$), and canonical class names.
   - Blocks arbitrary Python pickle code execution exploits.
2. **Decompression Bomb & Denial of Service Protection:**
   - Maximum image file size: 30 MB (`MAX_IMAGE_FILE_SIZE_BYTES = 30 * 1024 * 1024`).
   - Dimension ceiling: 5000 × 5000 pixels.
   - Guarded against PIL `Image.DecompressionBombError`.
3. **Format & Input Validation:**
   - Allowed image extensions: `.jpg`, `.jpeg`, `.png`.
   - Truncated or corrupt files rejected safely.
4. **Physiological Type Safety:**
   - Python boolean values (`True`/`False`) are explicitly rejected (`isinstance(val, (bool, np.bool_))`) to prevent boolean-as-integer leakage (`True == 1`).
   - Non-numeric strings, `None`, `NaN`, and `Inf` are caught and rejected.
   - Physiological plausibility ranges: HR (20–300), SpO2 (0–100), Temp (25–45).
5. **Grad-CAM Resource Management:**
   - Model forward and backward hooks are safely managed via `try ... finally` blocks to ensure hooks are removed even on runtime exceptions.
   - Gradients are zeroed (`model.zero_grad()`) before and after Grad-CAM generation.

---

## 8. Standalone CLI Tools

The package provides standalone command-line tools for screening, demonstration, and model management:

### A. Demonstration Runner (`ml.demo`)
Interactive runner supporting live real inference and safe contract validation:
```bash
# Pipeline validation mode (fast dry-run verifying all schemas & rules):
python -m ml.demo --mode validate

# Real inference on held-out test sample with Grad-CAM heatmap generation:
python -m ml.demo --mode real --sample --sample-class normal

# Real inference with diabetic retinopathy sample and elevated vitals:
python -m ml.demo --mode real --sample --sample-class diabetic_retinopathy --hr 105 --spo2 94 --temp 38.2

# Multi-sample 4-class evaluation on representative held-out images:
python -m ml.demo --batch

# Using installed CLI:
ml-demo --mode real --sample --sample-class glaucoma
```

### B. Screening Pipeline CLI (`ml.screening_pipeline`)
Production command-line interface for multimodal health screening:
```bash
# Full multimodal screening:
ml-screening --image path/to/retina.jpg --hr 75 --spo2 98 --temp 36.8

# Retinal image only:
ml-screening --image path/to/retina.jpg

# Vital signs only:
ml-screening --hr 115 --spo2 91 --temp 38.6

# Save output JSON to file:
ml-screening --image path/to/retina.jpg --output screening_report.json
```

### C. Checkpoint Selection & Lifecycle Manager (`ml.checkpoint_manager`)
Safe, auditable checkpoint inspection, promotion, and reversible rollback:
```bash
# View active champion checkpoint status and provenance:
ml-checkpoint status

# List and audit all candidate checkpoints in ml/weights/:
ml-checkpoint list

# Full 7-point eligibility check on candidate weights:
ml-checkpoint check ml/weights/eye_candidate_model.pth

# Compare candidates with clinical sensitivity regression warning:
ml-checkpoint compare ml/weights/eye_model.pth ml/weights/eye_candidate_model.pth

# Reversible rollback to previous verified champion:
ml-checkpoint rollback --reason "Revert model"
```

### D. Environment Diagnostics (`ml.diagnostics`)
```bash
# Run comprehensive PyTorch, CUDA, and path diagnostics:
ml-diagnostics
```

---

## 9. Automated Testing & Smoke Test

### Reproducible 4-Stage Smoke Test
To verify package imports, configuration discovery, checkpoint presence, and real end-to-end inference in a single command:
```bash
python ml/smoke_test.py
```
**Verification Stages:**
- **Stage 1:** Package imports & public function discovery (~2.7s).
- **Stage 2:** Configuration invariants & canonical 4-class order.
- **Stage 3:** Checkpoint verification & model loading on CUDA/CPU.
- **Stage 4:** Real end-to-end inference against a held-out test image with Grad-CAM and vitals triage.

### Full Test Suite Execution
Run all 300+ automated unit and integration tests across the package:
```bash
pytest ml/tests/ -v
```

### Focused Component Tests
```bash
# Screening pipeline tests:
pytest ml/tests/test_screening_pipeline.py -v

# Quality gate tests:
pytest ml/tests/test_quality_gate.py -v

# Grad-CAM explainability tests:
pytest ml/tests/test_gradcam.py -v

# Physiological triage tests:
pytest ml/tests/test_physio_model.py -v
```

---

## 10. Training & Evaluation Reproduction

### Training DenseNet201
```bash
python -m ml.train_eye --manifest ml/dataset/split_manifest.json --epochs 6 --batch-size 8 --lr 0.0005 --patience 3 --seed 42
```

### Checkpoint Evaluation
```bash
python -m ml.evaluate --checkpoint ml/weights/eye_model.pth --manifest ml/dataset/split_manifest.json --batch-size 16 --device cuda
```

### Dataset Integrity & Leakage Audit
```bash
python -m ml.data_integrity
```

---

## 11. Complete Runnable Code Example

Here is a self-contained Python script demonstrating how another developer can import and consume the package:

```python
import json
import ml

def main():
    print(f"ML Package Initialized.")
    print(f"Target Classes: {ml.CLASS_NAMES}")

    image_path = "ml/dataset/test/sample.jpg"  # Or your fundus image path

    # Step 1: Pre-inference Image Quality Gate
    quality_result = ml.check_image_quality(image_path)
    if not quality_result["overall_pass"]:
        print(f"Image rejected: {quality_result['reason']}")
        return

    print("Image passed quality gate.")

    # Step 2: Full Multimodal Screening
    screening = ml.run_screening(
        image_path=image_path,
        heart_rate=76.0,
        spo2=98.5,
        temperature=36.7,
        include_gradcam=True,
    )

    print("\n--- Screening Summary ---")
    print(f"Status:              {screening['status']}")
    print(f"Overall Risk Level:  {screening['overall_risk_level']}")
    
    if screening.get("eye_screening"):
        eye = screening["eye_screening"]
        print(f"Eye Prediction:      {eye['prediction'].upper()} ({eye['confidence']:.2%})")
        print(f"Grad-CAM Heatmap:    {'Attached (Base64 PNG)' if eye.get('gradcam_image') else 'None'}")
        
    if screening.get("vitals_screening"):
        vitals = screening["vitals_screening"]
        print(f"Vitals Triage:       {vitals['overall_status']} (Risk: {vitals['risk_level']})")

    print("\nDisclaimer:", screening["disclaimer"])

if __name__ == "__main__":
    main()
```

---

## 12. Known Limitations & Clinical Prototype Disclaimer

### 12.1 Engineering & Operational Limitations
1. **Clinical Screening Scope Only:** This software is an automated multimodal health screening prototype engineered strictly for clinical decision support. It does **not** provide definitive autonomous medical diagnoses.
2. **Single-Field Retinal Photography:** The DenseNet201 model is trained on single-field fundus photographs ($3 \times 224 \times 224$); it does not perform multi-field panoramic retinal montage synthesis or optical coherence tomography (OCT) volumetric analysis.
3. **Class-Conditional Sensitivity Variance:** On the held-out test split ($N=894$), sensitivity is highest on Diabetic Retinopathy ($91.85\%$ recall, $94.48\%$ F1) and Cataract ($85.71\%$ recall, $87.80\%$ F1), while Glaucoma recall is $86.70\%$ ($79.28\%$ precision). Glaucoma predictions should be carefully correlated with intraocular pressure and cup-to-disc ratio assessments.
4. **Standalone Python ML Boundary:** There is no web server, backend REST API, or frontend dashboard in this repository. All execution is local and programmatic via the `ml` package and CLI entry points.

> [!WARNING]
> **Prototype Screening & Decision-Support Notice:**  
> This software is an automated multimodal health screening prototype intended exclusively for triage research and clinical decision support. It does **not** provide definitive medical or ophthalmic diagnoses. All screening outputs, visual activation maps, and vital-sign triage flags must be reviewed by qualified healthcare professionals prior to clinical action.

---

## 13. Troubleshooting Guide for Common Setup Failures

| Issue / Error Message | Root Cause | Verified Resolution |
| :--- | :--- | :--- |
| **`CheckpointNotFoundError: Production checkpoint file not found at ...`** | Checkpoint weights file (`eye_model.pth`) is absent from `ml/weights/`. | Copy the trained `eye_model.pth` file into `ml/weights/` or set `$env:ML_CHECKPOINT_PATH="path/to/model.pth"`. For dry-run testing without weights, run `ml-demo --mode validate`. |
| **`DeviceUnavailableError: CUDA execution requested, but CUDA is unavailable`** | `--device cuda` was explicitly passed on a system without an active NVIDIA GPU or CUDA runtime. | Set `--device cpu` or set `$env:ML_DEVICE="cpu"`. In `auto` mode, the pipeline automatically detects and uses CPU. |
| **`torch.cuda.OutOfMemoryError` during batch inference/evaluation** | GPU VRAM capacity exceeded by large batch size. | Lower batch size: `python -m ml.evaluate --batch-size 8` or run on CPU via `--device cpu`. |
| **`QualityGateRejectionError` / Image Rejected** | Retinal photograph failed blur, illumination, or circularity checks. | Inspect the rejection reason in the returned JSON. Ensure image has Laplacian variance $\ge 100.0$, brightness $\in [30, 225]$, and contains a detectable circular retinal boundary. |
| **`ImportError: DLL load failed while importing cv2`** | Missing Microsoft Visual C++ Redistributable on Windows. | Download and install the [Visual C++ 2015–2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe). |
| **Windows Console `charmap` Encoding Errors** | Windows terminal code page CP1252 attempting to print non-ASCII characters. | All project CLI outputs use pure ASCII (`deg C`, `--`). If running custom wrappers, set `$env:PYTHONUTF8="1"` in PowerShell or `set PYTHONUTF8=1` in CMD. |
| **`Manifest file not found: split_manifest.json`** | Split manifest missing when running training or full dataset evaluation. | Standard inference and smoke testing do not require the manifest. For dataset operations, place `split_manifest.json` in `ml/dataset/` or set `$env:ML_MANIFEST_PATH`. |
