# ML Pipeline Cross-Environment & Reproducibility Report

**Generated:** 2026-10-12T08:09:57Z  
**Status:** **VERIFIED**  
**Role:** Senior ML Deployment & Reliability Engineer  

---

### 1. Host Environment & Checkpoint Specification

| Property | Value |
| :--- | :--- |
| **Operating System** | Windows-11-10.0.26200-SP0 (AMD64) |
| **Python Runtime** | Python `3.12.4` (MSC v.1940 64 bit (AMD64)) |
| **PyTorch & Torchvision** | PyTorch `2.6.0+cu124`, Torchvision `0.21.0+cu124` |
| **OpenCV & Pillow** | OpenCV `5.0.0`, Pillow `12.3.0` |
| **NumPy & scikit-learn**| NumPy `2.5.4`, scikit-learn `1.9.1` |
| **Hardware Acceleration** | **CUDA:** NVIDIA GeForce RTX 4060 Laptop GPU (Compute Capability [8, 9], cuDNN 90100) |
| **Checkpoint Path** | `eye_model.pth` |
| **Checkpoint Size & Hash** | `73,754,057` bytes \| SHA-256: `e819440c35f166c90e6469b071cc83136857e95aa80bcd6697336894bb459f5f` |
| **Canonical Class Order** | `('cataract', 'diabetic_retinopathy', 'glaucoma', 'normal')` |

---

### 2. Reproducible Held-Out Test Cohort

The reproducibility cohort was extracted from the authentic held-out test split (`split_manifest.json`).  
All 40 samples passed the automated image quality gate (blur, illumination, contrast, resolution $\ge 224 	imes 224$):

- **Sample Count:** 40 images (10 samples per canonical class)
- **Class Balance:**
  - `cataract`: 10 samples
  - `diabetic_retinopathy`: 10 samples
  - `glaucoma`: 10 samples
  - `normal`: 10 samples

---

### 3. Execution Mode Benchmarks & Latency Profiling

Model initialization latency (weights hydration from disk to device memory) is cleanly separated from steady-state per-image inference:

| Metric | CPU Execution | CUDA Execution (RTX 4060) | Speedup / Delta |
| :--- | :---: | :---: | :---: |
| **Model Init Latency** | `370.53 ms` | `436.21 ms` | 0.8x |
| **Mean Inference Latency** | `74.41 ms` | `37.39 ms` | **2.0x faster on GPU** |
| **Median Inference Latency** | `73.71 ms` | `34.66 ms` | — |
| **P95 Inference Latency** | `78.38 ms` | `47.38 ms` | — |
| **Cohort Accuracy** | `87.5%` | `87.5%` | **100% Identical Accuracy** |

---

### 4. Numerical Consistency & Device Parity Analysis

| Comparison Scenario | Prediction Agreement | Max Prob Delta | Mean Prob Delta | Mean Conf Delta | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **cpu_vs_cuda** | **100.0%** (40/40) | `0.0002` | `2.6e-05` | `3.8e-05` | **PASSED** |
| **cuda_vs_cuda_repeat** | **100.0%** (40/40) | `0.0` | `0.0` | `0.0` | **PASSED** |
| **in_process_cpu_vs_fresh_process_cpu** | **100.0%** (40/40) | `0.0` | `0.0` | `0.0` | **PASSED** |
| **in_process_cuda_vs_fresh_process_cuda** | **100.0%** (40/40) | `0.0` | `0.0` | `0.0` | **PASSED** |

#### Numerical Invariants & Tolerance Criteria:
1. **CPU vs. CUDA Floating-Point Variance:** Convolutional reduction trees in cuDNN vs. CPU OpenMP BLAS differ in floating-point summation ordering. The observed max probability delta is `0.0002`, well within the documented numerical tolerance threshold ($< 0.005$). Prediction agreement is **100.0%**.
2. **Same-Process Determinism:** Repeated inference on identical inputs in the same process yields `0.0` maximum probability difference and 100% prediction agreement.
3. **Process Isolation Invariance:** Running inference in a completely fresh, isolated Python interpreter reproduces identical predictions and logits ($< 10^{-4}$ delta).

---

### 5. Configuration & Artifact Portability Verification

| Portability Criterion | Test Evaluation | Result |
| :--- | :--- | :---: |
| **Explicit Device Selection (`cpu`)** | `resolve_device('cpu')` yields `torch.device('cpu')` | **VERIFIED** |
| **Explicit Device Selection (`cuda`)** | `resolve_device('cuda')` yields `torch.device('cuda')` | **VERIFIED** |
| **Missing Checkpoint Fast-Fail** | Raises `CheckpointNotFoundError` on non-existent path | **VERIFIED** |
| **Silent Weight Download Prevention** | DenseNet201 backbone instantiated with `weights=None` | **VERIFIED** |
| **Configurable Checkpoint Paths** | Accepts `--weights` CLI flag or function argument | **VERIFIED** |
| **Canonical Class Invariant** | Strictly verified across all runs: `('cataract', 'diabetic_retinopathy', 'glaucoma', 'normal')` | **VERIFIED** |

---

### 6. Scope & Clinical Decision-Support Notice

> [!CAUTION]
> **RESEARCH & DECISION-SUPPORT PROTOTYPE ONLY**
> This machine learning pipeline is designed exclusively as an investigational multimodal health-screening prototype. It does **not** provide clinical diagnosis, does **not** replace licensed ophthalmic examinations or medical doctor evaluation, and must **never** be used as an autonomous diagnostic device. All predictions and vital-sign severity classifications require independent clinical validation by a board-certified physician.
