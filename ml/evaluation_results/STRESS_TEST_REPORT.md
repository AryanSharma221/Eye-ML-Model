# Multimodal Health Screening Pipeline: Comprehensive Stress & Reliability Report

> **CLINICAL RESEARCH PROTOTYPE NOTICE**  
> This reliability and stress-test report evaluates an automated multimodal health-screening prototype  
> designed for clinical decision support. It is **not** an autonomous medical diagnostic device.

## 1. Environment & Hardware Baseline

- **Execution Timestamp:** `2026-10-12T07:46:54Z`
- **Model Checkpoint:** `eye_model.pth` (70.34 MB)
- **Compute Device:** `CUDA` (NVIDIA GeForce RTX 4060 Laptop GPU)
- **Software Environment:** Python 3.12.4 | PyTorch 2.6.0+cu124 | OS: Windows-11-10.0.26200-SP0
- **Cold-Start Model Initialization:** **432.52 ms** (RAM Delta: +88.19 MB, VRAM Delta: +70.09 MB)
- **Steady-State E2E Latency (N=20):** Median: **155.68 ms** | 95th Percentile: **176.18 ms** | Min: 140.97 ms | Max: 177.05 ms

### Pipeline Stage Latency Breakdown (Steady-State)

| Pipeline Stage | Mean (ms) | Median (ms) | 95th Percentile (ms) | Min (ms) | Max (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `quality_gate` | 6.1 | 6.09 | 7.52 | 5.04 | 8.09 |
| `preprocessing` | 3.81 | 3.78 | 4.28 | 3.4 | 4.54 |
| `eye_inference` | 31.92 | 30.72 | 37.72 | 25.82 | 42.75 |
| `gradcam` | 114.99 | 114.79 | 126.33 | 102.6 | 130.39 |
| `vitals_analysis` | 0.23 | 0.22 | 0.31 | 0.17 | 0.37 |

## 2. Repeated Execution Stress Scenarios

| Scenario Description | Iterations | Success Rate | Median Latency | P95 Latency | Memory Slope | Leak Detected |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Same Image Inference | 50 | 100.0% | 40.87 ms | 46.27 ms | 0.0004 MB/it | No (Stable) |
| Diverse Images Inference | 50 | 100.0% | 38.4 ms | 75.6 ms | 0.1578 MB/it | No (Stable) |
| Grad-CAM Generation | 30 | 100.0% | 111.49 ms | 121.95 ms | 0.0007 MB/it | No (Stable) |
| Quality Rejections | 30 | 100.0% | 59.44 ms | 63.48 ms | 0.0000 MB/it | No (Stable) |
| Alternating Success/Fail | 30 | 100.0% | 40.86 ms | 44.75 ms | 0.0000 MB/it | No (Stable) |
| Sequential Mixed Workflow | 50 | 100.0% | 41.01 ms | 159.67 ms | 0.0000 MB/it | No (Stable) |

### Malformed, Missing & Corrupted Input Resilience

- **Zero-Byte Empty Image (0 bytes):** Handled safely (`status: rejected / error`) without unhandled exceptions.
- **Random Garbage Binary File (4 KB non-image):** Decoded safely by Pillow/OpenCV with typed rejection.
- **Non-Existent Image Path:** Handled safely (`status: error`, file not found).
- **Directory Path Provided as File:** Handled safely (`status: error`).
- **Missing Model Checkpoint:** `CheckpointNotFoundError` raised with sanitized message.
- **High-Frequency Vital Sign Analysis:** 100 iterations completed with median latency **33.7 µs** (microseconds).

## 3. Concurrency Stress Test Results

| Worker Threads | Requests | Total Time (ms) | Throughput (QPS) | Errors | Predictions Match Baseline | Grad-CAM Hook Safety |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 16 | 2415.7 | **6.62** | 0 | PASS (100% Match) | PASS (Synchronized) |
| 4 | 16 | 2330.27 | **6.87** | 0 | PASS (100% Match) | PASS (Synchronized) |
| 8 | 16 | 2348.96 | **6.81** | 0 | PASS (100% Match) | PASS (Synchronized) |

### Concurrency Assessment:
- Model state remains strictly isolated and in `eval` mode across concurrent threads.
- Thread-safe `_GRADCAM_LOCK` in `ml/gradcam.py` ensures hooks and gradient backpropagation execute atomically per thread, preventing hook collision race conditions.
- All concurrent threads returned identical classification outputs matching sequential baseline execution within numerical floating-point precision ($< 10^{-4}$).

## 4. Resource Stability & Memory Leak Audit

- **Unreleased Forward Hooks:** `0` (PASS: 0 unreleased hooks)
- **Unreleased Backward Hooks:** `0` (PASS: 0 unreleased hooks)
- **Unintended Parameter Gradients Active:** `False` (PASS: None active)
- **Telemetry Ring Buffer Size:** `500 / 500` (PASS: Strictly bounded)
- **Logging Handlers Idempotency:** `True` (PASS: No duplicate handler leakage)
- **Final Working Set Memory:** `1228.92 MB`
- **Final CUDA VRAM Allocated:** `151.34 MB`
- **Overall Reliability Verdict:** **STABLE**

## 5. Stability Limitations & Production Hardening Advice

1. **Monocular 2D Classification Only:** The model relies on 2D color fundus images without OCT depth, causing higher uncertainty on borderline physiological cups.
2. **Single-GPU Serialized Grad-CAM:** While inference supports parallel tensor evaluation, Grad-CAM backpropagation utilizes a process-level lock (`_GRADCAM_LOCK`) to prevent hook collision. Under extreme concurrency (>32 QPS), Grad-CAM becomes the primary throughput bottleneck.
3. **Synchronous Disk I/O:** Image decoding via OpenCV and Pillow is synchronous; asynchronous image ingestion is recommended if integrated into high-throughput microservices in the future.

---
*Report generated automatically by `ml/stress_test.py`.*