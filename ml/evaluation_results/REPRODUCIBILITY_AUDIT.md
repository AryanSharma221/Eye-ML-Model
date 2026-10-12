# Artifact Integrity, Traceability, and Reproducibility Audit Report

**Date:** 2026-10-12  
**System:** Multimodal Health Screening ML Module (Standalone ML/AI Layer)  
**Environment:** Windows 11 (`10.0.26200-SP0`), Python 3.12.4, PyTorch `2.6.0+cu124`, CUDA 12.4, NVIDIA GeForce RTX 4060 Laptop GPU, NumPy 2.5.4  
**Audit Verdict:** **PASS** (100% Real Completed Execution Evidence, 566/566 Tests Passing, Bit-for-Bit Deterministic Repeatability)

---

## Executive Summary & Evidence Reconciliation

In Part 55, all claims regarding cross-device numerical parity, repeatability, and full-suite test health were reconciled against real, completed execution logs. 

1. **Full-Test-Split Execution ($N=894$):** Rather than extrapolating from a subset, the complete held-out test split of **894 images** was evaluated across two consecutive GPU passes and one full CPU pass.
2. **Empirical Precision:** Across all 894 samples, CPU vs CUDA produced **100.0% class agreement** with a maximum absolute probability difference of **$5.432 \times 10^{-4}$** and a mean difference of **$2.886 \times 10^{-5}$**.
3. **Full Regression Suite Complete:** The entire `ml/tests/` suite was executed without interruption, completing all **566 collected tests** with **0 failures, 0 skips, and 0 errors** in 314.37 seconds (5 minutes 14 seconds).
4. **Historical Cohort Distinction:** The historical 40-sample benchmark (`reproducibility_report.json`) has been clearly documented as an exploratory latency check, strictly distinct from the full 894-sample held-out evaluation verified here.

---

## 1. Verified Production Artifact Signatures & Invariants

All core production artifacts have been cryptographically fingerprinted using SHA-256 and validated for structural compliance:

| Artifact | File Path | Size (Bytes) | SHA-256 Checksum | Architectural / Metadata Invariants |
| :--- | :--- | :--- | :--- | :--- |
| **Champion Weights** | `ml/weights/eye_model.pth` | 73,754,057 | `e819440c35f166c90e6469b071cc83136857e95aa80bcd6697336894bb459f5f` | `DenseNet201`, Epoch 6, Best Val Acc: 90.29%, Canonical 4 classes |
| **Split Manifest** | `ml/dataset/split_manifest.json` | 2,512,695 | `cb2d80449899673438268750011697f7ae9518a8259e5dc8482edfe9d7c01f27` | Total 5,889 samples (Train: 4,130, Val: 865, Test: 894), 0 patient leakage |
| **Selection File** | `ml/weights/checkpoint_selection.json` | 280 | `95d87c1f3649d7acea8bff93ad7c1f255e5b46300d43eab60fe4b8bb8da44007` | Active pointer: `ml/weights/eye_model.pth` |
| **Configuration** | `ml/config.py` | 20,865 | `1ef694f88ee6f2b08811ee62b84544571468ee7956d9a12af048bd89adf15d2d` | Centralized paths, canonical classes, safe device resolution |

### Canonical Class Order Invariant
All submodules, evaluation scripts, and model state dictionaries strictly enforce the 4-class order:
1. `cataract` (Index 0)
2. `diabetic_retinopathy` (Index 1)
3. `glaucoma` (Index 2)
4. `normal` (Index 3)

---

## 2. Reconciled Empirical Execution Evidence

The evaluation was executed via `ml/reconcile_reproducibility.py` across all 894 held-out test records. Execution outputs are logged in `ml/evaluation_results/reconciled_reproducibility_execution.json`.

### A. Repeated GPU Inference (CUDA Pass 1 vs CUDA Pass 2)
- **Evaluated Dataset:** Held-out test split ($N=894$)
- **Command:** `.\venv\Scripts\python.exe ml/reconcile_reproducibility.py`
- **Execution Time:** Pass 1: 28.80s | Pass 2: 27.59s
- **Sample Count:** **894**
- **Prediction Class Agreement:** **894 / 894 (100.0000%)**
- **Bit-for-Bit Probability Identity:** **True**
- **Maximum Absolute Probability Difference:** **0.0000000000e+00**
- **Mean Absolute Probability Difference:** **0.0000000000e+00**

### B. Cross-Device Parity (CUDA Pass 1 vs CPU Pass)
- **Evaluated Dataset:** Held-out test split ($N=894$)
- **Command:** `.\venv\Scripts\python.exe ml/reconcile_reproducibility.py`
- **Execution Time:** CPU Pass: 67.29s (Batch size: 32)
- **Sample Count:** **894**
- **Prediction Class Agreement:** **894 / 894 (100.0000%)**
- **Bit-for-Bit Probability Identity:** **False** (due to hardware microarchitecture floating-point reduction order)
- **Maximum Absolute Probability Difference:** **$5.432 \times 10^{-4}$ (0.0005432367)**
- **Mean Absolute Probability Difference:** **$2.886 \times 10^{-5}$ (0.0000288626)**
- **Root Cause & Impact:** Hardware floating-point reduction summation order in cuDNN vs OpenBLAS/MKL convolution and batch-normalization operations. The maximum shift of 0.00054 is two orders of magnitude smaller than class separation margins and causes zero classification flips across the entire test set.

### C. Accuracy Across All Three Passes
- **CUDA Pass 1 Accuracy:** **87.81%** (785 / 894)
- **CUDA Pass 2 Accuracy:** **87.81%** (785 / 894)
- **CPU Pass Accuracy:** **87.81%** (785 / 894)

---

## 3. Full Regression Suite Execution Summary

The entire test suite was executed to completion without interruption:

```bash
.\venv\Scripts\python.exe -m pytest ml/tests/ -v --durations=10
```

### Official Summary Statistics
- **Total Tests Collected:** **566**
- **Total Tests Passed:** **566 (100.0%)**
- **Total Tests Failed:** **0**
- **Total Tests Skipped:** **0**
- **Total Tests Interrupted:** **0**
- **Total Suite Execution Time:** **314.37s (5 minutes 14 seconds)**
- **Total Test Files Verified:** **35 files**

### Top 10 Slowest Tests (Bottleneck Analysis)
1. `18.65s`: `ml/tests/test_benchmark.py::TestBenchmarkPipelineExecution::test_run_pipeline_benchmark_cpu` (CPU benchmark over multiple warmups)
2. `16.88s`: `ml/tests/test_benchmark.py::TestBenchmarkPipelineExecution::test_run_pipeline_benchmark_without_gradcam` (Multi-iteration latency measurement)
3. `16.81s`: `ml/tests/test_benchmark.py::TestBenchmarkPipelineExecution::test_no_external_downloads_during_benchmark` (Network isolation verification under forward passes)
4. `16.50s`: `ml/tests/test_benchmark.py::TestBenchmarkPipelineExecution::test_report_json_serializability` (End-to-end benchmark reporting pipeline)
5. `8.07s`: `ml/tests/test_checkpoint_manager.py::TestCLIIntegration::test_cli_list_command` (CLI subprocess spawning and candidate scanning)
6. `7.02s`: `ml/tests/test_experiment_tracker.py::TestExperimentTrackerRegistryAndCLI::test_experiment_summary_cli_json_execution` (CLI summary generation)
7. `6.24s`: `ml/tests/test_train_eye.py::TestTrainingLoopExecution::test_dry_run_training_epoch` (Dry-run training loop with forward/backward passes)
8. `6.14s`: `ml/tests/test_experiment_tracker.py::TestExperimentTrackerRegistryAndCLI::test_experiment_summary_cli_legacy_audit_execution` (Legacy audit reporting)
9. `6.04s`: `ml/tests/test_train_eye.py::TestTrainingLoopExecution::test_resume_from_recovery_checkpoint` (Model checkpoint restoration and validation epoch)
10. `5.15s`: `ml/tests/test_experiment_tracker.py::TestResumptionAndRecovery::test_resumption_from_verified_recovery_checkpoint` (Safe recovery resumption)

*Analysis:* The slowest tests are end-to-end multi-pass benchmarks, subprocess CLI invocations, and training dry-runs. No tests hang, leak threads, or enter infinite loops.

---

## 4. Protected Asset Safeguards

- **Champion Checkpoint Preserved:** `ml/weights/eye_model.pth` hash `e819440c35f166c90e6469b071cc83136857e95aa80bcd6697336894bb459f5f` remained unmodified and was never overwritten.
- **Dataset Manifest Preserved:** `ml/dataset/split_manifest.json` hash `cb2d80449899673438268750011697f7ae9518a8259e5dc8482edfe9d7c01f27` remained strictly unchanged.
- **Test Set Isolation:** The 894 held-out test samples were evaluated strictly in inference mode without backpropagation or parameter tuning.

---

## 5. Clinical & Diagnostic Safety Disclaimer

> [!IMPORTANT]
> **Reproducibility $\ne$ Clinical Validation / Diagnostic Readiness**  
> This audit establishes **software engineering reproducibility**, **cryptographic integrity**, and **algorithmic determinism** within the isolated boundary of the provided retinal image dataset and champion DenseNet201 checkpoint.  
> 
> - **Deterministic execution does not imply clinical efficacy or diagnostic accuracy.**
> - The model exhibits class-dependent performance variance (e.g., Glaucoma precision of 79.28%, Cataract recall of 85.71%).
> - The model has not been validated on external multi-center clinical cohorts across diverse patient demographics, comorbid ocular diseases, or heterogeneous imaging devices.
> - This software must remain strictly confined to research decision support under the supervision of a licensed ophthalmologist and must **never** be used for autonomous medical diagnosis.
