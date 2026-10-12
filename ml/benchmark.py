"""
Benchmark and Performance Optimization Suite for Eye Disease Screening Pipeline (Part 10).

Measures:
1. Image loading latency (disk -> memory).
2. Quality-gate validation latency (OpenCV blur, brightness, contrast, Hough circles).
3. Preprocessing pipeline latency (PIL decode -> resize 224x224 -> ImageNet normalize -> Tensor).
4. Steady-state DenseNet201 model inference latency (torch.inference_mode on GPU/CPU).
5. Grad-CAM visual explainability generation latency (forward/backward hooks + heatmap blend + PNG base64).
6. End-to-end screening workflow latency (Quality Gate -> Preprocess -> DenseNet201 -> Grad-CAM).
7. Repeated inference throughput (images/second across batch sizes 1, 4, 8, 16).
8. Process and CUDA memory footprints, cold vs cached model loading, and bottleneck identification.

Uses high-resolution monotonic timers (time.perf_counter) and explicit CUDA synchronization
(torch.cuda.synchronize) when executing on GPU.
"""

import argparse
import ctypes
from ctypes import wintypes
import gc
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from ml.eye_model import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT_PATH,
    NUM_CLASSES,
    build_densenet201_model,
    clear_model_cache,
    get_cached_model,
    get_default_device,
    get_model,
    predict_eye_disease,
    screen_eye_image,
)
from ml.gradcam import generate_gradcam
from ml.preprocessing import preprocess_image
from ml.quality_gate import check_image_quality


# ---------------------------------------------------------------------------
# High-Resolution Timing & Device Synchronization
# ---------------------------------------------------------------------------

def sync_device(device: torch.device) -> None:
    """Synchronizes device execution before and after timing CUDA operations."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Memory Measurement Utilities (No extra dependencies)
# ---------------------------------------------------------------------------

class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def get_process_memory_info() -> Dict[str, Any]:
    """
    Retrieves current OS process memory metrics without external packages.
    On Windows: Uses kernel32 / psapi GetProcessMemoryInfo.
    On Unix: Uses resource.getrusage.
    """
    mem_info: Dict[str, Any] = {
        "platform": platform.system(),
        "working_set_mb": None,
        "private_usage_mb": None,
    }

    if platform.system() == "Windows":
        try:
            get_mem = ctypes.windll.psapi.GetProcessMemoryInfo
            get_mem.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX),
                wintypes.DWORD,
            ]
            get_mem.restype = wintypes.BOOL

            get_proc = ctypes.windll.kernel32.GetCurrentProcess
            get_proc.restype = wintypes.HANDLE

            handle = get_proc()
            pmc = _PROCESS_MEMORY_COUNTERS_EX()
            pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS_EX)
            if get_mem(handle, ctypes.byref(pmc), pmc.cb):
                mem_info["working_set_mb"] = round(pmc.WorkingSetSize / (1024 ** 2), 2)
                mem_info["private_usage_mb"] = round(pmc.PrivateUsage / (1024 ** 2), 2)
        except Exception as err:
            mem_info["error"] = str(err)
    else:
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # maxrss is in KB on Linux, bytes on macOS
            factor = 1024 if platform.system() == "Darwin" else 1
            mem_info["working_set_mb"] = round(usage.ru_maxrss / (1024 * factor), 2)
        except Exception as err:
            mem_info["error"] = str(err)

    return mem_info


def get_cuda_memory_info(device: torch.device) -> Dict[str, Any]:
    """Retrieves PyTorch CUDA memory metrics if running on GPU."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "allocated_mb": 0.0,
            "max_allocated_mb": 0.0,
            "reserved_mb": 0.0,
        }

    return {
        "cuda_available": True,
        "allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 ** 2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 ** 2), 2),
    }


# ---------------------------------------------------------------------------
# Statistical Analysis Engine
# ---------------------------------------------------------------------------

def calculate_stats(
    latencies_ms: List[float],
    warmup_count: int = 0,
    batch_size: int = 1,
    successful_count: Optional[int] = None,
    failed_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Computes summary latency statistics and throughput from raw milliseconds measurements.

    Args:
        latencies_ms: List of measured latency samples in milliseconds.
        warmup_count: Number of initial warm-up iterations executed prior to measurement.
        batch_size: Number of items processed per iteration.
        successful_count: Number of successful requests (defaults to len(latencies_ms)).
        failed_count: Number of failed requests (defaults to 0).

    Returns:
        Dictionary containing mean, median, p95, min, max, std, throughput, etc.
    """
    if not latencies_ms:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
            "samples_count": 0,
            "warmup_count": warmup_count,
            "batch_size": batch_size,
            "throughput_fps": 0.0,
            "successful_count": successful_count or 0,
            "failed_count": failed_count or 0,
        }

    arr = np.array(latencies_ms, dtype=np.float64)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    std_val = float(np.std(arr))

    # Compute p95 when at least 1 sample exists (np.percentile supports any size)
    p95_val = float(np.percentile(arr, 95))

    # Throughput in images / second
    throughput_fps = round((batch_size * 1000.0) / mean_val, 2) if mean_val > 0 else 0.0

    return {
        "mean_ms": round(mean_val, 2),
        "median_ms": round(median_val, 2),
        "p95_ms": round(p95_val, 2),
        "min_ms": round(min_val, 2),
        "max_ms": round(max_val, 2),
        "std_ms": round(std_val, 2),
        "samples_count": len(latencies_ms),
        "warmup_count": warmup_count,
        "batch_size": batch_size,
        "throughput_fps": throughput_fps,
        "successful_count": len(latencies_ms) if successful_count is None else successful_count,
        "failed_count": 0 if failed_count is None else failed_count,
    }


# ---------------------------------------------------------------------------
# Dataset Discovery & Eligible Benchmark Image Locator
# ---------------------------------------------------------------------------

def locate_eligible_benchmark_image(
    candidate_path: Optional[str] = None,
    manifest_path: str = os.path.join("ml", "dataset", "split_manifest.json"),
) -> Tuple[str, bool]:
    """
    Locates or creates an eligible retinal fundus image that passes quality checks.

    Priority:
        1. Explicit candidate_path if passed and valid.
        2. First passing image found in test split of split_manifest.json.
        3. Synthetic compliant fundus image generated in scratch directory.

    Returns:
        (image_path, is_real_dataset_image)
    """
    if candidate_path and os.path.exists(candidate_path):
        q = check_image_quality(candidate_path)
        if q.get("overall_pass", False):
            return candidate_path, True

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            splits_data = manifest.get("splits", {})
            test_items = splits_data.get("test", [])
            for item in test_items:
                fp = item.get("filepath")
                if fp and os.path.exists(fp):
                    q = check_image_quality(fp)
                    if q.get("overall_pass", False):
                        return fp, True
        except Exception:
            pass

    # Fallback: create compliant synthetic image in scratch
    scratch_dir = os.path.join("ml", "dataset", "scratch_benchmark")
    os.makedirs(scratch_dir, exist_ok=True)
    fallback_path = os.path.join(scratch_dir, "synthetic_benchmark_fundus.png")

    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 100, (140, 130, 120), -1)
    np.random.seed(42)
    noise = np.random.randint(-40, 40, (300, 300, 3), dtype=np.int16)
    noisy_canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.circle(noisy_canvas, (190, 150), 22, (230, 220, 200), -1)
    cv2.line(noisy_canvas, (190, 150), (100, 100), (40, 30, 160), 2)
    cv2.line(noisy_canvas, (190, 150), (100, 200), (40, 30, 160), 2)
    cv2.imwrite(fallback_path, noisy_canvas)

    return fallback_path, False


# ---------------------------------------------------------------------------
# Core Benchmark Runner
# ---------------------------------------------------------------------------

def run_pipeline_benchmark(
    image_path: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    iterations: int = 30,
    warmup: int = 5,
    include_gradcam: bool = True,
) -> Dict[str, Any]:
    """
    Executes a comprehensive, stage-by-stage latency and throughput benchmark.

    Stages Measured:
        1. Image Loading (cv2.imread)
        2. Quality Gate (check_image_quality)
        3. Preprocessing (preprocess_image)
        4. Model Initialization (cold checkpoint load from disk)
        5. Steady-State Inference (forward pass under torch.inference_mode)
        6. Grad-CAM Generation (generate_gradcam)
        7. End-to-End Screening (screen_eye_image with Grad-CAM)
        8. End-to-End Screening without Grad-CAM (screen_eye_image no Grad-CAM)

    Returns:
        Structured dictionary with timing stats for each stage, memory metrics,
        bottleneck analysis, and runtime environment details.
    """
    target_device = device if device is not None else get_default_device()
    target_ckpt = checkpoint_path if checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH

    # Verify checkpoint existence
    is_trained_checkpoint = False
    if os.path.exists(target_ckpt):
        try:
            raw = torch.load(target_ckpt, map_location="cpu", weights_only=True)
            if isinstance(raw, dict) and "state_dict" in raw:
                is_trained_checkpoint = True
        except Exception:
            pass

    # Ensure baseline memory reading before benchmark
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(target_device)

    mem_start = get_process_memory_info()
    cuda_mem_start = get_cuda_memory_info(target_device)

    # -----------------------------------------------------------------------
    # Stage 1: Image Loading (cv2.imread)
    # -----------------------------------------------------------------------
    load_times: List[float] = []
    for _ in range(warmup):
        _ = cv2.imread(image_path)
    for _ in range(iterations):
        t0 = time.perf_counter()
        img = cv2.imread(image_path)
        t1 = time.perf_counter()
        load_times.append((t1 - t0) * 1000.0)
    stage_image_loading = calculate_stats(load_times, warmup_count=warmup)

    # -----------------------------------------------------------------------
    # Stage 2: Quality Gate (check_image_quality)
    # -----------------------------------------------------------------------
    qg_times: List[float] = []
    qg_successes = 0
    qg_fails = 0
    for _ in range(warmup):
        _ = check_image_quality(image_path)
    for _ in range(iterations):
        t0 = time.perf_counter()
        q_res = check_image_quality(image_path)
        t1 = time.perf_counter()
        qg_times.append((t1 - t0) * 1000.0)
        if q_res.get("overall_pass", False):
            qg_successes += 1
        else:
            qg_fails += 1
    stage_quality_gate = calculate_stats(
        qg_times,
        warmup_count=warmup,
        successful_count=qg_successes,
        failed_count=qg_fails,
    )

    # -----------------------------------------------------------------------
    # Stage 3: Preprocessing (preprocess_image)
    # -----------------------------------------------------------------------
    prep_times: List[float] = []
    for _ in range(warmup):
        _ = preprocess_image(image_path)
    for _ in range(iterations):
        t0 = time.perf_counter()
        tensor = preprocess_image(image_path)
        t1 = time.perf_counter()
        prep_times.append((t1 - t0) * 1000.0)
    stage_preprocessing = calculate_stats(prep_times, warmup_count=warmup)

    # -----------------------------------------------------------------------
    # Stage 4: Model Initialization (Cold load vs Cached load)
    # -----------------------------------------------------------------------
    clear_model_cache()
    sync_device(target_device)
    t0 = time.perf_counter()
    model = get_cached_model(weights_path=target_ckpt, device=target_device)
    sync_device(target_device)
    t1 = time.perf_counter()
    cold_load_ms = round((t1 - t0) * 1000.0, 2)

    # Measure cached retrieval (in-memory cache hit)
    sync_device(target_device)
    t0 = time.perf_counter()
    _ = get_cached_model(weights_path=target_ckpt, device=target_device)
    sync_device(target_device)
    t1 = time.perf_counter()
    cached_lookup_ms = round((t1 - t0) * 1000.0, 4)

    cuda_mem_post_model = get_cuda_memory_info(target_device)

    # -----------------------------------------------------------------------
    # Stage 5: Steady-State Model Inference (predict_eye_disease)
    # -----------------------------------------------------------------------
    input_tensor = tensor.to(target_device)
    # Warmup
    for _ in range(warmup):
        sync_device(target_device)
        with torch.inference_mode():
            _ = predict_eye_disease(
                input_tensor,
                model=model,
                device=target_device,
                validate_quality=False,
            )
        sync_device(target_device)

    inf_times: List[float] = []
    sample_prediction = None
    for _ in range(iterations):
        sync_device(target_device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            pred = predict_eye_disease(
                input_tensor,
                model=model,
                device=target_device,
                validate_quality=False,
            )
        sync_device(target_device)
        t1 = time.perf_counter()
        inf_times.append((t1 - t0) * 1000.0)
        sample_prediction = pred
    stage_inference = calculate_stats(inf_times, warmup_count=warmup, batch_size=1)

    # -----------------------------------------------------------------------
    # Stage 6: Grad-CAM Generation (generate_gradcam)
    # -----------------------------------------------------------------------
    pred_class_idx = sample_prediction["class_index"] if sample_prediction else 0
    gradcam_times: List[float] = []
    if include_gradcam:
        for _ in range(warmup):
            sync_device(target_device)
            _ = generate_gradcam(
                image_path=image_path,
                predicted_class_idx=pred_class_idx,
                model=model,
                device=target_device,
            )
            sync_device(target_device)

        for _ in range(iterations):
            sync_device(target_device)
            t0 = time.perf_counter()
            _ = generate_gradcam(
                image_path=image_path,
                predicted_class_idx=pred_class_idx,
                model=model,
                device=target_device,
            )
            sync_device(target_device)
            t1 = time.perf_counter()
            gradcam_times.append((t1 - t0) * 1000.0)
    stage_gradcam = calculate_stats(gradcam_times, warmup_count=warmup)

    # -----------------------------------------------------------------------
    # Stage 7: End-to-End Screening (screen_eye_image WITH Grad-CAM)
    # -----------------------------------------------------------------------
    e2e_gradcam_times: List[float] = []
    if include_gradcam:
        for _ in range(warmup):
            sync_device(target_device)
            _ = screen_eye_image(
                image_path=image_path,
                model=model,
                device=target_device,
                include_gradcam=True,
            )
            sync_device(target_device)

        for _ in range(iterations):
            sync_device(target_device)
            t0 = time.perf_counter()
            _ = screen_eye_image(
                image_path=image_path,
                model=model,
                device=target_device,
                include_gradcam=True,
            )
            sync_device(target_device)
            t1 = time.perf_counter()
            e2e_gradcam_times.append((t1 - t0) * 1000.0)
    stage_e2e_with_gradcam = calculate_stats(e2e_gradcam_times, warmup_count=warmup)

    # -----------------------------------------------------------------------
    # Stage 8: End-to-End Screening (screen_eye_image WITHOUT Grad-CAM)
    # -----------------------------------------------------------------------
    e2e_nogradcam_times: List[float] = []
    for _ in range(warmup):
        sync_device(target_device)
        _ = screen_eye_image(
            image_path=image_path,
            model=model,
            device=target_device,
            include_gradcam=False,
        )
        sync_device(target_device)

    for _ in range(iterations):
        sync_device(target_device)
        t0 = time.perf_counter()
        _ = screen_eye_image(
            image_path=image_path,
            model=model,
            device=target_device,
            include_gradcam=False,
        )
        sync_device(target_device)
        t1 = time.perf_counter()
        e2e_nogradcam_times.append((t1 - t0) * 1000.0)
    stage_e2e_no_gradcam = calculate_stats(e2e_nogradcam_times, warmup_count=warmup)

    # -----------------------------------------------------------------------
    # Stage 9: Batched Throughput Analysis
    # -----------------------------------------------------------------------
    batched_throughput: Dict[str, Any] = {}
    for b_size in [1, 4, 8, 16]:
        batched_tensor = input_tensor.repeat(b_size, 1, 1, 1)
        sync_device(target_device)
        with torch.inference_mode():
            _ = model(batched_tensor)
        sync_device(target_device)

        b_times: List[float] = []
        for _ in range(max(10, iterations // 2)):
            sync_device(target_device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model(batched_tensor)
            sync_device(target_device)
            t1 = time.perf_counter()
            b_times.append((t1 - t0) * 1000.0)
        batched_throughput[f"batch_{b_size}"] = calculate_stats(
            b_times,
            warmup_count=2,
            batch_size=b_size,
        )

    # -----------------------------------------------------------------------
    # Post-Benchmark Resource & Memory Accounting
    # -----------------------------------------------------------------------
    mem_end = get_process_memory_info()
    cuda_mem_end = get_cuda_memory_info(target_device)

    # -----------------------------------------------------------------------
    # Bottleneck Breakdown & Dominance Analysis
    # -----------------------------------------------------------------------
    total_breakdown_ms = (
        stage_image_loading["mean_ms"]
        + stage_quality_gate["mean_ms"]
        + stage_preprocessing["mean_ms"]
        + stage_inference["mean_ms"]
        + (stage_gradcam["mean_ms"] if include_gradcam else 0.0)
    )

    stage_percentages: Dict[str, float] = {}
    if total_breakdown_ms > 0:
        stage_percentages["image_loading_pct"] = round(
            (stage_image_loading["mean_ms"] / total_breakdown_ms) * 100.0, 1
        )
        stage_percentages["quality_gate_pct"] = round(
            (stage_quality_gate["mean_ms"] / total_breakdown_ms) * 100.0, 1
        )
        stage_percentages["preprocessing_pct"] = round(
            (stage_preprocessing["mean_ms"] / total_breakdown_ms) * 100.0, 1
        )
        stage_percentages["model_inference_pct"] = round(
            (stage_inference["mean_ms"] / total_breakdown_ms) * 100.0, 1
        )
        if include_gradcam:
            stage_percentages["gradcam_pct"] = round(
                (stage_gradcam["mean_ms"] / total_breakdown_ms) * 100.0, 1
            )

    # Rank stages by mean latency to locate dominant bottleneck
    stage_ranking = sorted(
        [
            ("Image Loading", stage_image_loading["mean_ms"]),
            ("Quality Gate", stage_quality_gate["mean_ms"]),
            ("Preprocessing", stage_preprocessing["mean_ms"]),
            ("Model Inference", stage_inference["mean_ms"]),
            ("Grad-CAM Generation", stage_gradcam["mean_ms"] if include_gradcam else 0.0),
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    primary_bottleneck = stage_ranking[0][0] if stage_ranking else "Unknown"

    report: Dict[str, Any] = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "platform": platform.platform(),
            "device": str(target_device),
            "device_name": (
                torch.cuda.get_device_name(target_device)
                if target_device.type == "cuda"
                else "CPU"
            ),
            "checkpoint_path": target_ckpt,
            "is_trained_checkpoint": is_trained_checkpoint,
            "benchmark_image_path": image_path,
            "iterations": iterations,
            "warmup_iterations": warmup,
            "include_gradcam": include_gradcam,
        },
        "model_loading_overhead": {
            "cold_load_from_disk_ms": cold_load_ms,
            "cached_lookup_ms": cached_lookup_ms,
            "model_memory_footprint_cuda_mb": round(
                cuda_mem_post_model["allocated_mb"] - cuda_mem_start["allocated_mb"], 2
            ),
        },
        "stage_latency_summary_ms": {
            "image_loading": stage_image_loading,
            "quality_gate": stage_quality_gate,
            "preprocessing": stage_preprocessing,
            "model_inference": stage_inference,
            "gradcam": stage_gradcam,
            "end_to_end_with_gradcam": stage_e2e_with_gradcam,
            "end_to_end_no_gradcam": stage_e2e_no_gradcam,
        },
        "bottleneck_analysis": {
            "primary_bottleneck": primary_bottleneck,
            "total_breakdown_ms": round(total_breakdown_ms, 2),
            "stage_percentages": stage_percentages,
            "stage_ranking": [{"stage": s, "mean_ms": m} for s, m in stage_ranking],
        },
        "batched_inference_throughput": batched_throughput,
        "memory_metrics": {
            "process_memory_start": mem_start,
            "process_memory_end": mem_end,
            "cuda_memory_start": cuda_mem_start,
            "cuda_memory_end": cuda_mem_end,
            "cuda_peak_allocated_mb": cuda_mem_end["max_allocated_mb"],
        },
    }

    return report


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark and optimize the DenseNet201 Eye Disease Screening pipeline."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of measurement iterations per stage (default: 30).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up iterations to exclude (default: 5).",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to retinal test image. If omitted, discovers real image from split_manifest.json.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to DenseNet201 checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to execute benchmark on (default: auto).",
    )
    parser.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Skip Grad-CAM generation in benchmark stages.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save benchmark JSON report.",
    )

    args = parser.parse_args()

    # Resolve device
    if args.device == "cuda":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        target_device = torch.device("cpu")
    else:
        target_device = get_default_device()

    # Discover eligible test image
    benchmark_image, is_real = locate_eligible_benchmark_image(candidate_path=args.image_path)
    print("=" * 70)
    print(" MULTIMODAL HEALTH SCREENING — ML PIPELINE BENCHMARK (PART 10)")
    print("=" * 70)
    print(f"Target Device:      {target_device} ({torch.cuda.get_device_name(0) if target_device.type == 'cuda' else 'CPU'})")
    print(f"PyTorch Version:    {torch.__version__}")
    print(f"Checkpoint Path:    {args.checkpoint_path}")
    print(f"Benchmark Image:    {benchmark_image} ({'Real Dataset' if is_real else 'Synthetic Compliant'})")
    print(f"Iterations:         {args.iterations} (warm-up: {args.warmup})")
    print(f"Include Grad-CAM:   {not args.no_gradcam}")
    print("-" * 70)

    report = run_pipeline_benchmark(
        image_path=benchmark_image,
        checkpoint_path=args.checkpoint_path,
        device=target_device,
        iterations=args.iterations,
        warmup=args.warmup,
        include_gradcam=not args.no_gradcam,
    )

    # Format Summary Table
    stages = report["stage_latency_summary_ms"]
    print("\n--- STAGE-BY-STAGE LATENCY & THROUGHPUT (STEADY STATE) ---")
    header = f"{'Pipeline Stage':<30} | {'Mean (ms)':<10} | {'Median (ms)':<11} | {'P95 (ms)':<10} | {'FPS (img/s)':<11}"
    print(header)
    print("-" * len(header))

    def print_row(name: str, s: Dict[str, Any]) -> None:
        if s["samples_count"] > 0:
            print(f"{name:<30} | {s['mean_ms']:<10.2f} | {s['median_ms']:<11.2f} | {s['p95_ms']:<10.2f} | {s['throughput_fps']:<11.1f}")

    print_row("1. Image Loading (cv2.imread)", stages["image_loading"])
    print_row("2. Quality Gate (5 Checks)", stages["quality_gate"])
    print_row("3. Preprocessing (Resize/Norm)", stages["preprocessing"])
    print_row("4. DenseNet201 Forward Pass", stages["model_inference"])
    if not args.no_gradcam:
        print_row("5. Grad-CAM (Hook/Blend/B64)", stages["gradcam"])
        print_row("6. End-to-End WITH Grad-CAM", stages["end_to_end_with_gradcam"])
    print_row("7. End-to-End WITHOUT Grad-CAM", stages["end_to_end_no_gradcam"])

    print("\n--- MODEL LOADING OVERHEAD ---")
    load_ov = report["model_loading_overhead"]
    print(f"Cold Model Load from Disk:    {load_ov['cold_load_from_disk_ms']:.2f} ms")
    print(f"Cached Memory Lookup:         {load_ov['cached_lookup_ms']:.4f} ms")
    if load_ov["model_memory_footprint_cuda_mb"] > 0:
        print(f"Model VRAM Allocation:        {load_ov['model_memory_footprint_cuda_mb']:.2f} MB")

    print("\n--- BOTTLENECK ANALYSIS ---")
    btn = report["bottleneck_analysis"]
    print(f"Primary Bottleneck:           {btn['primary_bottleneck']}")
    print("Stage Breakdown Percentages:")
    for k, v in btn["stage_percentages"].items():
        print(f"  - {k:<25}: {v:>5.1f}%")

    print("\n--- BATCHED INFERENCE THROUGHPUT ---")
    for b_name, b_stat in report["batched_inference_throughput"].items():
        print(f"  {b_name:<10}: Mean = {b_stat['mean_ms']:>6.2f} ms | Throughput = {b_stat['throughput_fps']:>7.1f} img/s")

    print("\n--- MEMORY FOOTPRINT ---")
    mem = report["memory_metrics"]
    proc_end = mem["process_memory_end"]
    if proc_end.get("working_set_mb"):
        print(f"Process Working Set:          {proc_end['working_set_mb']} MB")
        print(f"Process Private Usage:        {proc_end['private_usage_mb']} MB")
    cuda_end = mem["cuda_memory_end"]
    if cuda_end.get("cuda_available"):
        print(f"CUDA Memory Allocated:        {cuda_end['allocated_mb']} MB")
        print(f"CUDA Peak Allocated:          {cuda_end['max_allocated_mb']} MB")
        print(f"CUDA Memory Reserved:         {cuda_end['reserved_mb']} MB")

    # Output JSON if requested or default path
    out_path = args.output_json or os.path.join("ml", "evaluation_results", "benchmark_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Full benchmark report saved to: {out_path}\n" + "=" * 70)


if __name__ == "__main__":
    main()
