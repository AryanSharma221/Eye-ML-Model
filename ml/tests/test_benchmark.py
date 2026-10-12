"""
Automated Unit and Integration Tests for Benchmark and Optimization Suite (Part 10).

Validates:
1. Benchmark metric calculations (mean, median, p95, min, max, std, throughput).
2. Warm-up exclusion from measurement samples.
3. Successful and failed request accounting.
4. Process and CUDA memory metrics.
5. CPU execution compatibility.
6. CUDA timing synchronization when CUDA is available.
7. Missing checkpoint handling and labeling.
8. Zero external network downloads during benchmark execution.
9. End-to-end benchmark reporting structure and JSON serialization.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
import torch

from ml.benchmark import (
    calculate_stats,
    get_cuda_memory_info,
    get_process_memory_info,
    locate_eligible_benchmark_image,
    run_pipeline_benchmark,
    sync_device,
)
from ml.eye_model import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT_PATH,
    NUM_CLASSES,
    build_densenet201_model,
)
from ml.quality_gate import check_image_quality


@pytest.fixture
def synthetic_retina_image(tmp_path) -> str:
    """Creates a synthetic retinal fundus image passing all quality gate checks."""
    img_path = str(tmp_path / "test_benchmark_retina.png")
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 100, (140, 130, 120), -1)

    np.random.seed(42)
    noise = np.random.randint(-40, 40, (300, 300, 3), dtype=np.int16)
    noisy_canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.circle(noisy_canvas, (190, 150), 22, (230, 220, 200), -1)
    cv2.line(noisy_canvas, (190, 150), (100, 100), (40, 30, 160), 2)
    cv2.line(noisy_canvas, (190, 150), (100, 200), (40, 30, 160), 2)
    cv2.line(noisy_canvas, (190, 150), (130, 150), (40, 30, 160), 2)
    cv2.imwrite(img_path, noisy_canvas)
    return img_path


@pytest.fixture
def valid_mock_checkpoint(tmp_path) -> str:
    """Creates a temporary valid DenseNet201 checkpoint file."""
    ckpt_path = str(tmp_path / "valid_model.pth")
    model = build_densenet201_model(num_classes=NUM_CLASSES)
    payload = {
        "state_dict": model.state_dict(),
        "classes": list(CLASS_NAMES),
        "epoch": 1,
        "best_acc": 0.85,
        "architecture": "DenseNet201",
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


class TestBenchmarkMetricCalculations:
    """Verifies precision and robustness of mathematical and statistical computations."""

    def test_calculate_stats_empty_samples(self):
        stats = calculate_stats([], warmup_count=3)
        assert stats["mean_ms"] == 0.0
        assert stats["median_ms"] == 0.0
        assert stats["p95_ms"] == 0.0
        assert stats["min_ms"] == 0.0
        assert stats["max_ms"] == 0.0
        assert stats["samples_count"] == 0
        assert stats["warmup_count"] == 3
        assert stats["throughput_fps"] == 0.0

    def test_calculate_stats_known_values(self):
        samples = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = calculate_stats(samples, warmup_count=2, batch_size=1)
        assert stats["mean_ms"] == 30.0
        assert stats["median_ms"] == 30.0
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 50.0
        assert stats["samples_count"] == 5
        assert stats["warmup_count"] == 2
        # P95 of [10, 20, 30, 40, 50] is 48.0
        assert stats["p95_ms"] == 48.0
        # Throughput = 1000 / 30.0 = 33.33 FPS
        assert stats["throughput_fps"] == 33.33

    def test_calculate_stats_single_sample(self):
        samples = [25.5]
        stats = calculate_stats(samples, warmup_count=1)
        assert stats["mean_ms"] == 25.5
        assert stats["median_ms"] == 25.5
        assert stats["min_ms"] == 25.5
        assert stats["max_ms"] == 25.5
        assert stats["p95_ms"] == 25.5
        assert stats["std_ms"] == 0.0
        assert stats["throughput_fps"] == round(1000.0 / 25.5, 2)

    def test_warmup_exclusion(self):
        # 5 warm-up iterations executed, 10 measurements recorded
        warmup_iters = 5
        measurement_iters = 10
        raw_measurements = [12.0 + i for i in range(measurement_iters)]

        stats = calculate_stats(raw_measurements, warmup_count=warmup_iters)
        assert stats["samples_count"] == measurement_iters
        assert stats["warmup_count"] == warmup_iters

    def test_failed_requests_accounting(self):
        stats = calculate_stats([15.0, 16.0], successful_count=2, failed_count=3)
        assert stats["successful_count"] == 2
        assert stats["failed_count"] == 3

    def test_throughput_scales_with_batch_size(self):
        samples = [100.0]  # 100 ms per step
        stat_b1 = calculate_stats(samples, batch_size=1)
        stat_b4 = calculate_stats(samples, batch_size=4)
        stat_b8 = calculate_stats(samples, batch_size=8)

        assert stat_b1["throughput_fps"] == 10.0  # (1 * 1000) / 100
        assert stat_b4["throughput_fps"] == 40.0  # (4 * 1000) / 100
        assert stat_b8["throughput_fps"] == 80.0  # (8 * 1000) / 100


class TestMemoryAndResourceUtilities:
    """Tests process and GPU memory retrieval routines."""

    def test_process_memory_info_format(self):
        mem = get_process_memory_info()
        assert isinstance(mem, dict)
        assert "platform" in mem
        if mem["platform"] == "Windows":
            assert mem.get("working_set_mb") is not None
            assert mem["working_set_mb"] > 0
            assert mem.get("private_usage_mb") is not None

    def test_cuda_memory_info_cpu_behavior(self):
        cpu_device = torch.device("cpu")
        cuda_mem = get_cuda_memory_info(cpu_device)
        assert cuda_mem["cuda_available"] is False
        assert cuda_mem["allocated_mb"] == 0.0
        assert cuda_mem["max_allocated_mb"] == 0.0

    def test_cuda_sync_device_safety(self):
        # Must execute without error on both CPU and CUDA
        sync_device(torch.device("cpu"))
        if torch.cuda.is_available():
            sync_device(torch.device("cuda"))


@pytest.mark.slow
@pytest.mark.integration
class TestBenchmarkPipelineExecution:
    """Verifies pipeline benchmark execution, devices, and edge cases."""

    def test_locate_eligible_benchmark_image_synthetic(self, tmp_path):
        img_path, is_real = locate_eligible_benchmark_image(manifest_path="nonexistent_manifest.json")
        assert os.path.exists(img_path)
        q = check_image_quality(img_path)
        assert q["overall_pass"] is True

    def test_locate_eligible_benchmark_image_candidate_override(self, synthetic_retina_image):
        img_path, is_real = locate_eligible_benchmark_image(candidate_path=synthetic_retina_image)
        assert img_path == synthetic_retina_image
        assert is_real is True

    def test_run_pipeline_benchmark_cpu(self, synthetic_retina_image, valid_mock_checkpoint):
        cpu_device = torch.device("cpu")
        report = run_pipeline_benchmark(
            image_path=synthetic_retina_image,
            checkpoint_path=valid_mock_checkpoint,
            device=cpu_device,
            iterations=2,
            warmup=1,
            include_gradcam=True,
        )

        assert "benchmark_metadata" in report
        assert report["benchmark_metadata"]["device"] == "cpu"
        assert report["benchmark_metadata"]["is_trained_checkpoint"] is True

        stages = report["stage_latency_summary_ms"]
        assert "image_loading" in stages
        assert "quality_gate" in stages
        assert "preprocessing" in stages
        assert "model_inference" in stages
        assert "gradcam" in stages
        assert "end_to_end_with_gradcam" in stages
        assert "end_to_end_no_gradcam" in stages

        # Check that all timings are positive numbers
        for stage_name, stat in stages.items():
            assert stat["mean_ms"] >= 0.0
            assert stat["min_ms"] >= 0.0
            assert stat["max_ms"] >= 0.0
            assert stat["samples_count"] == 2

        btn = report["bottleneck_analysis"]
        assert "primary_bottleneck" in btn
        assert "stage_percentages" in btn
        assert len(btn["stage_ranking"]) > 0

    def test_run_pipeline_benchmark_without_gradcam(self, synthetic_retina_image, valid_mock_checkpoint):
        cpu_device = torch.device("cpu")
        report = run_pipeline_benchmark(
            image_path=synthetic_retina_image,
            checkpoint_path=valid_mock_checkpoint,
            device=cpu_device,
            iterations=2,
            warmup=1,
            include_gradcam=False,
        )

        stages = report["stage_latency_summary_ms"]
        assert stages["gradcam"]["samples_count"] == 0
        assert stages["end_to_end_with_gradcam"]["samples_count"] == 0
        assert stages["end_to_end_no_gradcam"]["samples_count"] == 2

    def test_missing_checkpoint_handling(self, synthetic_retina_image, tmp_path):
        missing_ckpt = str(tmp_path / "does_not_exist.pth")
        # Should raise FileNotFoundError when attempting to benchmark missing checkpoint
        with pytest.raises(FileNotFoundError):
            run_pipeline_benchmark(
                image_path=synthetic_retina_image,
                checkpoint_path=missing_ckpt,
                device=torch.device("cpu"),
                iterations=1,
                warmup=0,
            )

    def test_no_external_downloads_during_benchmark(self, synthetic_retina_image, valid_mock_checkpoint):
        with patch("urllib.request.urlopen") as mock_url, patch("http.client.HTTPConnection.connect") as mock_http:
            mock_url.side_effect = RuntimeError("External download prohibited")
            mock_http.side_effect = RuntimeError("External download prohibited")

            report = run_pipeline_benchmark(
                image_path=synthetic_retina_image,
                checkpoint_path=valid_mock_checkpoint,
                device=torch.device("cpu"),
                iterations=1,
                warmup=1,
                include_gradcam=False,
            )
            assert report["benchmark_metadata"]["iterations"] == 1
            mock_url.assert_not_called()
            mock_http.assert_not_called()

    def test_report_json_serializability(self, synthetic_retina_image, valid_mock_checkpoint):
        report = run_pipeline_benchmark(
            image_path=synthetic_retina_image,
            checkpoint_path=valid_mock_checkpoint,
            device=torch.device("cpu"),
            iterations=1,
            warmup=0,
            include_gradcam=False,
        )
        json_str = json.dumps(report)
        reloaded = json.loads(json_str)
        assert reloaded["benchmark_metadata"]["python_version"] == report["benchmark_metadata"]["python_version"]
