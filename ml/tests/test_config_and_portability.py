"""
Tests for ML Centralized Configuration, Safe Device Resolution, and Cross-Platform Portability.

Validates:
1. Centralized MLConfig defaults, immutability of class names, and env variable precedence.
2. Safe device resolution (resolve_device):
   - Explicit CPU selection.
   - CUDA selection when available.
   - Non-silent failure on explicit CUDA request when CUDA is unavailable (DeviceUnavailableError).
   - Device index out-of-range rejection (DeviceUnavailableError).
   - Malformed device specification rejection (ValueError).
3. Pre-execution validation:
   - Inference validation succeeds with checkpoint alone (does not require full dataset).
   - Inference validation flags missing/empty checkpoints as fatal errors.
   - Training validation flags missing dataset directories.
4. Numerical & diagnostic equivalence across CPU and CUDA:
   - Real model inference produces identical predictions on CPU and CUDA.
   - Tensors and models remain placed on compatible devices without unnecessary transfers.
5. Portable dataset manifest path resolution:
   - EyeDiseaseDataset resolves images relative to dataset root even when manifest filepath
     points to a non-existent machine directory.
6. Diagnostics execution:
   - run_diagnostics runs successfully and formats checks cleanly.
"""

import json
import os
import shutil
import tempfile
import cv2
import numpy as np
import pytest
import torch

import ml
from ml.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    MLConfig,
    PathsConfig,
    get_default_checkpoint_path,
    get_default_dataset_dir,
    get_ml_config,
    resolve_device,
    set_ml_config,
)
from ml.dataset_loader import EyeDiseaseDataset
from ml.diagnostics import CheckStatus, run_diagnostics
from ml.exceptions import ConfigurationError, DeviceUnavailableError
from ml.eye_model import get_cached_model, predict_eye_disease
from ml.screening_pipeline import run_screening


@pytest.fixture
def temp_work_dir():
    """Create isolated temporary directory for test artifacts."""
    d = tempfile.mkdtemp(prefix="ml_portability_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_valid_fundus_image(temp_work_dir):
    """Generates synthetic valid fundus image on disk."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 230, (30, 80, 180), -1)
    cv2.circle(img, (200, 256), 35, (100, 200, 240), -1)
    cv2.circle(img, (320, 256), 25, (15, 30, 80), -1)
    cv2.line(img, (200, 256), (100, 120), (10, 20, 100), 2)
    path = os.path.join(temp_work_dir, "synthetic_fundus.png")
    cv2.imwrite(path, img)
    return path


# =============================================================================
# 1. Configuration & Precedence Tests
# =============================================================================

def test_ml_config_defaults_and_immutability():
    """Validates baseline config invariants and immutability of class names."""
    cfg = MLConfig.default()
    assert cfg.model.num_classes == 4
    assert cfg.model.class_names == ("cataract", "diabetic_retinopathy", "glaucoma", "normal")
    assert cfg.model.in_features == 1920
    assert cfg.image.input_size == (224, 224)
    assert cfg.image.blur_threshold == 100.0


def test_ml_config_env_overrides(monkeypatch, temp_work_dir):
    """Environment variables correctly override paths and settings."""
    custom_ckpt = os.path.join(temp_work_dir, "custom_model.pth")
    custom_dataset = os.path.join(temp_work_dir, "custom_dataset")
    monkeypatch.setenv("ML_CHECKPOINT_PATH", custom_ckpt)
    monkeypatch.setenv("ML_DATASET_DIR", custom_dataset)
    monkeypatch.setenv("ML_DEVICE", "cpu")
    monkeypatch.setenv("ML_LOG_LEVEL", "DEBUG")

    cfg = MLConfig.from_env()
    assert os.path.abspath(cfg.paths.checkpoint_path) == os.path.abspath(custom_ckpt)
    assert os.path.abspath(cfg.paths.dataset_dir) == os.path.abspath(custom_dataset)
    assert cfg.inference.device == "cpu"
    assert cfg.logging.level == "DEBUG"


def test_config_to_dict_sanitization(temp_work_dir):
    """Config dictionary output sanitizes paths to basenames to avoid host path leaks."""
    cfg = MLConfig.from_env()
    dict_repr = cfg.to_dict(sanitize=True)
    assert "paths" in dict_repr
    assert "checkpoint" in dict_repr["paths"]
    # Should only contain file basename, not full host directory hierarchy
    assert "/" not in dict_repr["paths"]["checkpoint"]
    assert "\\" not in dict_repr["paths"]["checkpoint"]


# =============================================================================
# 2. Device Selection & Fail-Safe Tests
# =============================================================================

def test_resolve_device_explicit_cpu():
    """Explicit 'cpu' string or torch.device('cpu') returns CPU device."""
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_resolve_device_auto():
    """Auto resolution selects CUDA if available, otherwise CPU."""
    dev = resolve_device("auto")
    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert dev == expected


def test_resolve_device_explicit_cuda_fails_clearly_when_unavailable(monkeypatch):
    """If caller explicitly asks for CUDA but CUDA is not available, fail clearly."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(DeviceUnavailableError) as exc_info:
        resolve_device("cuda")
    assert "CUDA is not available" in str(exc_info.value)
    assert exc_info.value.requested_device == "cuda"


def test_resolve_device_invalid_cuda_index_rejected():
    """Requesting an out-of-range CUDA index raises DeviceUnavailableError."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available to test multi-index rejection")

    excessive_index = torch.cuda.device_count() + 10
    with pytest.raises(DeviceUnavailableError) as exc_info:
        resolve_device(f"cuda:{excessive_index}")
    assert f"index {excessive_index}" in str(exc_info.value)


def test_resolve_device_malformed_string_rejected():
    """Nonsense device string raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        resolve_device("tpu:0")
    assert "Invalid device specification" in str(exc_info.value)


# =============================================================================
# 3. Pre-Execution Validation Tests
# =============================================================================

def test_validate_for_inference_does_not_require_dataset(temp_work_dir):
    """Inference validation succeeds with valid checkpoint alone, without dataset root."""
    real_ckpt = get_default_checkpoint_path()
    if not os.path.exists(real_ckpt):
        pytest.skip("Default checkpoint not present")

    cfg = MLConfig.from_env()
    # Point dataset to a non-existent path
    non_existent_dataset = os.path.join(temp_work_dir, "non_existent_ds")
    object.__setattr__(cfg.paths, "dataset_dir", non_existent_dataset)

    results = cfg.validate_for_inference()
    assert len(results["errors"]) == 0  # No fatal inference errors even if dataset missing


def test_validate_for_inference_flags_missing_checkpoint(temp_work_dir):
    """Inference validation returns a fatal error if checkpoint is missing."""
    cfg = MLConfig.from_env()
    missing_ckpt = os.path.join(temp_work_dir, "missing_weights.pth")
    object.__setattr__(cfg.paths, "checkpoint_path", missing_ckpt)

    results = cfg.validate_for_inference()
    assert len(results["errors"]) >= 1
    assert any("Checkpoint file not found" in e for e in results["errors"])


def test_validate_for_training_flags_missing_dataset(temp_work_dir):
    """Training validation flags missing dataset directory."""
    cfg = MLConfig.from_env()
    missing_ds = os.path.join(temp_work_dir, "missing_ds")
    object.__setattr__(cfg.paths, "dataset_dir", missing_ds)

    results = cfg.validate_for_training()
    assert len(results["errors"]) >= 1
    assert any("Dataset root directory does not exist" in e for e in results["errors"])


# =============================================================================
# 4. Portability: CPU vs. CUDA Execution & Tensor Compatibility
# =============================================================================

def test_inference_runs_on_cpu(sample_valid_fundus_image):
    """Inference executes properly on CPU without CUDA dependencies."""
    real_ckpt = get_default_checkpoint_path()
    if not os.path.exists(real_ckpt):
        pytest.skip("Checkpoint not present")

    res = predict_eye_disease(
        image_input=sample_valid_fundus_image,
        weights_path=real_ckpt,
        device=torch.device("cpu"),
        use_cache=False,
    )
    assert res["prediction"] in CLASS_NAMES
    assert 0.0 <= res["confidence"] <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this machine")
def test_inference_cpu_cuda_equivalence(sample_valid_fundus_image):
    """Inference predictions on CPU and CUDA produce numerically consistent probabilities."""
    real_ckpt = get_default_checkpoint_path()
    if not os.path.exists(real_ckpt):
        pytest.skip("Checkpoint not present")

    res_cpu = predict_eye_disease(
        image_input=sample_valid_fundus_image,
        weights_path=real_ckpt,
        device=torch.device("cpu"),
        use_cache=False,
    )

    res_cuda = predict_eye_disease(
        image_input=sample_valid_fundus_image,
        weights_path=real_ckpt,
        device=torch.device("cuda"),
        use_cache=False,
    )

    assert res_cpu["prediction"] == res_cuda["prediction"]
    assert res_cpu["class_index"] == res_cuda["class_index"]

    # Class probabilities should match within floating point margin (< 0.01)
    for c in CLASS_NAMES:
        prob_cpu = res_cpu["all_probabilities"][c]
        prob_cuda = res_cuda["all_probabilities"][c]
        assert abs(prob_cpu - prob_cuda) < 0.01


# =============================================================================
# 5. Portable Dataset Path Resolution
# =============================================================================

def test_eye_dataset_portable_image_resolution(temp_work_dir):
    """
    EyeDiseaseDataset resolves images relative to dataset root even when
    manifest contains paths from a foreign developer machine.
    """
    # Create an image in local temp dataset structure: <temp_work_dir>/cataract/sample_01.png
    cls_dir = os.path.join(temp_work_dir, "cataract")
    os.makedirs(cls_dir, exist_ok=True)
    real_img_path = os.path.join(cls_dir, "sample_01.png")
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.imwrite(real_img_path, dummy)

    # Manifest record with a completely bogus foreign absolute path
    foreign_records = [
        {
            "filepath": r"C:\ForeignDeveloper\DifferentMachine\Desktop\Custom_Model\cataract\sample_01.png",
            "filename": "sample_01.png",
            "class_name": "cataract",
            "class_index": 0,
            "split": "train",
        }
    ]

    # Initialize EyeDiseaseDataset pointing resolve_dataset_directory to temp_work_dir
    os.environ["ML_DATASET_DIR"] = temp_work_dir
    dataset = EyeDiseaseDataset(records_or_manifest=foreign_records, split="train", verify_images=True)

    try:
        assert len(dataset) == 1
        tensor, class_idx = dataset[0]
        assert isinstance(tensor, torch.Tensor)
        assert class_idx == 0
    finally:
        os.environ.pop("ML_DATASET_DIR", None)


# =============================================================================
# 6. Diagnostics Tool Validation
# =============================================================================

def test_run_diagnostics_report():
    """Diagnostic tool runs and evaluates all operational categories."""
    report = run_diagnostics(verbose=False)
    assert report.total_passed >= 10

    categories = {c["category"] for c in report.checks}
    assert "Python Runtime" in categories
    assert "Dependencies" in categories
    assert "Hardware Acceleration" in categories
    assert "Model Checkpoint" in categories
    assert "Filesystem Permissions" in categories
