"""
Centralized ML Configuration and Device Management Module.

Provides single-source-of-truth configuration for:
1. Project and package filesystem paths (relative, portable, and overridable via env vars).
2. Hardware device selection (safe CPU/CUDA resolution, no silent fallbacks on explicit requests).
3. Canonical model invariants (4-class order, DenseNet201 architecture dimensions).
4. Image processing & quality gate parameters.
5. Structured logging and telemetry configuration.
6. Pre-execution configuration validation for training, evaluation, and inference.

Precedence Rules:
    1. Explicit function / CLI arguments (Highest precedence)
    2. Environment variables (ML_CHECKPOINT_PATH, ML_DEVICE, etc.)
    3. Centralized project-relative defaults (Lowest precedence)
"""

from dataclasses import dataclass, field
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ml.exceptions import DeviceUnavailableError

# Root path resolution relative to this file
ML_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ML_DIR)

# Canonical 4-Class Ordering (Preserved Strictly)
CLASS_NAMES: Tuple[str, ...] = (
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal",
)
NUM_CLASSES: int = len(CLASS_NAMES)
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}
DENSENET201_IN_FEATURES: int = 1920

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

# Thread safety lock for configuration singleton
_CONFIG_LOCK = threading.Lock()
_GLOBAL_CONFIG: Optional["MLConfig"] = None


# =============================================================================
# Device Selection & Resolution
# =============================================================================

def resolve_device(
    device_req: Optional[Union[str, torch.device]] = None,
    allow_cuda: bool = True,
) -> torch.device:
    """
    Safely resolves the PyTorch execution device with strict validation.

    Rules:
    - If device_req is None or "auto":
        Selects CUDA if available, compatible, and allowed; otherwise CPU.
    - If device_req is explicitly "cpu" or torch.device("cpu"):
        Always returns torch.device("cpu").
    - If device_req is explicitly "cuda" or "cuda:<idx>":
        Strictly requires CUDA support. If CUDA is unavailable, raises DeviceUnavailableError.
        DOES NOT silently fall back to CPU when the caller explicitly requested CUDA.

    Args:
        device_req: Requested device ("auto", "cpu", "cuda", "cuda:0", or torch.device).
        allow_cuda: If False, enforces CPU execution even if CUDA is requested.

    Returns:
        torch.device instance.

    Raises:
        DeviceUnavailableError: If CUDA is explicitly requested but unavailable or incompatible.
        ValueError: If device_req is an unparseable or unsupported string.
    """
    if not allow_cuda:
        if device_req and str(device_req).startswith("cuda"):
            raise DeviceUnavailableError(
                f"CUDA execution requested ('{device_req}'), but allow_cuda is set to False.",
                requested_device=str(device_req),
            )
        return torch.device("cpu")

    # 1. Resolve from environment variable if None
    if device_req is None:
        env_dev = os.environ.get("ML_DEVICE", os.environ.get("TORCH_DEVICE", "auto")).strip().lower()
        device_req = env_dev if env_dev else "auto"

    # 2. Normalize torch.device to string for inspection
    if isinstance(device_req, torch.device):
        dev_str = str(device_req).lower()
    elif isinstance(device_req, str):
        dev_str = device_req.strip().lower()
    else:
        raise ValueError(f"Unsupported device type: {type(device_req).__name__}. Expected str or torch.device.")

    # 3. Handle Auto-selection
    if dev_str in ("auto", "default"):
        if torch.cuda.is_available():
            try:
                # Test basic tensor allocation
                _ = torch.zeros(1, device="cuda")
                return torch.device("cuda")
            except Exception:
                return torch.device("cpu")
        return torch.device("cpu")

    # 4. Handle CPU
    if dev_str == "cpu":
        return torch.device("cpu")

    # 5. Handle CUDA requests
    if dev_str == "cuda" or dev_str.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise DeviceUnavailableError(
                f"CUDA device '{dev_str}' was explicitly requested, but CUDA is not available on this system.",
                requested_device=dev_str,
            )

        # Validate index if specified (e.g. cuda:0, cuda:1)
        if ":" in dev_str:
            parts = dev_str.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                available_count = torch.cuda.device_count()
                if idx >= available_count:
                    raise DeviceUnavailableError(
                        f"Requested CUDA device index {idx} ('{dev_str}'), but only {available_count} CUDA device(s) are available.",
                        requested_device=dev_str,
                    )
            else:
                raise ValueError(f"Malformed CUDA device specification: '{dev_str}'. Expected 'cuda:<integer>'.")

        try:
            target_device = torch.device(dev_str)
            # Test memory allocation on requested index
            _ = torch.zeros(1, device=target_device)
            return target_device
        except Exception as err:
            raise DeviceUnavailableError(
                f"Requested CUDA device '{dev_str}' cannot allocate memory or initialize: {err}",
                requested_device=dev_str,
            ) from err

    raise ValueError(
        f"Invalid device specification: '{dev_str}'. Supported options: 'auto', 'cpu', 'cuda', 'cuda:<index>'."
    )


# =============================================================================
# Dataclass Configurations
# =============================================================================

@dataclass(frozen=True)
class PathsConfig:
    """Project filesystem paths resolved relative to repository root."""
    project_root: str = PROJECT_ROOT
    ml_dir: str = ML_DIR
    weights_dir: str = os.path.join(ML_DIR, "weights")
    checkpoint_path: str = os.path.join(ML_DIR, "weights", "eye_model.pth")
    selection_file: str = os.path.join(ML_DIR, "weights", "checkpoint_selection.json")
    dataset_dir: str = os.path.join(ML_DIR, "dataset")
    manifest_path: str = os.path.join(ML_DIR, "dataset", "split_manifest.json")
    evaluation_dir: str = os.path.join(ML_DIR, "evaluation_results")
    experiments_dir: str = os.path.join(ML_DIR, "experiments")
    demo_dir: str = os.path.join(ML_DIR, "demo_results")
    log_dir: str = os.path.join(PROJECT_ROOT, "logs")

    @classmethod
    def from_env(cls) -> "PathsConfig":
        """Builds PathsConfig incorporating environment variable overrides."""
        project_root = os.path.abspath(os.environ.get("ML_PROJECT_ROOT", PROJECT_ROOT))
        ml_dir = os.path.abspath(os.environ.get("ML_DIR", os.path.join(project_root, "ml")))
        weights_dir = os.path.abspath(os.environ.get("ML_WEIGHTS_DIR", os.path.join(ml_dir, "weights")))
        selection_file = os.path.abspath(
            os.environ.get("ML_SELECTION_FILE", os.path.join(weights_dir, "checkpoint_selection.json"))
        )

        # Checkpoint resolution precedence:
        # 1. Explicit environment override (ML_CHECKPOINT_PATH)
        # 2. Managed selection record in selection_file (if valid and target exists)
        # 3. Default production weights (eye_model.pth or eye_densenet201_best.pth)
        env_ckpt = os.environ.get("ML_CHECKPOINT_PATH")
        ckpt_path = None
        if env_ckpt:
            ckpt_path = os.path.abspath(env_ckpt)
        elif os.path.isfile(selection_file):
            try:
                with open(selection_file, "r", encoding="utf-8") as sf:
                    sel_data = json.load(sf)
                active_rel = sel_data.get("active_checkpoint_path")
                if active_rel:
                    candidate = (
                        active_rel
                        if os.path.isabs(active_rel)
                        else os.path.abspath(os.path.join(project_root, active_rel))
                    )
                    if os.path.isfile(candidate):
                        ckpt_path = candidate
            except Exception:
                ckpt_path = None

        if not ckpt_path:
            std_ckpt = os.path.join(weights_dir, "eye_model.pth")
            best_ckpt = os.path.join(weights_dir, "eye_densenet201_best.pth")
            ckpt_path = std_ckpt if os.path.exists(std_ckpt) else (best_ckpt if os.path.exists(best_ckpt) else std_ckpt)

        # Dataset dir
        env_dataset = os.environ.get("ML_DATASET_DIR", os.environ.get("EYE_DATASET_DIR"))
        dataset_dir = os.path.abspath(env_dataset) if env_dataset else os.path.join(ml_dir, "dataset")

        manifest_path = os.path.abspath(
            os.environ.get("ML_MANIFEST_PATH", os.path.join(ml_dir, "dataset", "split_manifest.json"))
        )
        eval_dir = os.path.abspath(
            os.environ.get("ML_EVAL_DIR", os.path.join(ml_dir, "evaluation_results"))
        )
        experiments_dir = os.path.abspath(
            os.environ.get("ML_EXPERIMENTS_DIR", os.path.join(ml_dir, "experiments"))
        )
        demo_dir = os.path.abspath(
            os.environ.get("ML_DEMO_DIR", os.path.join(ml_dir, "demo_results"))
        )
        log_dir = os.path.abspath(
            os.environ.get("ML_LOG_DIR", os.path.join(project_root, "logs"))
        )

        return cls(
            project_root=project_root,
            ml_dir=ml_dir,
            weights_dir=weights_dir,
            checkpoint_path=ckpt_path,
            selection_file=selection_file,
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            evaluation_dir=eval_dir,
            experiments_dir=experiments_dir,
            demo_dir=demo_dir,
            log_dir=log_dir,
        )


@dataclass(frozen=True)
class ModelConfig:
    """Canonical 4-class DenseNet201 architecture configuration."""
    architecture: str = "DenseNet201"
    class_names: Tuple[str, ...] = CLASS_NAMES
    num_classes: int = NUM_CLASSES
    in_features: int = DENSENET201_IN_FEATURES
    class_to_idx: Dict[str, int] = field(default_factory=lambda: dict(CLASS_TO_IDX))
    idx_to_class: Dict[int, str] = field(default_factory=lambda: dict(IDX_TO_CLASS))


@dataclass(frozen=True)
class ImageProcessingConfig:
    """Standard ImageNet preprocessing and retinal image quality gate bounds."""
    input_size: Tuple[int, int] = (224, 224)
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    min_dimension: int = 224
    max_dimension: int = 4096
    blur_threshold: float = 100.0
    brightness_min: float = 30.0
    brightness_max: float = 225.0
    contrast_min: float = 20.0


@dataclass(frozen=True)
class InferenceConfig:
    """Inference execution defaults."""
    device: str = "auto"
    use_cache: bool = True
    include_gradcam: bool = True
    gradcam_target_layer: str = "features.denseblock4.denselayer32.conv2"
    gradcam_alpha: float = 0.4

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        device_str = os.environ.get("ML_DEVICE", os.environ.get("TORCH_DEVICE", "auto")).strip()
        use_cache = os.environ.get("ML_USE_CACHE", "true").lower() in ("1", "true", "yes")
        include_gradcam = os.environ.get("ML_INCLUDE_GRADCAM", "true").lower() in ("1", "true", "yes")
        return cls(device=device_str, use_cache=use_cache, include_gradcam=include_gradcam)


@dataclass(frozen=True)
class LoggingConfig:
    """Structured telemetry logging configuration."""
    level: str = "INFO"
    log_to_console: bool = True
    log_file: Optional[str] = None
    json_format: bool = False

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        level = os.environ.get("ML_LOG_LEVEL", "INFO").upper()
        log_file = os.environ.get("ML_LOG_FILE")
        json_format = os.environ.get("ML_LOG_JSON", "false").lower() in ("1", "true", "yes")
        return cls(level=level, log_to_console=True, log_file=log_file, json_format=json_format)


@dataclass
class MLConfig:
    """
    Central unified configuration container for the Multimodal Health Screening ML layer.
    """
    paths: PathsConfig = field(default_factory=PathsConfig.from_env)
    model: ModelConfig = field(default_factory=ModelConfig)
    image: ImageProcessingConfig = field(default_factory=ImageProcessingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig.from_env)
    logging: LoggingConfig = field(default_factory=LoggingConfig.from_env)

    @classmethod
    def default(cls) -> "MLConfig":
        """Returns standard configuration without environment variable overrides."""
        return cls(
            paths=PathsConfig(),
            model=ModelConfig(),
            image=ImageProcessingConfig(),
            inference=InferenceConfig(),
            logging=LoggingConfig(),
        )

    @classmethod
    def from_env(cls) -> "MLConfig":
        """Instantiates MLConfig incorporating all active environment variables."""
        return cls(
            paths=PathsConfig.from_env(),
            model=ModelConfig(),
            image=ImageProcessingConfig(),
            inference=InferenceConfig.from_env(),
            logging=LoggingConfig.from_env(),
        )

    def get_resolved_device(self) -> torch.device:
        """Resolves target torch.device using configured device preference."""
        return resolve_device(self.inference.device)

    def to_dict(self, sanitize: bool = True) -> Dict[str, Any]:
        """
        Converts configuration to dictionary.
        When sanitize is True, sensitive paths are trimmed to relative/basenames and secrets are masked.
        """
        raw = {
            "paths": {
                "checkpoint": os.path.basename(self.paths.checkpoint_path) if sanitize else self.paths.checkpoint_path,
                "dataset_dir": os.path.basename(self.paths.dataset_dir) if sanitize else self.paths.dataset_dir,
                "manifest_path": os.path.basename(self.paths.manifest_path) if sanitize else self.paths.manifest_path,
                "evaluation_dir": os.path.basename(self.paths.evaluation_dir) if sanitize else self.paths.evaluation_dir,
                "log_dir": os.path.basename(self.paths.log_dir) if sanitize else self.paths.log_dir,
            },
            "model": {
                "architecture": self.model.architecture,
                "num_classes": self.model.num_classes,
                "class_names": list(self.model.class_names),
            },
            "image": {
                "input_size": list(self.image.input_size),
                "blur_threshold": self.image.blur_threshold,
                "brightness_range": [self.image.brightness_min, self.image.brightness_max],
                "contrast_min": self.image.contrast_min,
            },
            "inference": {
                "device": self.inference.device,
                "resolved_device": str(self.get_resolved_device()),
                "use_cache": self.inference.use_cache,
                "include_gradcam": self.inference.include_gradcam,
            },
            "logging": {
                "level": self.logging.level,
                "json_format": self.logging.json_format,
            },
        }
        return raw

    def validate_for_inference(self) -> Dict[str, List[str]]:
        """
        Validates configuration specifically for running inference.
        Does NOT require a full training dataset to exist.

        Returns:
            Dict with 'errors' (fatal blockers) and 'warnings'.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Checkpoint validation
        ckpt = self.paths.checkpoint_path
        if not os.path.exists(ckpt):
            errors.append(f"Checkpoint file not found: '{os.path.basename(ckpt)}'. Real inference requires valid weights.")
        elif os.path.isdir(ckpt):
            errors.append(f"Checkpoint path is a directory, not a file: '{os.path.basename(ckpt)}'.")
        else:
            try:
                sz = os.path.getsize(ckpt)
                if sz == 0:
                    errors.append(f"Checkpoint file is 0 bytes (corrupt/empty): '{os.path.basename(ckpt)}'.")
            except OSError as err:
                errors.append(f"Cannot read checkpoint metadata: {err}")

        # 2. Device validation
        try:
            resolved = self.get_resolved_device()
            if resolved.type == "cpu" and self.inference.device == "auto" and torch.cuda.is_available():
                warnings.append("CUDA is available on system, but device resolved to CPU.")
        except DeviceUnavailableError as dev_err:
            errors.append(str(dev_err))
        except Exception as err:
            errors.append(f"Device resolution failed: {err}")

        # 3. Model class invariants
        if len(self.model.class_names) != 4:
            errors.append(f"Invalid class count: {len(self.model.class_names)}. Expected exactly 4 classes.")
        if list(self.model.class_names) != ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]:
            errors.append(f"Canonical class ordering violated: {self.model.class_names}")

        return {"errors": errors, "warnings": warnings}

    def validate_for_training(self) -> Dict[str, List[str]]:
        """
        Validates configuration for dataset loading, training, and full evaluation.
        """
        res = self.validate_for_inference()
        errors = res["errors"]
        warnings = res["warnings"]

        # Dataset directory
        ds_dir = self.paths.dataset_dir
        if not os.path.exists(ds_dir):
            errors.append(f"Dataset root directory does not exist: '{os.path.basename(ds_dir)}'.")
        elif not os.path.isdir(ds_dir):
            errors.append(f"Dataset path is not a directory: '{os.path.basename(ds_dir)}'.")

        # Manifest
        manifest = self.paths.manifest_path
        if not os.path.exists(manifest):
            warnings.append(f"Split manifest not found at: '{os.path.basename(manifest)}'.")
        elif os.path.getsize(manifest) == 0:
            errors.append(f"Split manifest file is empty: '{os.path.basename(manifest)}'.")

        # Output directory permissions
        for dir_path, label in [(self.paths.evaluation_dir, "Evaluation directory"), (self.paths.log_dir, "Log directory")]:
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as err:
                errors.append(f"Cannot create or access {label}: {err}")

        return {"errors": errors, "warnings": warnings}


# =============================================================================
# Configuration Accessors
# =============================================================================

def get_ml_config(refresh: bool = False) -> MLConfig:
    """
    Returns the thread-safe global ML configuration instance.

    Args:
        refresh: If True, re-evaluates environment variables and paths.
    """
    global _GLOBAL_CONFIG
    with _CONFIG_LOCK:
        if _GLOBAL_CONFIG is None or refresh:
            _GLOBAL_CONFIG = MLConfig.from_env()
        return _GLOBAL_CONFIG


def set_ml_config(config: MLConfig) -> None:
    """Sets the global ML configuration instance."""
    global _GLOBAL_CONFIG
    with _CONFIG_LOCK:
        _GLOBAL_CONFIG = config


def get_default_checkpoint_path() -> str:
    """Returns the active default checkpoint path."""
    return get_ml_config().paths.checkpoint_path


def get_default_dataset_dir() -> str:
    """Returns the active default dataset directory."""
    return get_ml_config().paths.dataset_dir


def get_default_evaluation_dir() -> str:
    """Returns the active default evaluation results directory."""
    return get_ml_config().paths.evaluation_dir


def get_default_experiments_dir() -> str:
    """Returns the active default experiment runs directory."""
    return get_ml_config().paths.experiments_dir


def get_default_selection_file() -> str:
    """Returns the active checkpoint selection configuration file."""
    return get_ml_config().paths.selection_file


def get_default_manifest_path() -> str:
    """Returns the active default dataset split manifest path."""
    return get_ml_config().paths.manifest_path

