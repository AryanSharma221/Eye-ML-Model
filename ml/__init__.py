"""
Machine Learning and AI module for the Multimodal Health Screening system.
"""

from ml.quality_gate import check_image_quality
from ml.preprocessing import preprocess_image
from ml.eye_model import (
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    build_densenet201_model,
    load_eye_model_checkpoint,
    get_model,
    get_cached_model,
    clear_model_cache,
    predict_eye_disease,
    screen_eye_image,
)
from ml.gradcam import generate_gradcam
from ml.physio_model import analyze_vitals
from ml.screening_pipeline import run_screening
from ml.exceptions import (
    MLException,
    CheckpointNotFoundError,
    CheckpointCorruptError,
    CheckpointIncompatibleError,
    InvalidImageInputError,
    QualityGateRejectionError,
    ModelInferenceError,
    GradCAMGenerationError,
    PhysiologicalValidationError,
    PhysiologicalTypeValidationError,
    ResourceExhaustionError,
    DeviceUnavailableError,
    ConfigurationError,
    CheckpointManagerError,
    IneligibleCheckpointError,
    RollbackUnavailableError,
    CheckpointVerificationError,
)
from ml.config import (
    MLConfig,
    get_ml_config,
    set_ml_config,
    resolve_device,
    get_default_checkpoint_path,
    get_default_dataset_dir,
    get_default_evaluation_dir,
)
from ml.diagnostics import run_diagnostics
from ml.monitoring import (
    log_ml_event,
    measure_ml_latency,
    sanitize_path,
    get_recent_ml_events,
    clear_ml_event_buffer,
    ml_logger,
    configure_logging,
    ErrorCode,
    MLStructuredFormatter,
    MLJsonFormatter,
)
from ml.experiment_tracker import (
    ExperimentTracker,
    ExperimentRun,
    compute_file_sha256,
    verify_checkpoint_integrity,
    audit_legacy_checkpoints,
)
from ml.checkpoint_manager import (
    check_checkpoint_eligibility,
    compare_checkpoints,
    promote_checkpoint,
    rollback_checkpoint,
    get_active_checkpoint_status,
    recover_selection_state,
    audit_all_checkpoints,
)
from ml.security import (
    validate_safe_path,
    safe_atomic_write_file,
    safe_atomic_write_json,
    sanitize_error_message,
)

__all__ = [
    "check_image_quality",
    "preprocess_image",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "build_densenet201_model",
    "load_eye_model_checkpoint",
    "get_model",
    "get_cached_model",
    "clear_model_cache",
    "predict_eye_disease",
    "screen_eye_image",
    "generate_gradcam",
    "analyze_vitals",
    "run_screening",
    # Configuration & Diagnostics
    "MLConfig",
    "get_ml_config",
    "set_ml_config",
    "resolve_device",
    "get_default_checkpoint_path",
    "get_default_dataset_dir",
    "get_default_evaluation_dir",
    "run_diagnostics",
    # Exceptions
    "MLException",
    "CheckpointNotFoundError",
    "CheckpointCorruptError",
    "CheckpointIncompatibleError",
    "InvalidImageInputError",
    "QualityGateRejectionError",
    "ModelInferenceError",
    "GradCAMGenerationError",
    "PhysiologicalValidationError",
    "PhysiologicalTypeValidationError",
    "ResourceExhaustionError",
    "DeviceUnavailableError",
    "ConfigurationError",
    # Monitoring & Telemetry
    "log_ml_event",
    "measure_ml_latency",
    "sanitize_path",
    "get_recent_ml_events",
    "clear_ml_event_buffer",
    "ml_logger",
    "configure_logging",
    "ErrorCode",
    "MLStructuredFormatter",
    "MLJsonFormatter",
    # Experiment Tracking & Checkpoint Versioning
    "ExperimentTracker",
    "ExperimentRun",
    "compute_file_sha256",
    "verify_checkpoint_integrity",
    "audit_legacy_checkpoints",
    # Checkpoint Selection & Safe Management
    "check_checkpoint_eligibility",
    "compare_checkpoints",
    "promote_checkpoint",
    "rollback_checkpoint",
    "get_active_checkpoint_status",
    "recover_selection_state",
    "audit_all_checkpoints",
    "CheckpointManagerError",
    "IneligibleCheckpointError",
    "RollbackUnavailableError",
    "CheckpointVerificationError",
    # Security & Defensive Hardening
    "validate_safe_path",
    "safe_atomic_write_file",
    "safe_atomic_write_json",
    "sanitize_error_message",
]
