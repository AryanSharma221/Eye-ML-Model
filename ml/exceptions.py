"""
Distinguishable Exception Hierarchy for the ML Pipeline.

Provides typed domain exceptions for specific ML operational failures while
preserving full backward compatibility with Python built-in exception types
(FileNotFoundError, IOError, ValueError, RuntimeError, TypeError).

Public Exceptions:
    MLException: Base exception for all ML pipeline domain errors.
    CheckpointNotFoundError: Checkpoint file does not exist on disk.
    CheckpointCorruptError: Checkpoint file is empty, truncated, or unpickling fails.
    CheckpointIncompatibleError: Checkpoint metadata, classes, or weights mismatch.
    InvalidImageInputError: Image file is corrupted, truncated, or unreadable.
    QualityGateRejectionError: Image fails quality gate criteria in strict mode.
    ModelInferenceError: Forward pass or tensor computation fails during inference.
    GradCAMGenerationError: Feature layer hooks or visualization encoding fails.
    PhysiologicalValidationError: Vital sign measurement is NaN, infinite, or unphysical.
    PhysiologicalTypeValidationError: Vital sign measurement is missing or non-numeric.
    ResourceExhaustionError: Memory limit or CUDA out-of-memory encountered.
    DeviceUnavailableError: Hardware device (e.g. CUDA) unavailable or unsupported.
    ConfigurationError: Pipeline configuration is invalid or missing required resources.
    CheckpointManagerError: Base domain exception for checkpoint lifecycle operations.
    IneligibleCheckpointError: Candidate checkpoint fails eligibility criteria.
    RollbackUnavailableError: No prior verified checkpoint exists in selection history.
    CheckpointVerificationError: Promoted checkpoint fails post-selection verification.
"""

from typing import List, Optional


class MLException(Exception):
    """Base domain exception for all Multimodal Health Screening ML pipeline errors."""

    def __init__(self, message: str, operation: str = "unknown"):
        super().__init__(message)
        self.message = message
        self.operation = operation


class CheckpointNotFoundError(FileNotFoundError, MLException):
    """Raised when model weights checkpoint file cannot be found on disk."""

    def __init__(self, message: str, operation: str = "load_checkpoint"):
        super().__init__(message)
        self.message = message
        self.operation = operation


class CheckpointCorruptError(IOError, MLException):
    """Raised when checkpoint file is empty, truncated, or failed unpickling."""

    def __init__(self, message: str, operation: str = "load_checkpoint"):
        super().__init__(message)
        self.message = message
        self.operation = operation


class CheckpointIncompatibleError(ValueError, RuntimeError, MLException):
    """Raised when checkpoint metadata, class list, or tensor shapes mismatch."""

    def __init__(self, message: str, operation: str = "load_checkpoint"):
        super().__init__(message)
        self.message = message
        self.operation = operation


class InvalidImageInputError(IOError, MLException):
    """Raised when an input image cannot be decoded, identified, or is empty."""

    def __init__(self, message: str, operation: str = "image_input"):
        super().__init__(message)
        self.message = message
        self.operation = operation


class QualityGateRejectionError(ValueError, MLException):
    """Raised when image quality validation fails in strict rejection mode."""

    def __init__(self, message: str, operation: str = "quality_gate", failure_reason: str = ""):
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.failure_reason = failure_reason


class ModelInferenceError(RuntimeError, MLException):
    """Raised when neural network forward pass fails during inference."""

    def __init__(self, message: str, operation: str = "inference"):
        super().__init__(message)
        self.message = message
        self.operation = operation


class GradCAMGenerationError(RuntimeError, MLException):
    """Raised when Grad-CAM feature capture or visualization encoding fails."""

    def __init__(self, message: str, operation: str = "gradcam"):
        super().__init__(message)
        self.message = message
        self.operation = operation


class PhysiologicalValidationError(ValueError, MLException):
    """Raised when physiological vital sign readings are NaN, infinite, or out of range."""

    def __init__(self, message: str, metric_name: str = "", operation: str = "physio_validation"):
        super().__init__(message)
        self.message = message
        self.metric_name = metric_name
        self.operation = operation


class PhysiologicalTypeValidationError(TypeError, MLException):
    """Raised when physiological vital sign readings are missing, boolean, or non-numeric."""

    def __init__(self, message: str, metric_name: str = "", operation: str = "physio_validation"):
        super().__init__(message)
        self.message = message
        self.metric_name = metric_name
        self.operation = operation


class ResourceExhaustionError(RuntimeError, MLException):
    """Raised when CUDA out-of-memory or system resource limits are reached."""

    def __init__(self, message: str, operation: str = "resource_allocation"):
        super().__init__(message)
        self.message = message
        self.operation = operation


class DeviceUnavailableError(RuntimeError, ValueError, MLException):
    """Raised when a requested hardware device (e.g. CUDA) cannot be utilized."""

    def __init__(self, message: str, requested_device: str = "", operation: str = "device_resolution"):
        super().__init__(message)
        self.message = message
        self.requested_device = requested_device
        self.operation = operation


class ConfigurationError(ValueError, MLException):
    """Raised when ML pipeline configuration is invalid or missing required resources."""

    def __init__(self, message: str, operation: str = "configuration_validation"):
        super().__init__(message)
        self.message = message
        self.operation = operation


class CheckpointManagerError(MLException):
    """Base domain exception for checkpoint management, selection, and promotion failures."""

    def __init__(self, message: str, operation: str = "checkpoint_management"):
        super().__init__(message, operation=operation)


class IneligibleCheckpointError(ValueError, CheckpointManagerError):
    """Raised when a candidate checkpoint fails eligibility criteria for consideration or promotion."""

    def __init__(
        self,
        message: str,
        candidate_path: str = "",
        reasons: Optional[List[str]] = None,
        operation: str = "eligibility_check",
    ):
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.candidate_path = candidate_path
        self.reasons = reasons or []


class RollbackUnavailableError(RuntimeError, CheckpointManagerError):
    """Raised when attempting a rollback but no prior verified checkpoint exists in history."""

    def __init__(
        self,
        message: str = "No previous verified checkpoint exists in selection history.",
        operation: str = "rollback",
    ):
        super().__init__(message)
        self.message = message
        self.operation = operation


class CheckpointVerificationError(RuntimeError, CheckpointManagerError):
    """Raised when a promoted or rolled-back checkpoint fails post-selection verification smoke tests."""

    def __init__(
        self,
        message: str,
        checkpoint_path: str = "",
        operation: str = "post_selection_verification",
    ):
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.checkpoint_path = checkpoint_path

