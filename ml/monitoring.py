"""
ML Structured Logging, Telemetry, and Failure Observability Module.

Provides standard Python logging utilities for operational ML events:
- Model initialization & checkpoint loading
- Quality gate evaluations & rejections
- Preprocessing operations & decoding failures
- Neural network inference latency & outcomes
- Grad-CAM explainability generation & resource cleanup
- Physiological input validations & abnormal triage flags
- End-to-end multimodal health screening pipeline tracking

Privacy Safeguards:
- Strict sanitization: File paths are reduced to base filenames to avoid leaking local filesystem structure.
- Never logs raw images, Base64 strings, tensor arrays, patient identifiers, or health records.
- Masks request payloads and vital sign values to preserve patient confidentiality.

Configurability:
- configure_logging(): Configurable log level, console output, optional log file, and JSON format.
"""

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Union

# ML Root Logger
ml_logger = logging.getLogger("ml")

# Thread-safe ring buffer for test inspection and local telemetry audit
_MAX_AUDIT_EVENTS: int = 500
_EVENT_RING_BUFFER: Deque[Dict[str, Any]] = deque(maxlen=_MAX_AUDIT_EVENTS)
_BUFFER_LOCK = threading.Lock()
_CONFIG_LOCK = threading.Lock()


class ErrorCode:
    """Stable error codes for operational diagnostics and monitoring."""
    IMAGE_NOT_FOUND = "ERR_IMAGE_NOT_FOUND"
    QUALITY_REJECTED = "ERR_QUALITY_REJECTED"
    PREPROCESSING_FAILED = "ERR_PREPROCESSING_FAILED"
    CHECKPOINT_NOT_FOUND = "ERR_CHECKPOINT_NOT_FOUND"
    CHECKPOINT_CORRUPT = "ERR_CHECKPOINT_CORRUPT"
    CHECKPOINT_INCOMPATIBLE = "ERR_CHECKPOINT_INCOMPATIBLE"
    INFERENCE_FAILED = "ERR_INFERENCE_FAILED"
    NON_FINITE_OUTPUT = "ERR_NON_FINITE_OUTPUT"
    GRADCAM_FAILED = "ERR_GRADCAM_FAILED"
    VITALS_MISSING = "ERR_VITALS_MISSING"
    VITALS_INCOMPLETE = "ERR_VITALS_INCOMPLETE"
    VITALS_VALIDATION = "ERR_VITALS_VALIDATION"
    UNEXPECTED_EXCEPTION = "ERR_UNEXPECTED_EXCEPTION"


def sanitize_path(file_path: Optional[str]) -> Optional[str]:
    """
    Sanitizes filesystem paths by stripping directories and returning the basename.
    Prevents leakage of sensitive host paths, usernames, or directory structures.
    """
    if not file_path or not isinstance(file_path, str):
        return None
    return os.path.basename(file_path.strip())


def sanitize_details(details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deeply sanitizes metadata dictionaries to recursively remove PII, secrets, Base64 data,
    raw vitals, and absolute filesystem paths.
    """
    if not details or not isinstance(details, dict):
        return {}

    _SENSITIVE_KEYS = {
        "password", "secret", "token", "key", "auth", "credential",
        "api_key", "bearer", "private", "access_token"
    }
    _PII_KEYS = {
        "patient_id", "patient_name", "mrn", "ssn", "dob", "patient", "name"
    }
    _PHYSIO_MEASUREMENT_KEYS = {"heart_rate", "spo2", "temperature", "pulse", "bpm", "hr"}
    _IMAGE_DATA_KEYS = {"gradcam", "gradcam_image", "image_data", "raw_image", "b64_string", "base64", "heatmap", "photo"}

    def _sanitize_val(k_name: str, val: Any) -> Any:
        if isinstance(val, dict):
            return {sub_k: _sanitize_val(sub_k, sub_v) for sub_k, sub_v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_sanitize_val(k_name, item) for item in val]

        k_lower = str(k_name).lower()
        if any(s in k_lower for s in _SENSITIVE_KEYS):
            return "[REDACTED]"
        if any(p == k_lower for p in _PII_KEYS):
            return "[REDACTED_PII]"
        if any(img_k in k_lower for img_k in _IMAGE_DATA_KEYS):
            return "[BLOB_DATA_MASKED]"
        if any(phys_k in k_lower for phys_k in _PHYSIO_MEASUREMENT_KEYS) and isinstance(val, (int, float)):
            return "[PHYSIO_DATA_MASKED]"
        if isinstance(val, str):
            if len(val) > 150 or val.startswith("data:image"):
                return "[BLOB_DATA_MASKED]"
            if ("/" in val or "\\" in val) and not val.startswith("http"):
                return sanitize_path(val)
        return val

    sanitized: Dict[str, Any] = {}
    for k, v in details.items():
        sanitized[k] = _sanitize_val(k, v)
    return sanitized


from ml.security import sanitize_error_message  # re-export for monitoring callers



class LatencyTimer:
    """Helper object tracking high-resolution monotonic elapsed time in milliseconds."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0

    def start(self) -> "LatencyTimer":
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        self.duration_ms = max(0.0, round((self.end_time - self.start_time) * 1000.0, 3))
        return self.duration_ms


@contextmanager
def measure_ml_latency():
    """
    Context manager measuring execution latency in milliseconds via high-res monotonic timer.

    Usage:
        with measure_ml_latency() as timer:
            # execute operation
        print(f"Elapsed: {timer.duration_ms} ms")
    """
    timer = LatencyTimer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


# -----------------------------------------------------------------------------
# Structured Log Formatters
# -----------------------------------------------------------------------------

class MLStructuredFormatter(logging.Formatter):
    """
    Human-readable structured log formatter with consistent operational fields.
    Format: [timestamp] [LEVEL] [ML:stage] event=... status=... req_id=... duration=...ms error_code=... | details
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "ml_event", None)
        if not payload:
            return super().format(record)

        iso_ts = datetime.fromtimestamp(payload.get("timestamp", time.time()), tz=timezone.utc).isoformat()
        stage = payload.get("stage") or payload.get("event_type", "PIPELINE").lower()
        event_name = payload.get("event_name", payload.get("operation", "operation"))
        status = payload.get("status", record.levelname)

        parts = [
            f"[{iso_ts}]",
            f"[{record.levelname}]",
            f"[ML:{stage}]",
            f"event={event_name}",
            f"status={status}",
        ]

        if payload.get("request_id"):
            parts.append(f"req_id={payload['request_id']}")
        if payload.get("duration_ms") is not None:
            parts.append(f"duration={payload['duration_ms']:.2f}ms")
        if payload.get("error_code"):
            parts.append(f"error_code={payload['error_code']}")
        if payload.get("device"):
            parts.append(f"dev={payload['device']}")

        details = payload.get("details", {})
        if details:
            details_str = " ".join(f"{k}={v}" for k, v in details.items())
            return " ".join(parts) + " | " + details_str

        return " ".join(parts)


class MLJsonFormatter(logging.Formatter):
    """Machine-readable JSON log formatter for structured log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "ml_event", None)
        if not payload:
            base_payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                base_payload["exception"] = self.formatException(record.exc_info)
            return json.dumps(base_payload)

        json_dict = dict(payload)
        json_dict["level"] = record.levelname
        json_dict["logger"] = record.name
        json_dict["timestamp_iso"] = datetime.fromtimestamp(
            payload.get("timestamp", time.time()), tz=timezone.utc
        ).isoformat()
        if record.exc_info:
            json_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(json_dict)


# -----------------------------------------------------------------------------
# Core Structured Event Logging
# -----------------------------------------------------------------------------

def log_ml_event(
    logger: logging.Logger,
    event_type: str,
    operation: str,
    status: str,
    duration_ms: Optional[float] = None,
    device: Optional[str] = None,
    request_id: Optional[str] = None,
    stage: Optional[str] = None,
    event_name: Optional[str] = None,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    level: Optional[int] = None,
    exc_info: bool = False,
) -> Dict[str, Any]:
    """
    Emits a structured ML telemetry event through Python's logging module with privacy guards.

    Args:
        logger: Target logging.Logger instance.
        event_type: High-level event category (e.g., 'INFERENCE', 'QUALITY_GATE', 'SCREENING_PIPELINE').
        operation: Function or sub-process name (e.g., 'predict_eye_disease', 'run_screening').
        status: Operation outcome ('START' | 'SUCCESS' | 'FAILURE' | 'REJECTED' | 'WARNING').
        duration_ms: Latency in milliseconds (measured via high-resolution monotonic timer).
        device: Device identifier ('cpu', 'cuda', 'cuda:0').
        request_id: Correlation identifier for request tracing.
        stage: Pipeline stage name ('quality_gate', 'preprocessing', 'eye_model', 'gradcam', 'vitals', 'pipeline').
        event_name: Standardized event name (e.g., 'screening.started', 'quality_gate.evaluated').
        error_code: Stable diagnostic error code from ErrorCode.
        details: Metadata dictionary (automatically sanitized).
        level: Override logging level (DEBUG, INFO, WARNING, ERROR).
        exc_info: If True, attaches traceback.

    Returns:
        Structured event dictionary.
    """
    sanitized = sanitize_details(details)

    resolved_stage = stage or event_type.lower()
    resolved_event_name = event_name or f"{resolved_stage}.{operation}"

    event_payload: Dict[str, Any] = {
        "event_type": event_type,
        "event_name": resolved_event_name,
        "stage": resolved_stage,
        "operation": operation,
        "status": status,
        "timestamp": time.time(),
    }
    if request_id:
        event_payload["request_id"] = str(request_id)
    if duration_ms is not None:
        event_payload["duration_ms"] = max(0.0, round(float(duration_ms), 3))
    if device is not None:
        event_payload["device"] = str(device)
    if error_code is not None:
        event_payload["error_code"] = str(error_code)
    if sanitized:
        event_payload["details"] = sanitized

    # Construct human-readable message string
    msg_parts: List[str] = [f"[ML:{event_type}] op={operation} status={status}"]
    if request_id:
        msg_parts.append(f"req_id={request_id}")
    if duration_ms is not None:
        msg_parts.append(f"duration={duration_ms:.2f}ms")
    if error_code:
        msg_parts.append(f"err={error_code}")
    if device:
        msg_parts.append(f"dev={device}")
    if sanitized:
        items_str = " ".join(f"{k}={v}" for k, v in sanitized.items())
        msg_parts.append(items_str)

    log_message = " | ".join(msg_parts)

    # Determine default logging level based on status
    if level is not None:
        chosen_level = level
    elif status in ("FAILURE", "ERROR"):
        chosen_level = logging.ERROR
    elif status in ("REJECTED", "WARNING"):
        chosen_level = logging.WARNING
    elif status == "START":
        chosen_level = logging.DEBUG
    else:
        chosen_level = logging.INFO

    # Safe log emission: logging failures must NEVER crash caller model execution
    try:
        logger.log(chosen_level, log_message, extra={"ml_event": event_payload}, exc_info=exc_info)
    except Exception:
        pass

    # Record in local in-memory audit ring buffer
    with _BUFFER_LOCK:
        _EVENT_RING_BUFFER.append(event_payload)

    return event_payload


def get_recent_ml_events(
    event_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Retrieves recent ML audit events from the thread-safe ring buffer."""
    with _BUFFER_LOCK:
        events = list(_EVENT_RING_BUFFER)

    if event_type:
        events = [e for e in events if e.get("event_type") == event_type or e.get("event_name") == event_type]
    if status:
        events = [e for e in events if e.get("status") == status]
    if limit:
        events = events[-limit:]

    return events


def clear_ml_event_buffer() -> None:
    """Clears the internal in-memory ML telemetry event buffer."""
    with _BUFFER_LOCK:
        _EVENT_RING_BUFFER.clear()


# -----------------------------------------------------------------------------
# Observability Configuration
# -----------------------------------------------------------------------------

def configure_logging(
    level: Union[int, str] = logging.INFO,
    log_to_console: bool = True,
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Configures structured logging for the Multimodal Health Screening ML layer.

    Features:
    - Thread-safe, idempotent initialization (avoids duplicate handlers).
    - Customizable log level (e.g. DEBUG, INFO, WARNING, ERROR).
    - Console output with structured human-readable or JSON formatting.
    - Optional persistent local log file with automatic directory creation.
    - Privacy safeguards: logs are stripped of PII and raw image blobs.

    Args:
        level: Logging level (int or string name like "INFO", "DEBUG").
        log_to_console: Whether to attach stdout stream handler.
        log_file: Optional path to a local log file.
        json_format: If True, uses JSON-lines formatting instead of standard text.

    Returns:
        The configured 'ml' logger.
    """
    with _CONFIG_LOCK:
        # Resolve level
        if isinstance(level, str):
            resolved_level = getattr(logging, level.upper(), logging.INFO)
        else:
            resolved_level = int(level)

        ml_logger.setLevel(resolved_level)

        # Select formatter
        formatter: logging.Formatter
        if json_format:
            formatter = MLJsonFormatter()
        else:
            formatter = MLStructuredFormatter()

        # Remove existing ML handlers to avoid duplication on repeated initialization
        existing_handlers = list(ml_logger.handlers)
        for h in existing_handlers:
            if getattr(h, "_is_ml_managed", False):
                ml_logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass

        # 1. Console Handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(resolved_level)
            console_handler.setFormatter(formatter)
            setattr(console_handler, "_is_ml_managed", True)
            ml_logger.addHandler(console_handler)

        # 2. File Handler
        if log_file:
            abs_log_path = os.path.abspath(log_file)
            log_dir = os.path.dirname(abs_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(abs_log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(resolved_level)
            file_handler.setFormatter(formatter)
            setattr(file_handler, "_is_ml_managed", True)
            ml_logger.addHandler(file_handler)

        return ml_logger
