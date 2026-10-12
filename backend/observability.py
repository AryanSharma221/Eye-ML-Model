"""
Production Observability, Structured Logging, and Middleware for Backend Service.

Provides:
- Machine-readable JSON logging formatter and human-readable text formatter.
- Safe privacy guards: zero image binary, Base64 data, patient PII, or credential logging.
- Request correlation ID propagation (X-Request-ID) in headers and state.
- High-resolution request and sub-stage latency tracking.
- Structured HTTP access events and in-memory ring buffer for telemetry inspection.
"""

from collections import deque
from datetime import datetime, timezone
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Dedicated backend logger
backend_logger = logging.getLogger("backend")

# Thread-safe ring buffer for HTTP telemetry events
_MAX_HTTP_EVENTS: int = 500
_HTTP_EVENT_RING_BUFFER: Deque[Dict[str, Any]] = deque(maxlen=_MAX_HTTP_EVENTS)
_HTTP_BUFFER_LOCK = threading.Lock()
_LOG_CONFIG_LOCK = threading.Lock()


class BackendStructuredJsonFormatter(logging.Formatter):
    """
    Structured JSON log formatter for cloud aggregators (Render, Datadog, CloudWatch, GCP).
    Emits single-line JSON with standard operational fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge HTTP event data if attached
        http_event = getattr(record, "http_event", None)
        if http_event and isinstance(http_event, dict):
            for k, v in http_event.items():
                payload[k] = v

        # Merge ML event data if attached
        ml_event = getattr(record, "ml_event", None)
        if ml_event and isinstance(ml_event, dict):
            for k, v in ml_event.items():
                if k not in payload:
                    payload[k] = v

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class BackendStructuredTextFormatter(logging.Formatter):
    """
    Human-readable structured text log formatter for local development console.
    Format: [timestamp] [LEVEL] [logger] [req_id] [endpoint] status=... duration=...ms | message
    """

    def format(self, record: logging.LogRecord) -> str:
        iso_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        parts = [f"[{iso_ts}]", f"[{record.levelname:<7}]", f"[{record.name}]"]

        http_event = getattr(record, "http_event", None)
        if http_event and isinstance(http_event, dict):
            req_id = http_event.get("request_id")
            if req_id:
                parts.append(f"[{req_id[:8]}]")
            method = http_event.get("method")
            path = http_event.get("path")
            if method and path:
                parts.append(f"{method} {path}")
            status_code = http_event.get("status_code")
            if status_code is not None:
                parts.append(f"status={status_code}")
            duration = http_event.get("duration_ms")
            if duration is not None:
                parts.append(f"duration={duration:.2f}ms")
            timings = http_event.get("ml_timings_ms")
            if timings:
                t_str = " ".join(f"{k}={v}ms" for k, v in timings.items() if v is not None)
                parts.append(f"[{t_str}]")

        parts.append(f"| {record.getMessage()}")

        formatted = " ".join(parts)
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        return formatted


def configure_backend_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
) -> None:
    """
    Configures backend structured logging idempotently.
    Supports 'json' and 'text' formats.
    """
    with _LOG_CONFIG_LOCK:
        resolved_level = getattr(logging, level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        backend_logger.setLevel(resolved_level)

        if log_format.lower() == "text":
            formatter = BackendStructuredTextFormatter()
        else:
            formatter = BackendStructuredJsonFormatter()

        # Remove existing managed handlers to prevent duplication
        for h in list(backend_logger.handlers):
            if getattr(h, "_is_backend_managed", False):
                backend_logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass

        # Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        setattr(console_handler, "_is_backend_managed", True)
        backend_logger.addHandler(console_handler)

        # Optional File Handler
        if log_file:
            abs_log = os.path.abspath(log_file)
            os.makedirs(os.path.dirname(abs_log), exist_ok=True)
            file_handler = logging.FileHandler(abs_log, mode="a", encoding="utf-8")
            file_handler.setLevel(resolved_level)
            file_handler.setFormatter(formatter)
            setattr(file_handler, "_is_backend_managed", True)
            backend_logger.addHandler(file_handler)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    1. Extracts or generates correlation X-Request-ID.
    2. Measures high-resolution request duration.
    3. Emits structured operational HTTP access log event.
    4. Attaches X-Request-ID header to all responses.
    5. Safeguards privacy: never logs request bodies, images, or patient PII.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Extract existing X-Request-ID or generate new UUID4
        incoming_id = request.headers.get("x-request-id") or request.headers.get("X-Request-ID")
        if incoming_id and len(incoming_id.strip()) > 0 and len(incoming_id.strip()) <= 128:
            request_id = incoming_id.strip()
        else:
            request_id = uuid.uuid4().hex

        # Populate request state
        request.state.request_id = request_id
        request.state.ml_timings = None
        request.state.error_category = None

        start_time = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as exc:
            status_code = 500
            request.state.error_category = type(exc).__name__
            raise
        finally:
            duration_ms = max(0.0, round((time.perf_counter() - start_time) * 1000.0, 2))

            # Sanitize path to exclude query string parameters (prevents leaking tokens/PII)
            sanitized_path = request.url.path

            ml_timings = getattr(request.state, "ml_timings", None)
            error_category = getattr(request.state, "error_category", None)

            http_event: Dict[str, Any] = {
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": sanitized_path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            }

            if ml_timings:
                http_event["ml_timings_ms"] = ml_timings
            if error_category:
                http_event["error_category"] = error_category

            # Determine log severity based on HTTP status code
            if status_code >= 500:
                log_level = logging.ERROR
                msg = f"HTTP {request.method} {sanitized_path} failed with {status_code} ({duration_ms:.2f}ms)"
            elif status_code >= 400:
                log_level = logging.WARNING
                msg = f"HTTP {request.method} {sanitized_path} responded {status_code} ({duration_ms:.2f}ms)"
            else:
                log_level = logging.INFO
                msg = f"HTTP {request.method} {sanitized_path} completed {status_code} ({duration_ms:.2f}ms)"

            # Emit structured log
            backend_logger.log(log_level, msg, extra={"http_event": http_event})

            # Append to in-memory event buffer for test inspection & operational probes
            with _HTTP_BUFFER_LOCK:
                _HTTP_EVENT_RING_BUFFER.append(http_event)


def get_recent_http_events(
    status_code: Optional[int] = None,
    path: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Retrieves recent HTTP access events from the in-memory ring buffer."""
    with _HTTP_BUFFER_LOCK:
        events = list(_HTTP_EVENT_RING_BUFFER)

    if status_code is not None:
        events = [e for e in events if e.get("status_code") == status_code]
    if path is not None:
        events = [e for e in events if e.get("path") == path]
    if limit is not None:
        events = events[-limit:]

    return events


def clear_http_event_buffer() -> None:
    """Clears the internal in-memory HTTP telemetry ring buffer."""
    with _HTTP_BUFFER_LOCK:
        _HTTP_EVENT_RING_BUFFER.clear()
