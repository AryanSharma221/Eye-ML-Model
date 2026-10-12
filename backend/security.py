"""
Backend Security Hardening and Defensive Controls Module.

Provides production-grade defensive controls:
1. SecurityHeadersMiddleware: OWASP-recommended HTTP security headers (nosniff, DENY, CSP, etc.).
2. RequestSizeLimitMiddleware: Transport-level HTTP request size validation blocking oversized payloads upfront.
3. InMemoryRateLimiter / RateLimiterMiddleware: Sliding-window rate limiter protecting heavy ML inference routes against DoS.
4. Image Signature & Magic Byte Verification: Validates file headers (JPEG/PNG) preventing file extension spoofing.
5. Decompression Bomb Defense: Strict image pixel threshold ceiling preventing memory-exhaustion exploits.
6. Data Privacy & Storage Lifecycle: Secure physical file deletion utilities for patient and session data.
"""

import html
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.config import (
    ALLOWED_EXTENSIONS,
    ENABLE_RATE_LIMIT,
    MAX_FILE_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    RATE_LIMIT_PER_MINUTE,
    UPLOADS_DIR,
)

logger = logging.getLogger("backend.security")

# Magic byte signatures for authorized retinal fundus image formats
MAGIC_BYTE_SIGNATURES: Dict[str, List[bytes]] = {
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
}

# Whitelist of allowed subfolders under UPLOADS_DIR
ALLOWED_UPLOAD_SUBFOLDERS: Set[str] = {"fundus", "gradcam", ""}

# Safe upper limit for image dimensions (25 megapixels)
MAX_IMAGE_PIXELS_CEILING: int = 25_000_000


def verify_image_signature(file_bytes: bytes, file_ext: str = "") -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Detects the true image format from header magic bytes.
    Does not trust client-supplied file extensions or MIME headers alone.

    Returns:
        Tuple of (is_valid: bool, detected_ext: Optional[str], error_reason: Optional[str])
    """
    if file_bytes.startswith(b"\xff\xd8\xff"):
        return True, ".jpg", None
    if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return True, ".png", None

    return False, None, "File content does not match expected image signature (JPEG or PNG required)."


def validate_upload_subfolder(subfolder: str) -> Path:
    """
    Validates and resolves target upload subfolder, strictly preventing path traversal attacks.
    """
    clean_sub = (subfolder or "").strip().replace("\\", "/").strip("/")
    # Check against whitelist
    if clean_sub not in ALLOWED_UPLOAD_SUBFOLDERS:
        raise ValueError(f"Unauthorized upload subfolder '{clean_sub}'.")

    target = (UPLOADS_DIR / clean_sub).resolve()
    uploads_resolved = UPLOADS_DIR.resolve()

    if not str(target).startswith(str(uploads_resolved)):
        raise ValueError("Upload path traversal attempt detected.")

    return target


class SecurityHeadersMiddleware:
    """
    ASGI middleware that injects OWASP-recommended HTTP security headers into all responses:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Referrer-Policy: strict-origin-when-cross-origin
    - Content-Security-Policy: restrict resource loading and disallow embedding
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["X-XSS-Protection"] = "1; mode=block"
                headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                headers["Content-Security-Policy"] = (
                    "default-src 'self'; "
                    "img-src 'self' data: blob:; "
                    "style-src 'self' 'unsafe-inline'; "
                    "script-src 'self'; "
                    "frame-ancestors 'none';"
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)


class RequestSizeLimitMiddleware:
    """
    ASGI middleware that validates the HTTP Content-Length header upfront,
    rejecting requests larger than MAX_FILE_SIZE_BYTES before reading body into memory.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_FILE_SIZE_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            content_length = headers.get(b"content-length")
            if content_length:
                try:
                    length = int(content_length.decode("latin1"))
                    if length > self.max_bytes:
                        logger.warning(
                            f"Rejected oversized request ({length} bytes > {self.max_bytes} bytes limit)."
                        )
                        response = JSONResponse(
                            status_code=413,
                            content={
                                "status": "error",
                                "error_type": "PayloadTooLarge",
                                "message": (
                                    f"Uploaded file exceeds maximum limit of "
                                    f"{self.max_bytes // (1024 * 1024)} MB."
                                ),
                                "detail": (
                                    f"Uploaded file exceeds maximum limit of "
                                    f"{self.max_bytes // (1024 * 1024)} MB."
                                ),
                            },
                        )
                        await response(scope, receive, send)
                        return
                except (ValueError, UnicodeDecodeError):
                    pass

        await self.app(scope, receive, send)


class InMemoryRateLimiter:
    """
    Thread-safe in-memory sliding-window rate limiter.
    Tracks client requests by IP address within a configurable sliding window.
    """

    def __init__(self, default_rpm: int = RATE_LIMIT_PER_MINUTE, window_seconds: float = 60.0):
        self.default_rpm = default_rpm
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._records: Dict[str, List[float]] = {}
        self._heavy_endpoints: Set[str] = {
            "/api/v1/predict_eye",
            "/api/v1/screen",
            "/api/v1/submit_vitals",
        }

    def is_allowed(self, client_ip: str, path: str) -> Tuple[bool, int]:
        """
        Evaluates whether request from client_ip is permitted.
        Returns: (is_allowed, retry_after_seconds)
        """
        if self.default_rpm <= 0:
            return True, 0

        # Dedicated tighter limit for resource-intensive ML screening endpoints
        max_rpm = (
            min(self.default_rpm, 30)
            if any(path.startswith(p) for p in self._heavy_endpoints)
            else self.default_rpm
        )

        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            timestamps = self._records.setdefault(client_ip, [])
            # Evict timestamps outside current window
            valid_timestamps = [t for t in timestamps if t > window_start]
            self._records[client_ip] = valid_timestamps

            if len(valid_timestamps) >= max_rpm:
                oldest_in_window = valid_timestamps[0]
                retry_after = max(1, int(math.ceil(self.window_seconds - (now - oldest_in_window))))
                return False, retry_after

            valid_timestamps.append(now)
            return True, 0

    def cleanup_expired(self) -> None:
        """Periodically cleans up old records to prevent memory accumulation."""
        now = time.time()
        cutoff = now - (self.window_seconds * 2)
        with self._lock:
            expired_keys = [ip for ip, times in self._records.items() if not times or times[-1] < cutoff]
            for ip in expired_keys:
                del self._records[ip]


class RateLimiterMiddleware:
    """
    ASGI middleware enforcing rate limiting on incoming HTTP requests.
    Exempts static assets and health check endpoints.
    """

    def __init__(self, app: ASGIApp, limiter: InMemoryRateLimiter):
        self.app = app
        self.limiter = limiter
        self._exempt_prefixes = ("/health", "/docs", "/redoc", "/openapi.json", "/uploads")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not ENABLE_RATE_LIMIT:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if any(path.startswith(prefix) for prefix in self._exempt_prefixes):
            await self.app(scope, receive, send)
            return

        # Determine client IP address
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"

        # Check if this is a local client without an explicit rate-limit testing header
        raw_headers = dict(scope.get("headers", []))
        is_testing_rate_limit = (
            b"x-test-rate-limit" in raw_headers
            or raw_headers.get(b"x-test-rate-limit") == b"true"
        )
        if client_ip in ("127.0.0.1", "::1", "localhost", "testclient") and not is_testing_rate_limit:
            await self.app(scope, receive, send)
            return

        allowed, retry_after = self.limiter.is_allowed(client_ip, path)
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for IP '{client_ip}' on '{path}'. Retry-After: {retry_after}s"
            )
            response = JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "error_type": "RateLimitExceeded",
                    "message": "Too many requests. Please wait before submitting additional screening queries.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def sanitize_html_text(text: Optional[str]) -> str:
    """Sanitizes user-supplied text to prevent Stored XSS in HTML report generation."""
    if text is None:
        return ""
    return html.escape(str(text), quote=True)


def delete_physical_file(relative_or_absolute_path: Optional[str]) -> bool:
    """
    Securely unlinks a physical file within UPLOADS_DIR, blocking path traversal.
    """
    if not relative_or_absolute_path:
        return False
    try:
        clean_path = relative_or_absolute_path.strip().replace("\\", "/")
        if clean_path.startswith("/uploads/"):
            clean_path = clean_path[len("/uploads/"):]

        full_path = (UPLOADS_DIR / clean_path).resolve()
        uploads_resolved = UPLOADS_DIR.resolve()

        if not str(full_path).startswith(str(uploads_resolved)):
            logger.warning(f"Path traversal blocked during file deletion: {relative_or_absolute_path}")
            return False

        if full_path.is_file():
            full_path.unlink()
            return True
    except Exception as exc:
        logger.warning(f"Failed to delete physical file {relative_or_absolute_path}: {exc}")
    return False
