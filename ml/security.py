"""
ML Security and Defensive Hardening Module.

Provides security controls for:
1. Safe path validation (blocking null bytes, traversal attacks, and protection of critical project assets).
2. Atomic, crash-resilient file writes with guaranteed cleanup of temporary files on failure.
3. Strict protection against accidental deletion or overwriting of trained weights, source code, and manifests.
4. Error message and telemetry sanitization to prevent sensitive filesystem or credential disclosure.
"""

import json
import logging
import os
import re
from typing import Any, Dict, Optional, Set, Union
import uuid

logger = logging.getLogger(__name__)

# Critical project file extensions that must NEVER be overwritten via generic output sinks
PROTECTED_EXTENSIONS: Set[str] = {
    ".pth",
    ".pt",
    ".py",
    ".pyc",
    ".pyd",
    ".git",
    ".sh",
    ".bat",
    ".ps1",
    ".exe",
    ".dll",
}

# Critical filenames that must NEVER be overwritten via general I/O
PROTECTED_FILENAMES: Set[str] = {
    "split_manifest.json",
    "eye_model.pth",
    "eye_densenet201_best.pth",
    "recovery_checkpoint.pth",
    "requirements.txt",
    "pyproject.toml",
    "MODEL_CARD.md",
    "README.md",
}


def validate_safe_path(
    path: Any,
    allow_missing: bool = True,
    disallow_critical: bool = True,
    name: str = "Path",
) -> str:
    """
    Validates a filesystem path against security vulnerabilities.

    Checks:
    - Type verification (must be a non-empty string).
    - Rejection of embedded null bytes (\x00).
    - Prevention of accidental overwriting of protected project assets (.pth, .py, manifests).
    - Path resolution and directory validation.

    Args:
        path: Path string to validate.
        allow_missing: If False, verifies the file exists on disk.
        disallow_critical: If True, blocks targeting critical project files for modification/write.
        name: Parameter name for descriptive error reporting.

    Returns:
        Cleaned, stripped path string.

    Raises:
        TypeError: If path is not a string.
        ValueError: If path is empty, contains null bytes, or targets a protected file.
        FileNotFoundError: If allow_missing=False and file does not exist.
        IsADirectoryError: If path refers to an existing directory when a file is expected.
    """
    if not isinstance(path, str):
        raise TypeError(f"{name} must be a string, got {type(path).__name__}.")

    clean_path = path.strip()
    if not clean_path:
        raise ValueError(f"{name} cannot be empty or whitespace.")

    # 1. Null-byte injection check
    if "\x00" in clean_path:
        raise ValueError(f"{name} contains an invalid embedded null byte character.")

    norm_path = os.path.normpath(clean_path)
    base_name = os.path.basename(norm_path).lower()
    _, ext = os.path.splitext(norm_path)
    ext_lower = ext.lower()

    # 2. Critical file protection guard
    if disallow_critical:
        if base_name in PROTECTED_FILENAMES:
            raise ValueError(
                f"{name} targets a protected critical project file ('{base_name}'). "
                "Direct modification or overwriting is prohibited."
            )
        if ext_lower in PROTECTED_EXTENSIONS:
            raise ValueError(
                f"{name} specifies a protected file extension ('{ext_lower}'). "
                f"Generic outputs cannot overwrite {', '.join(sorted(PROTECTED_EXTENSIONS))} files."
            )

    # 3. File existence checks if required
    if not allow_missing:
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"{name} does not exist: {base_name}")
        if os.path.isdir(norm_path):
            raise IsADirectoryError(f"{name} points to a directory, not a regular file: {base_name}")

    return norm_path


def safe_atomic_write_file(
    target_path: str,
    data: Union[str, bytes],
    mode: str = "w",
    encoding: Optional[str] = "utf-8",
    make_dirs: bool = True,
    disallow_critical: bool = True,
) -> str:
    """
    Writes data to target_path atomically using a temporary file in the same directory.

    Guarantees:
    - Atomicity: The target file is either fully updated or left untouched.
    - Zero Partial Files: If writing fails or is interrupted, the temporary file is deleted.
    - Asset Protection: Prevents overwriting protected file types (.pth, .py, split_manifest.json).
    - Crash Resilience: Calls flush() and fsync() before renaming.

    Args:
        target_path: Destination file path.
        data: String or bytes to write.
        mode: 'w' for text, 'wb' for binary.
        encoding: Text encoding (default: 'utf-8'). Ignored if mode is binary.
        make_dirs: If True, creates parent directories safely.
        disallow_critical: If True, blocks targeting protected system/model files.

    Returns:
        Normalized target path on successful completion.
    """
    clean_target = validate_safe_path(target_path, allow_missing=True, disallow_critical=disallow_critical)

    if os.path.isdir(clean_target):
        raise IsADirectoryError(f"Target path is an existing directory: {os.path.basename(clean_target)}")

    parent_dir = os.path.dirname(os.path.abspath(clean_target))
    if make_dirs and parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    # Unique temporary file in the SAME directory ensures atomic rename on the same filesystem
    tmp_path = f"{clean_target}.tmp_{uuid.uuid4().hex}"

    try:
        if "b" in mode:
            with open(tmp_path, "wb") as f:
                if isinstance(data, str):
                    data = data.encode(encoding or "utf-8")
                f.write(data)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass  # Non-critical if fsync unsupported on filesystem
        else:
            with open(tmp_path, "w", encoding=encoding or "utf-8") as f:
                if isinstance(data, bytes):
                    data = data.decode(encoding or "utf-8")
                f.write(data)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

        # Atomic replacement
        try:
            os.replace(tmp_path, clean_target)
        except OSError:
            # Fallback for platforms with locking conflicts
            if os.path.exists(clean_target):
                os.remove(clean_target)
            os.replace(tmp_path, clean_target)

        return clean_target

    finally:
        # Guarantee no partial temporary files remain on disk after error
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def safe_atomic_write_json(
    target_path: str,
    data: Any,
    indent: int = 2,
    make_dirs: bool = True,
    disallow_critical: bool = True,
) -> str:
    """
    Atomically serializes and writes JSON data to disk with crash-resilience.
    """
    json_str = json.dumps(data, indent=indent, default=str)
    return safe_atomic_write_file(
        target_path=target_path,
        data=json_str,
        mode="w",
        encoding="utf-8",
        make_dirs=make_dirs,
        disallow_critical=disallow_critical,
    )


def sanitize_error_message(error: Union[str, Exception]) -> str:
    """
    Sanitizes error and exception messages to prevent exposing sensitive local paths,
    usernames, environment secrets, or detailed host filesystem hierarchies.

    Examples:
        "File not found at C:\\Users\\Aryan\\Documents\\test.jpg"
        -> "File not found at [PATH]\\test.jpg"
    """
    if error is None:
        return ""
    msg = str(error)

    # 1. Redact Windows user home paths (e.g. C:\Users\<Username>\...)
    msg = re.sub(r"[A-Za-z]:\\Users\\[^\s\\/]+(?:\\[^\s\\/]+)*\\([^\s\\/:]+)", r"[PATH]\\\1", msg)

    # 2. Redact POSIX user home paths (e.g. /home/<username>/...)
    msg = re.sub(r"/home/[^\s/]+(?:/[^\s/]+)*/([^\s/:]+)", r"[PATH]/\1", msg)

    # 3. Redact generic Windows absolute paths (e.g. C:\path\to\file.ext)
    msg = re.sub(r"[A-Za-z]:\\[^\s:<>\"|?*]+\\([^\s:<>\"|?*]+)", r"[PATH]\\\1", msg)

    # 4. Redact potential credential / API key patterns
    msg = re.sub(r"(?i)(api[_-]?key|token|secret|password|auth)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?", r"\1=[REDACTED]", msg)

    return msg
