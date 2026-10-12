"""
Utility helpers for file upload validation, decodability checks, sanitization, storage, and cleanup.
"""

import base64
import logging
import os
from pathlib import Path
import uuid
from typing import Optional, Tuple
from fastapi import HTTPException, UploadFile, status
from PIL import Image

from backend.config import (
    ALLOWED_EXTENSIONS,
    GRADCAM_DIR,
    MAX_FILE_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    UPLOADS_DIR,
)

logger = logging.getLogger("backend.utils")


def cleanup_file(path_or_str: Optional[str]) -> bool:
    """
    Safely removes a file from disk if it exists.
    Returns True if removed, False otherwise.
    """
    if not path_or_str:
        return False
    try:
        p = Path(path_or_str)
        if p.is_file():
            p.unlink()
            return True
    except Exception as exc:
        logger.warning(f"Failed to cleanup file {path_or_str}: {exc}")
    return False


def validate_image_decodable(path_or_str: str) -> bool:
    """
    Verifies that the file at the given path can be successfully decoded as a valid image.
    """
    try:
        with Image.open(path_or_str) as img:
            img.verify()
        return True
    except Exception:
        return False


async def save_uploaded_image(file: UploadFile, subfolder: str = "") -> Tuple[str, str]:
    """
    Validates file extension, size, and decodability, then saves it safely to disk.

    Returns:
        Tuple of (absolute_file_path, relative_url_path)
    """
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed extensions are: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    target_dir = UPLOADS_DIR / subfolder if subfolder else UPLOADS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    unique_filename = f"{uuid.uuid4().hex}{ext}"
    dest_path = target_dir / unique_filename

    total_bytes = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    try:
        with open(dest_path, "wb") as buffer:
            while chunk := await file.read(chunk_size):
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    buffer.close()
                    cleanup_file(str(dest_path))
                    raise HTTPException(
                        status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                        detail=f"Uploaded file exceeds maximum limit of {MAX_UPLOAD_SIZE_MB} MB.",
                    )
                buffer.write(chunk)
    except HTTPException:
        cleanup_file(str(dest_path))
        raise
    except Exception as exc:
        cleanup_file(str(dest_path))
        logger.error(f"Error saving uploaded file: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file safely.",
        )

    # Validate image decodability
    if not validate_image_decodable(str(dest_path)):
        cleanup_file(str(dest_path))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is corrupted or not a decodable image.",
        )

    rel_path = f"/uploads/{subfolder}/{unique_filename}" if subfolder else f"/uploads/{unique_filename}"
    return str(dest_path), rel_path


def save_base64_gradcam(base64_str: str) -> str:
    """
    Saves a base64 encoded PNG string to the gradcam uploads directory.

    Returns:
        Relative web-accessible path (e.g., /uploads/gradcam/<id>.png)
    """
    if not base64_str:
        return ""

    try:
        if "base64," in base64_str:
            raw_data = base64_str.split("base64,")[1]
        else:
            raw_data = base64_str

        binary_data = base64.b64decode(raw_data)
        out_name = f"gradcam_{uuid.uuid4().hex}.png"
        out_path = GRADCAM_DIR / out_name

        with open(out_path, "wb") as f:
            f.write(binary_data)

        return f"/uploads/gradcam/{out_name}"
    except Exception as exc:
        logger.warning(f"Failed to persist base64 Grad-CAM: {exc}")
        return ""
