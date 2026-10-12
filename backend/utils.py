"""
Utility helpers for file upload validation, decodability checks, sanitization, storage, and cleanup.
Includes magic byte signature verification and decompression bomb defense.
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
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_SIZE_MB,
    UPLOADS_DIR,
)
from backend.security import (
    validate_upload_subfolder,
    verify_image_signature,
)

logger = logging.getLogger("backend.utils")

# Enforce strict ceiling on pixel count to mitigate Decompression Bomb exploits
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


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


def validate_image_decodable(path_or_str: str) -> Tuple[bool, Optional[str]]:
    """
    Verifies that the file at the given path can be successfully decoded as a valid image
    and does not exceed safe dimensions (decompression bomb defense).

    Returns:
        Tuple of (is_valid: bool, error_reason: Optional[str])
    """
    try:
        with Image.open(path_or_str) as img:
            img.verify()

        # Reopen to check bounds (verify() closes the file pointer)
        with Image.open(path_or_str) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                return False, "Image has invalid zero or negative dimensions."
            total_pixels = width * height
            if total_pixels > MAX_IMAGE_PIXELS:
                return False, f"Image dimensions ({width}x{height} = {total_pixels} pixels) exceed safe limit ({MAX_IMAGE_PIXELS} pixels)."
        return True, None
    except Image.DecompressionBombError as dce:
        return False, f"Image exceeds maximum safe dimensions (decompression bomb protection): {dce}"
    except Exception as exc:
        return False, f"Uploaded file is corrupted or not a decodable image: {exc}"


async def save_uploaded_image(file: UploadFile, subfolder: str = "") -> Tuple[str, str]:
    """
    Validates file extension, magic byte header, upload size, and decodability,
    then saves it safely to disk with an unguessable UUID filename.

    Returns:
        Tuple of (absolute_file_path, relative_url_path)
    """
    filename = file.filename or ""
    # Strip any directory traversal components from original filename
    base_filename = os.path.basename(filename.replace("\\", "/"))
    ext = os.path.splitext(base_filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed extensions are: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Validate subfolder against traversal
    try:
        target_dir = validate_upload_subfolder(subfolder)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid upload location: {val_err}",
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    dest_path = target_dir / unique_filename

    total_bytes = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    first_chunk = True

    try:
        with open(dest_path, "wb") as buffer:
            while chunk := await file.read(chunk_size):
                total_bytes += len(chunk)

                # 1. Magic byte verification on initial payload
                if first_chunk:
                    is_valid_sig, detected_ext, sig_err = verify_image_signature(chunk, ext)
                    if not is_valid_sig:
                        buffer.close()
                        cleanup_file(str(dest_path))
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid file content: {sig_err}",
                        )
                    first_chunk = False

                # 2. Maximum file size check
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

    # 3. Validate image decodability & decompression bomb protection
    is_decodable, decode_err = validate_image_decodable(str(dest_path))
    if not is_decodable:
        cleanup_file(str(dest_path))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=decode_err or "Uploaded file is corrupted or not a decodable image.",
        )

    clean_sub = subfolder.strip().replace("\\", "/").strip("/")
    rel_path = f"/uploads/{clean_sub}/{unique_filename}" if clean_sub else f"/uploads/{unique_filename}"
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

        # Validate PNG magic bytes on decoded payload
        if not binary_data.startswith(b"\x89PNG\r\n\x1a\n"):
            logger.warning("Decoded Grad-CAM artifact is not a valid PNG image.")
            return ""

        out_name = f"gradcam_{uuid.uuid4().hex}.png"
        out_path = GRADCAM_DIR / out_name

        with open(out_path, "wb") as f:
            f.write(binary_data)

        return f"/uploads/gradcam/{out_name}"
    except Exception as exc:
        logger.warning(f"Failed to persist base64 Grad-CAM: {exc}")
        return ""
