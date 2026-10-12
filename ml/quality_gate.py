"""
Image Quality Gate Module for Retinal Fundus Screening.

This module validates that an incoming eye photograph satisfies structural, focus,
lighting, and contrast criteria before it is passed to downstream neural network
models for disease classification.

Public Contract:
    check_image_quality(image_path: str, request_id: Optional[str] = None) -> dict
"""

import logging
import math
import os
from typing import Any, Dict, Optional
import cv2
import numpy as np

from ml.config import SUPPORTED_IMAGE_EXTENSIONS
from ml.monitoring import log_ml_event, measure_ml_latency, sanitize_path

logger = logging.getLogger(__name__)


MAX_IMAGE_FILE_SIZE_BYTES: int = 30 * 1024 * 1024  # 30 MB maximum file size
MAX_IMAGE_DIMENSION: int = 5000                     # 5000x5000 maximum resolution


def check_image_quality(
    image_path: str,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validates a retinal fundus image against resolution, blur, illumination,
    contrast, and retinal contour criteria.

    Args:
        image_path: Path to the target image file on disk.
        request_id: Optional request correlation identifier for telemetry.

    Returns:
        dict containing exactly seven keys:
            - overall_pass (bool): True only if all individual criteria pass.
            - blur_score (float): Laplacian variance (threshold: >= 100.0).
            - brightness_mean (float): Mean grayscale intensity (interval: [30.0, 225.0]).
            - contrast_std (float): Standard deviation of grayscale intensity (threshold: >= 20.0).
            - retinal_region_detected (bool): Whether Hough Circle heuristic detected a circular boundary.
            - resolution_ok (bool): True if width >= 224 and height >= 224.
            - error_message (str | None): First failing reason by priority order, or None if all pass.

    Sentinel Convention:
        For unreadable, missing, or corrupt image files where measurements cannot be
        computed, numeric metrics are set to sentinel 0.0, boolean checks to False,
        and overall_pass to False with a descriptive error_message.
    """
    with measure_ml_latency() as timer:
        # Sentinel response builder for uncomputable inputs
        def _sentinel_failure(message: str) -> Dict[str, Any]:
            timer.stop()
            log_ml_event(
                logger,
                event_type="QUALITY_GATE",
                operation="check_image_quality",
                status="REJECTED",
                duration_ms=timer.duration_ms,
                request_id=request_id,
                details={
                    "file": sanitize_path(image_path),
                    "reason": message,
                    "overall_pass": False,
                },
            )
            return {
                "overall_pass": False,
                "blur_score": 0.0,
                "brightness_mean": 0.0,
                "contrast_std": 0.0,
                "retinal_region_detected": False,
                "resolution_ok": False,
                "error_message": message,
            }

        # 1. Input Path & File Validation
        if not isinstance(image_path, str) or not image_path.strip():
            return _sentinel_failure("Image path must be a non-empty string.")

        if "\x00" in image_path:
            return _sentinel_failure("Image path contains invalid null byte character.")

        clean_name = sanitize_path(image_path) or "unknown"

        if not os.path.exists(image_path):
            return _sentinel_failure(f"File not found at specified path: {clean_name}")

        if os.path.isdir(image_path) or not os.path.isfile(image_path):
            return _sentinel_failure(f"Path does not point to a regular file: {clean_name}")

        _, ext = os.path.splitext(image_path)
        if ext.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return _sentinel_failure(
                f"Failed to decode image: unsupported file format '{ext}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}."
            )

        try:
            file_size = os.path.getsize(image_path)
        except OSError as err:
            return _sentinel_failure(f"Unable to read file metadata: {err}")

        if file_size == 0:
            return _sentinel_failure("Image file is empty (0 bytes).")

        if file_size > MAX_IMAGE_FILE_SIZE_BYTES:
            return _sentinel_failure(
                f"Image file exceeds maximum allowable size of 30 MB ({file_size / (1024*1024):.1f} MB)."
            )

        # 2. Image Decoding via OpenCV
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        except Exception as err:
            return _sentinel_failure(f"OpenCV failed to decode image: {err}")

        if image is None:
            return _sentinel_failure(
                "Failed to decode image. File may be corrupt, truncated, or in an unsupported format."
            )

        if image.size == 0 or len(image.shape) < 2:
            return _sentinel_failure("Decoded image contains no pixel data.")

        if not np.isfinite(image).all():
            return _sentinel_failure("Decoded image contains non-finite pixel values (NaN or Inf).")

        # 3. Resolution Verification
        height, width = int(image.shape[0]), int(image.shape[1])
        if width <= 0 or height <= 0:
            return _sentinel_failure(f"Invalid image dimensions ({width}x{height}): {clean_name}")

        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return _sentinel_failure(
                f"Image dimensions ({width}x{height}) exceed maximum allowable resolution ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})."
            )

        resolution_ok = bool(width >= 224 and height >= 224)
        res_error = (
            None
            if resolution_ok
            else f"Insufficient resolution ({width}x{height}). Minimum required is 224x224."
        )

        # Convert to grayscale for single-channel photometric metrics
        try:
            if len(image.shape) == 3 and image.shape[2] >= 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 2:
                gray = image
            else:
                return _sentinel_failure(f"Unexpected image channel structure: {image.shape}")
        except Exception as err:
            return _sentinel_failure(f"Failed converting image to grayscale: {err}")

        # 4. Blur Measurement (Laplacian Variance)
        try:
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if not math.isfinite(laplacian_var):
                laplacian_var = 0.0
        except Exception:
            laplacian_var = 0.0

        blur_score = round(laplacian_var, 2)
        blur_pass = bool(blur_score >= 100.0)
        blur_error = (
            None
            if blur_pass
            else f"Excessive image blur (blur score {blur_score:.2f} < threshold 100.0)."
        )

        # 5. Brightness Measurement (Grayscale Mean)
        try:
            mean_val = float(np.mean(gray))
            if not math.isfinite(mean_val):
                mean_val = 0.0
        except Exception:
            mean_val = 0.0

        brightness_mean = round(mean_val, 2)
        brightness_pass = bool(30.0 <= brightness_mean <= 225.0)
        if brightness_pass:
            brightness_error = None
        elif brightness_mean < 30.0:
            brightness_error = f"Image underexposed/too dark (brightness {brightness_mean:.2f} < minimum 30.0)."
        else:
            brightness_error = f"Image overexposed/too bright (brightness {brightness_mean:.2f} > maximum 225.0)."

        # 6. Contrast Measurement (Grayscale Standard Deviation)
        try:
            std_val = float(np.std(gray))
            if not math.isfinite(std_val):
                std_val = 0.0
        except Exception:
            std_val = 0.0

        contrast_std = round(std_val, 2)
        contrast_pass = bool(contrast_std >= 20.0)
        contrast_error = (
            None
            if contrast_pass
            else f"Insufficient contrast (contrast std {contrast_std:.2f} < threshold 20.0)."
        )

        # 7. Retinal-Region Detection (Hough Circle Heuristic)
        retinal_region_detected = False
        retinal_error = None
        try:
            resized_gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
            circles = cv2.HoughCircles(
                resized_gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=100,
                param1=50,
                param2=30,
                minRadius=40,
                maxRadius=112,
            )
            if circles is not None and len(circles) > 0 and circles[0] is not None and len(circles[0]) > 0:
                retinal_region_detected = True
            else:
                retinal_error = "No circular retinal boundary detected via Hough transform heuristic."
        except Exception as err:
            retinal_region_detected = False
            retinal_error = f"Retinal region detection failed during Hough transform: {err}"

        # 8. Final Decision & Deterministic Error Prioritization
        overall_pass = bool(
            resolution_ok
            and blur_pass
            and brightness_pass
            and contrast_pass
            and retinal_region_detected
        )

        if overall_pass:
            chosen_error = None
        elif not resolution_ok:
            chosen_error = res_error
        elif not blur_pass:
            chosen_error = blur_error
        elif not brightness_pass:
            chosen_error = brightness_error
        elif not contrast_pass:
            chosen_error = contrast_error
        elif not retinal_region_detected:
            chosen_error = retinal_error
        else:
            chosen_error = "Quality gate criteria not satisfied."

        timer.stop()
        log_ml_event(
            logger,
            event_type="QUALITY_GATE",
            operation="check_image_quality",
            status="SUCCESS" if overall_pass else "REJECTED",
            duration_ms=timer.duration_ms,
            request_id=request_id,
            details={
                "file": sanitize_path(image_path),
                "overall_pass": overall_pass,
                "blur_score": blur_score,
                "brightness_mean": brightness_mean,
                "contrast_std": contrast_std,
                "reason": chosen_error,
            },
        )

        return {
            "overall_pass": overall_pass,
            "blur_score": blur_score,
            "brightness_mean": brightness_mean,
            "contrast_std": contrast_std,
            "retinal_region_detected": retinal_region_detected,
            "resolution_ok": resolution_ok,
            "error_message": chosen_error,
        }
