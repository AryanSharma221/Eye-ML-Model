"""
Image Preprocessing Module for Retinal Fundus Screening.

This module converts incoming retinal fundus images into standardized, normalized
PyTorch tensors suitable for DenseNet201 deep learning model inference.

Public Contract:
    preprocess_image(image_path: str, request_id: Optional[str] = None) -> torch.Tensor
"""

import logging
import os
from typing import Optional, Sequence
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

from ml.config import SUPPORTED_IMAGE_EXTENSIONS
from ml.exceptions import InvalidImageInputError
from ml.monitoring import log_ml_event, measure_ml_latency, sanitize_path

logger = logging.getLogger(__name__)

# Standard ImageNet normalization parameters required by torchvision models
IMAGENET_MEAN: Sequence[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: Sequence[float] = [0.229, 0.224, 0.225]

# Maximum file size and dimension guardrails for security and DoS prevention
MAX_IMAGE_FILE_SIZE_BYTES: int = 30 * 1024 * 1024  # 30 MB
MAX_IMAGE_DIMENSION: int = 5000                     # 5000x5000 pixels
Image.MAX_IMAGE_PIXELS = 50_000_000                 # 50 megapixels decompression limit

# Transformation pipeline: Resize to 224x224 -> Convert to Tensor -> Normalize
PREPROCESSING_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_image_transforms(is_train: bool = False) -> transforms.Compose:
    """
    Returns standard image transformation pipeline for training or evaluation.

    Args:
        is_train: If True, applies data augmentations (flips, rotation, color jitter).
                  If False, applies standard deterministic resize and ImageNet normalization.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return PREPROCESSING_TRANSFORM


def preprocess_image(
    image_path: str,
    request_id: Optional[str] = None,
) -> torch.Tensor:
    """
    Loads, formats, and normalizes a retinal fundus image for neural network inference.

    Pipeline:
        1. Validates input path and file existence.
        2. Loads image via Pillow and converts to 3-channel RGB.
        3. Resizes image to (224, 224).
        4. Converts PIL Image to PyTorch Tensor [3, 224, 224] scaled to [0.0, 1.0].
        5. Normalizes channels using ImageNet mean and standard deviation.
        6. Adds a leading batch dimension, returning shape [1, 3, 224, 224].

    Args:
        image_path: Path to the target image file.
        request_id: Optional request correlation identifier for telemetry.

    Returns:
        torch.Tensor of shape [1, 3, 224, 224] with dtype torch.float32.

    Raises:
        TypeError: If image_path is not a string.
        ValueError: If image_path is empty or whitespace.
        FileNotFoundError: If the specified file does not exist.
        IsADirectoryError: If image_path refers to a directory rather than a file.
        InvalidImageInputError: If the file is 0 bytes, corrupted, or cannot be decoded as an image.
    """
    with measure_ml_latency() as timer:
        # 1. Path and Argument Validation
        if not isinstance(image_path, str):
            raise TypeError(f"Expected image_path to be a string, got {type(image_path).__name__}.")

        clean_path = image_path.strip()
        if not clean_path:
            raise ValueError("Image path cannot be empty or whitespace.")

        if "\x00" in clean_path:
            raise InvalidImageInputError("Image path contains invalid null byte character.", operation="preprocess_image")

        clean_name = sanitize_path(clean_path) or "unknown"

        if not os.path.exists(clean_path):
            raise FileNotFoundError(f"Image file not found: {clean_name}")

        if os.path.isdir(clean_path):
            raise IsADirectoryError(f"Expected a file path, but got a directory: {clean_name}")

        if not os.path.isfile(clean_path):
            raise InvalidImageInputError(f"Path does not point to a regular file: {clean_name}", operation="preprocess_image")

        _, ext = os.path.splitext(clean_path)
        if ext.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise InvalidImageInputError(
                f"Unsupported image file format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}: {clean_name}",
                operation="preprocess_image",
            )

        try:
            file_size = os.path.getsize(clean_path)
        except OSError as err:
            timer.stop()
            log_ml_event(
                logger,
                event_type="PREPROCESSING",
                operation="preprocess_image",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                request_id=request_id,
                details={"file": clean_name, "error": str(err)},
            )
            raise InvalidImageInputError(f"Unable to access file metadata for {clean_name}: {err}", operation="preprocess_image") from err

        if file_size == 0:
            timer.stop()
            log_ml_event(
                logger,
                event_type="PREPROCESSING",
                operation="preprocess_image",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                request_id=request_id,
                details={"file": clean_name, "error": "Empty file (0 bytes)"},
            )
            raise InvalidImageInputError(f"Image file is empty (0 bytes): {clean_name}", operation="preprocess_image")

        if file_size > MAX_IMAGE_FILE_SIZE_BYTES:
            timer.stop()
            log_ml_event(
                logger,
                event_type="PREPROCESSING",
                operation="preprocess_image",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                request_id=request_id,
                details={"file": clean_name, "error": "File size exceeds limit"},
            )
            raise InvalidImageInputError(
                f"Image file exceeds maximum allowable size of 30 MB ({file_size / (1024*1024):.1f} MB): {clean_name}",
                operation="preprocess_image",
            )

        # 2. Image Loading and Mode Conversion via Pillow
        try:
            with Image.open(clean_path) as raw_img:
                if raw_img.width <= 0 or raw_img.height <= 0:
                    raise InvalidImageInputError(
                        f"Image dimensions are invalid ({raw_img.width}x{raw_img.height}): {clean_name}",
                        operation="preprocess_image",
                    )
                if raw_img.width > MAX_IMAGE_DIMENSION or raw_img.height > MAX_IMAGE_DIMENSION:
                    raise InvalidImageInputError(
                        f"Image dimensions ({raw_img.width}x{raw_img.height}) exceed maximum allowable resolution ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}): {clean_name}",
                        operation="preprocess_image",
                    )
                raw_img.load()
                rgb_img = raw_img.convert("RGB")
        except (UnidentifiedImageError, Image.DecompressionBombError) as err:
            timer.stop()
            log_ml_event(
                logger,
                event_type="PREPROCESSING",
                operation="preprocess_image",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                request_id=request_id,
                details={"file": clean_name, "error": type(err).__name__},
            )
            raise InvalidImageInputError(
                f"Cannot identify or decode image file at '{clean_name}'. "
                "File may be corrupt, truncated, or in an unsupported format.",
                operation="preprocess_image",
            ) from err
        except InvalidImageInputError:
            raise
        except Exception as err:
            timer.stop()
            log_ml_event(
                logger,
                event_type="PREPROCESSING",
                operation="preprocess_image",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                request_id=request_id,
                details={"file": clean_name, "error": str(err)},
            )
            raise InvalidImageInputError(
                f"Failed to process image file {clean_name}: {err}",
                operation="preprocess_image",
            ) from err

        # 3. Apply Transformations
        try:
            tensor = PREPROCESSING_TRANSFORM(rgb_img)
        except Exception as err:
            timer.stop()
            log_ml_event(
                logger,
                event_type="PREPROCESSING",
                operation="preprocess_image",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                request_id=request_id,
                details={"file": sanitize_path(clean_path), "error": str(err)},
            )
            raise RuntimeError(f"Failed to apply preprocessing transforms: {err}") from err

        # 4. Add Batch Dimension [1, 3, 224, 224]
        batched_tensor = tensor.unsqueeze(0)

        # 5. Sanity Check on Output Tensor
        if batched_tensor.shape != torch.Size([1, 3, 224, 224]):
            raise RuntimeError(
                f"Unexpected output tensor shape: {batched_tensor.shape}. "
                "Expected torch.Size([1, 3, 224, 224])."
            )

        if not batched_tensor.is_floating_point():
            raise RuntimeError(
                f"Unexpected output tensor dtype: {batched_tensor.dtype}. Expected floating point."
            )

        if not torch.isfinite(batched_tensor).all():
            raise InvalidImageInputError(
                f"Preprocessed image tensor contains non-finite values (NaN or Inf): {clean_name}",
                operation="preprocess_image",
            )

        timer.stop()
        log_ml_event(
            logger,
            event_type="PREPROCESSING",
            operation="preprocess_image",
            status="SUCCESS",
            duration_ms=timer.duration_ms,
            request_id=request_id,
            details={"file": sanitize_path(clean_path), "shape": list(batched_tensor.shape)},
            level=logging.DEBUG,
        )

        return batched_tensor
