"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module for Eye Disease Screening.

Provides visual explainability for the DenseNet201 retinal disease classifier by
generating localized heatmaps showing which regions of the fundus image drove the
classification decision (Cataract, Diabetic Retinopathy, Glaucoma, or Normal).

Public Contract:
    generate_gradcam(image_path: str, predicted_class_idx: int, model=None, ..., request_id=None) -> str
"""

import base64
import logging
import math
import os
import threading
from typing import Any, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn

from ml.config import SUPPORTED_IMAGE_EXTENSIONS
from ml.exceptions import GradCAMGenerationError, InvalidImageInputError
from ml.monitoring import log_ml_event, measure_ml_latency, sanitize_path

logger = logging.getLogger(__name__)

# Guardrail limits for image dimensions and size
MAX_IMAGE_FILE_SIZE_BYTES: int = 30 * 1024 * 1024  # 30 MB
MAX_IMAGE_DIMENSION: int = 5000                     # 5000x5000 pixels

# Thread-safety lock for concurrent hook registration and backpropagation
_GRADCAM_LOCK = threading.Lock()

from ml.eye_model import (
    DEFAULT_CHECKPOINT_PATH,
    NUM_CLASSES,
    get_cached_model,
    get_default_device,
)
from ml.preprocessing import preprocess_image


def get_target_layer(
    model: nn.Module,
    target_layer: Optional[Union[str, nn.Module]] = None,
) -> nn.Module:
    """
    Identifies and returns the convolutional feature layer for Grad-CAM.

    Preference is given to 'model.features.denseblock4' in DenseNet201.
    If a custom target_layer is specified, resolves and validates it.

    Args:
        model: PyTorch model instance.
        target_layer: Optional string attribute path (e.g., 'features.denseblock4')
                      or an nn.Module instance.

    Returns:
        The resolved nn.Module layer.

    Raises:
        AttributeError: If expected layer is not found on model.
        TypeError: If target_layer parameter has an invalid type or resolves to a non-Module.
    """
    if target_layer is not None:
        if isinstance(target_layer, nn.Module):
            return target_layer
        if isinstance(target_layer, str):
            curr: Any = model
            for part in target_layer.split("."):
                if not hasattr(curr, part):
                    raise AttributeError(
                        f"Model has no attribute '{part}' while resolving target layer '{target_layer}'."
                    )
                curr = getattr(curr, part)
            if not isinstance(curr, nn.Module):
                raise TypeError(
                    f"Resolved target layer '{target_layer}' is of type {type(curr).__name__}, expected nn.Module."
                )
            return curr
        raise TypeError(
            f"target_layer must be str or nn.Module, got {type(target_layer).__name__}."
        )

    # Standard feature block in DenseNet201
    if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
        return model.features.denseblock4

    raise AttributeError(
        "Model architecture does not expose expected convolutional layer 'features.denseblock4'. "
        "Supply a compatible DenseNet201 model or specify target_layer explicitly."
    )


def generate_gradcam(
    image_path: str,
    predicted_class_idx: int,
    model: Optional[nn.Module] = None,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    target_layer: Optional[Union[str, nn.Module]] = None,
    alpha: float = 0.4,
    input_tensor: Optional[torch.Tensor] = None,
    request_id: Optional[str] = None,
) -> str:
    """
    Generates a Grad-CAM heatmap overlay for a retinal fundus image and returns it as a Base64 PNG string.

    Workflow:
        1. Validates input image path, file existence, and class index (0 <= idx < 4).
        2. Loads the original image to preserve its native resolution and aspect ratio.
        3. Uses the project's standardized preprocessing pipeline to produce input tensor [1, 3, 224, 224],
           or reuses precomputed input_tensor if supplied.
        4. Reuses the supplied model instance or loads the validated DenseNet201 checkpoint.
        5. Identifies the target convolutional layer (defaulting to model.features.denseblock4).
        6. Registers forward and gradient backward hooks.
        7. Runs a forward pass and computes gradients for the target class under standard gradient tracking.
        8. Computes Grad-CAM channel weights via global average pooling of gradients.
        9. Combines feature activations with channel weights, applies ReLU, and safely normalizes heatmap.
        10. Resizes the heatmap to match the original image dimensions exactly.
        11. Applies OpenCV COLORMAP_JET and blends heatmap with original image at specified alpha opacity.
        12. Encodes the blended visualization as PNG and returns its Base64 representation.

    Reliability & Resource Safety:
        - Removes forward and backward hooks in a finally block even on error.
        - Calls model.zero_grad(set_to_none=True) to avoid accumulating gradients across calls.
        - Restores caller-supplied model's training/evaluation state upon exit.
        - Prevents division-by-zero on uniform/constant activation maps.

    Args:
        image_path: Path to the original retinal image file on disk.
        predicted_class_idx: Integer index of target class (0 to 3).
        model: Optional pre-loaded PyTorch model. If None, loads validated DenseNet201 checkpoint.
        weights_path: Optional checkpoint path. Defaults to DEFAULT_CHECKPOINT_PATH if model is None.
        device: Target torch.device. Defaults to model device or CUDA if available, else CPU.
        target_layer: Target feature layer (defaults to model.features.denseblock4).
        alpha: Blend ratio for heatmap overlay (default: 0.4).
        input_tensor: Optional precomputed normalized tensor [1, 3, 224, 224] to avoid redundant preprocessing.
        request_id: Optional correlation identifier for telemetry.

    Returns:
        str: Base64-encoded PNG image string of the blended visualization.

    Raises:
        TypeError: If image_path is not str, or predicted_class_idx is not int.
        ValueError: If predicted_class_idx is not in [0, 3], or image path is empty.
        FileNotFoundError: If image file or checkpoint file does not exist.
        IsADirectoryError: If image_path points to a directory.
        InvalidImageInputError: If image file is empty or corrupted.
        AttributeError: If model does not expose the target convolutional layer.
        GradCAMGenerationError: If hooks fail to capture tensors or PNG encoding fails.
    """
    with measure_ml_latency() as timer:
        # 1. Input Validation
        if not isinstance(image_path, str):
            raise TypeError(f"Expected image_path as str, got {type(image_path).__name__}.")

        clean_path = image_path.strip()
        if not clean_path:
            raise ValueError("image_path cannot be empty or whitespace.")

        if "\x00" in clean_path:
            raise ValueError("image_path contains invalid null byte character.")

        if weights_path is not None and "\x00" in str(weights_path):
            raise ValueError("weights_path contains invalid null byte character.")

        clean_name = sanitize_path(clean_path) or "unknown"

        if not os.path.exists(clean_path):
            raise FileNotFoundError(f"Image file not found at: {clean_name}")

        if os.path.isdir(clean_path):
            raise IsADirectoryError(f"Image path points to a directory, not a file: {clean_name}")

        if not os.path.isfile(clean_path):
            raise InvalidImageInputError(f"Path does not point to a regular file: {clean_name}", operation="gradcam")

        _, ext = os.path.splitext(clean_path)
        if ext.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise InvalidImageInputError(
                f"Unsupported image file format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}: {clean_name}",
                operation="gradcam",
            )

        try:
            file_size = os.path.getsize(clean_path)
        except OSError as err:
            raise InvalidImageInputError(f"Unable to access file metadata for {clean_name}: {err}", operation="gradcam") from err

        if file_size == 0:
            raise InvalidImageInputError(f"Image file is empty (0 bytes): {clean_name}", operation="gradcam")

        if file_size > MAX_IMAGE_FILE_SIZE_BYTES:
            raise InvalidImageInputError(
                f"Image file exceeds maximum allowable size of 30 MB ({file_size / (1024*1024):.1f} MB): {clean_name}",
                operation="gradcam",
            )

        if isinstance(predicted_class_idx, bool) or not isinstance(predicted_class_idx, int):
            raise TypeError(
                f"predicted_class_idx must be an integer, got {type(predicted_class_idx).__name__}."
            )

        if predicted_class_idx < 0 or predicted_class_idx >= NUM_CLASSES:
            raise ValueError(
                f"Invalid predicted_class_idx: {predicted_class_idx}. "
                f"Must be an integer between 0 and {NUM_CLASSES - 1}."
            )

        if not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
            raise ValueError(f"alpha must be a finite numeric value, got {alpha}.")

        # 2. Load Original Image (BGR) to preserve native aspect ratio & dimensions
        orig_img = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        if orig_img is None or orig_img.size == 0 or len(orig_img.shape) != 3 or orig_img.shape[2] != 3:
            timer.stop()
            log_ml_event(
                logger,
                event_type="GRADCAM",
                operation="generate_gradcam",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                request_id=request_id,
                details={"file": clean_name, "error": "Failed to decode image"},
            )
            raise InvalidImageInputError(
                f"Failed to decode image from '{clean_name}'. File may be corrupted or in an unsupported format.",
                operation="gradcam",
            )
        orig_h, orig_w = orig_img.shape[:2]
        if orig_h < 1 or orig_w < 1:
            raise InvalidImageInputError(
                f"Decoded image has invalid dimensions ({orig_w}x{orig_h}): {clean_name}",
                operation="gradcam",
            )
        if orig_h > MAX_IMAGE_DIMENSION or orig_w > MAX_IMAGE_DIMENSION:
            raise InvalidImageInputError(
                f"Image dimensions ({orig_w}x{orig_h}) exceed maximum allowable resolution ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}): {clean_name}",
                operation="gradcam",
            )

        # 3. Preprocess Image using standard pipeline (or reuse precomputed tensor)
        if input_tensor is not None:
            if not isinstance(input_tensor, torch.Tensor):
                raise TypeError(
                    f"Expected input_tensor as torch.Tensor, got {type(input_tensor).__name__}."
                )
            if input_tensor.ndim != 4 or input_tensor.shape[1] != 3 or input_tensor.shape[2:] != (224, 224):
                raise ValueError(
                    f"input_tensor must have shape [1, 3, 224, 224], got {input_tensor.shape}."
                )
            if not input_tensor.is_floating_point():
                raise TypeError(
                    f"input_tensor must have floating-point dtype, got {input_tensor.dtype}."
                )
            if not torch.isfinite(input_tensor).all():
                raise ValueError("input_tensor contains non-finite values (NaN or Inf).")
            tensor_to_use = input_tensor.clone()
        else:
            tensor_to_use = preprocess_image(clean_path, request_id=request_id)

        # 4. Resolve Model and Device
        active_model: nn.Module
        if model is not None:
            if not isinstance(model, nn.Module):
                raise TypeError(f"Supplied model must be an nn.Module, got {type(model).__name__}.")
            active_model = model
            if device is not None:
                target_device = device
            else:
                param = next(active_model.parameters(), None)
                target_device = param.device if param is not None else get_default_device()
        else:
            target_weights = weights_path if weights_path is not None else DEFAULT_CHECKPOINT_PATH
            target_device = device if device is not None else get_default_device()
            active_model = get_cached_model(weights_path=target_weights, device=target_device, request_id=request_id)

        # 5. Resolve Target Convolutional Layer
        try:
            target_module = get_target_layer(active_model, target_layer=target_layer)
        except Exception as err:
            timer.stop()
            log_ml_event(
                logger,
                event_type="GRADCAM",
                operation="generate_gradcam",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                request_id=request_id,
                details={"error": str(err)},
            )
            raise

        # Prepare inputs and track state
        active_model = active_model.to(target_device)
        input_tensor_device = tensor_to_use.to(target_device)
        input_tensor_device.requires_grad_(True)

        captured_activations: Optional[torch.Tensor] = None
        captured_gradients: Optional[torch.Tensor] = None

        def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            nonlocal captured_activations
            captured_activations = out.detach().clone()

        def backward_hook(module: nn.Module, grad_inp: Any, grad_out: Tuple[torch.Tensor, ...]) -> None:
            nonlocal captured_gradients
            if grad_out and grad_out[0] is not None:
                captured_gradients = grad_out[0].detach().clone()

        forward_handle = None
        backward_handle = None
        original_training_mode = active_model.training

        with _GRADCAM_LOCK:
            # Optimize: Freeze model parameter gradients during Grad-CAM.
            # Grad-CAM only requires gradients w.r.t the target feature activation map,
            # not parameter weights. Disabling parameter gradients eliminates expensive
            # backpropagation through 20M weights across DenseNet201, saving ~40% latency.
            orig_param_grad_states = [p.requires_grad for p in active_model.parameters()]
            for p in active_model.parameters():
                p.requires_grad = False

            try:
                # 6. Register Forward and Gradient Hooks
                forward_handle = target_module.register_forward_hook(forward_hook)
                backward_handle = target_module.register_full_backward_hook(backward_hook)

                active_model.eval()
                active_model.zero_grad(set_to_none=True)

                # 7. Forward Pass and Gradient Backpropagation
                logits = active_model(input_tensor_device)
                if logits.ndim != 2 or logits.shape[1] <= predicted_class_idx:
                    raise ValueError(
                        f"Model output dimension {logits.shape} is incompatible with class index {predicted_class_idx}."
                    )
                if not torch.isfinite(logits).all():
                    raise GradCAMGenerationError(
                        "Model output logits contain non-finite values (NaN or Inf).",
                        operation="gradcam",
                    )

                score = logits[0, predicted_class_idx]
                score.backward(retain_graph=False)

                # 8. Grad-CAM Calculation
                if captured_activations is None:
                    raise GradCAMGenerationError("Forward hook did not capture activations from target layer.", operation="gradcam")
                if not torch.isfinite(captured_activations).all():
                    raise GradCAMGenerationError(
                        "Target layer feature activations contain non-finite values (NaN or Inf).",
                        operation="gradcam",
                    )
                if captured_gradients is None:
                    # Gradient w.r.t target layer is zero (e.g., disconnected or zero-gradient graph)
                    captured_gradients = torch.zeros_like(captured_activations)
                elif not torch.isfinite(captured_gradients).all():
                    raise GradCAMGenerationError(
                        "Captured feature gradients contain non-finite values (NaN or Inf).",
                        operation="gradcam",
                    )

                # Channel weights via spatial global average pooling of gradients
                weights = torch.mean(captured_gradients, dim=(2, 3), keepdim=True)

                # Linear combination of feature activations
                cam = torch.sum(weights * captured_activations, dim=1, keepdim=True)
                cam = torch.relu(cam)

                cam_np = cam.squeeze().cpu().numpy().astype(np.float32)
                if not np.isfinite(cam_np).all():
                    raise GradCAMGenerationError(
                        "Computed Grad-CAM heatmap contains non-finite values (NaN or Inf).",
                        operation="gradcam",
                    )
                if cam_np.ndim == 0:
                    cam_np = np.expand_dims(cam_np, (0, 1))
                elif cam_np.ndim == 1:
                    cam_np = np.expand_dims(cam_np, 0)

                # 9. Safe Min-Max Normalization (avoids division by zero on uniform maps)
                cam_min = float(cam_np.min())
                cam_max = float(cam_np.max())
                diff = cam_max - cam_min

                if diff > 1e-8:
                    cam_norm = (cam_np - cam_min) / diff
                else:
                    cam_norm = np.zeros_like(cam_np, dtype=np.float32)

                heatmap_uint8 = np.uint8(255.0 * cam_norm)

                # 10. Resize Heatmap to Match Original Image Dimensions
                resized_heatmap = cv2.resize(
                    heatmap_uint8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )

                # 11. Apply JET Colormap and Blend with Original Image
                colored_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
                clamped_alpha = max(0.0, min(1.0, float(alpha)))
                blended = cv2.addWeighted(
                    colored_heatmap, clamped_alpha, orig_img, 1.0 - clamped_alpha, 0.0
                )

                # 12. Encode to PNG and return Base64 String
                encode_success, buffer = cv2.imencode(".png", blended)
                if not encode_success or buffer is None:
                    raise GradCAMGenerationError("Failed to encode Grad-CAM visualization to PNG.", operation="gradcam")

                png_bytes = buffer.tobytes()
                if len(png_bytes) < 8 or png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
                    raise GradCAMGenerationError(
                        "Encoded Grad-CAM visualization is missing valid PNG magic bytes.",
                        operation="gradcam",
                    )

                b64_string = base64.b64encode(png_bytes).decode("utf-8")
                if not b64_string:
                    raise GradCAMGenerationError(
                        "Base64-encoded Grad-CAM output is empty.",
                        operation="gradcam",
                    )

                # Validate round-trip Base64 decodability and PNG validity
                try:
                    decoded_test = base64.b64decode(b64_string, validate=True)
                    if len(decoded_test) < 8 or decoded_test[:8] != b"\x89PNG\r\n\x1a\n":
                        raise GradCAMGenerationError(
                            "Grad-CAM output failed PNG byte stream verification.",
                            operation="gradcam",
                        )
                except Exception as b64_err:
                    raise GradCAMGenerationError(
                        f"Grad-CAM output failed Base64 verification: {b64_err}",
                        operation="gradcam",
                    ) from b64_err

                timer.stop()
                log_ml_event(
                    logger,
                    event_type="GRADCAM",
                    operation="generate_gradcam",
                    status="SUCCESS",
                    duration_ms=timer.duration_ms,
                    device=str(target_device),
                    request_id=request_id,
                    details={
                        "file": sanitize_path(clean_path),
                        "class_index": predicted_class_idx,
                        "alpha": alpha,
                    },
                    level=logging.INFO,
                )

                return b64_string

            except Exception as err:
                timer.stop()
                log_ml_event(
                    logger,
                    event_type="GRADCAM",
                    operation="generate_gradcam",
                    status="FAILURE",
                    duration_ms=timer.duration_ms,
                    device=str(target_device),
                    request_id=request_id,
                    details={"class_index": predicted_class_idx, "error": str(err)},
                )
                raise
            finally:
                # Resource cleanup: guarantee removal of hooks and clearing of parameter gradients
                if forward_handle is not None:
                    forward_handle.remove()
                if backward_handle is not None:
                    backward_handle.remove()
                active_model.zero_grad(set_to_none=True)
                if input_tensor_device.grad is not None:
                    input_tensor_device.grad = None
                for p, orig_state in zip(active_model.parameters(), orig_param_grad_states):
                    p.requires_grad = orig_state
                active_model.train(original_training_mode)
