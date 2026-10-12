"""
DenseNet201 Eye Disease Classification Architecture, Checkpoint Loader & Inference.

This module defines the neural network architecture for retinal disease screening
(Cataract, Diabetic Retinopathy, Glaucoma, Normal) based on a DenseNet201 backbone,
provides robust checkpoint loading, structured telemetry logging, and implements
the complete inference pipeline.

Public Contracts:
    CLASS_NAMES: list[str]
    NUM_CLASSES: int
    CLASS_TO_IDX: dict[str, int]
    IDX_TO_CLASS: dict[int, str]
    build_densenet201_model(num_classes: int = 4) -> nn.Module
    load_eye_model_checkpoint(model: nn.Module, checkpoint_path: str, device: Optional[torch.device]) -> nn.Module
    get_model(weights_path: Optional[str] = None, device: Optional[torch.device] = None) -> nn.Module
    predict_eye_disease(image_input: Union[str, torch.Tensor], ..., request_id: Optional[str] = None) -> dict
    screen_eye_image(image_path: str, ..., request_id: Optional[str] = None) -> dict
"""

import logging
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torchvision import models

from ml.exceptions import (
    CheckpointCorruptError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    DeviceUnavailableError,
    ModelInferenceError,
    ResourceExhaustionError,
)
from ml.config import get_default_checkpoint_path, resolve_device
from ml.monitoring import log_ml_event, measure_ml_latency, sanitize_path
from ml.preprocessing import preprocess_image
from ml.quality_gate import check_image_quality

logger = logging.getLogger(__name__)

# Canonical class order and label mapping for the project
CLASS_NAMES: List[str] = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal",
]
NUM_CLASSES: int = len(CLASS_NAMES)

CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# DenseNet201 feature dimension before the classifier head
DENSENET201_IN_FEATURES: int = 1920

# Default project checkpoint path convention
DEFAULT_CHECKPOINT_PATH: str = get_default_checkpoint_path()

# Device-aware cache and thread-safety lock for loaded model instances
_MODEL_CACHE: Dict[Tuple[Optional[str], str], nn.Module] = {}
_CACHE_LOCK = threading.Lock()


def clear_model_cache() -> None:
    """Clears all cached model instances from memory in a thread-safe manner."""
    global _MODEL_CACHE
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


def get_default_device() -> torch.device:
    """Selects CUDA if available, falling back cleanly to CPU via centralized config."""
    return resolve_device("auto")


def build_densenet201_model(num_classes: int = NUM_CLASSES) -> models.DenseNet:
    """
    Constructs a DenseNet201 architecture with a 4-class classification head.

    Args:
        num_classes: Number of output classes (default: 4).

    Returns:
        models.DenseNet model with model.classifier replaced by nn.Linear(1920, num_classes).
        Initialized without downloading pretrained weights (weights=None).
    """
    model = models.densenet201(weights=None)
    in_features = model.classifier.in_features
    if in_features != DENSENET201_IN_FEATURES:
        raise ValueError(
            f"Unexpected DenseNet201 in_features: {in_features}. Expected {DENSENET201_IN_FEATURES}."
        )

    model.classifier = nn.Linear(in_features, num_classes)
    log_ml_event(
        logger,
        event_type="MODEL_INIT",
        operation="build_densenet201_model",
        status="SUCCESS",
        details={"architecture": "DenseNet201", "num_classes": num_classes},
        level=logging.DEBUG,
    )
    return model


def load_eye_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    request_id: Optional[str] = None,
) -> nn.Module:
    """
    Loads and strictly validates model weights from a checkpoint file.

    Validates:
        1. Path existence and file readability.
        2. Unpickling / decoding integrity (catches corrupted binaries).
        3. Supported checkpoint container formats (raw state_dict or dictionary with metadata).
        4. Metadata class ordering consistency when metadata is present.
        5. Classifier layer dimensions (must match [NUM_CLASSES, 1920]).
        6. Strict parameter matching across all layers (no missing or unexpected keys).

    Args:
        model: DenseNet201 model instance to load weights into.
        checkpoint_path: Path to the .pth or .pt checkpoint file on disk.
        device: Target torch.device for weight tensors (defaults to get_default_device()).
        request_id: Optional correlation identifier for telemetry.

    Returns:
        The model instance updated with checkpoint weights and placed in eval mode on target device.

    Raises:
        TypeError: If checkpoint_path is not a string.
        CheckpointNotFoundError: If checkpoint file does not exist.
        IsADirectoryError: If path points to a directory.
        CheckpointCorruptError: If checkpoint is empty or corrupted.
        CheckpointIncompatibleError: If checkpoint contents or classifier dimensions are incompatible.
        RuntimeError: If parameter keys are missing or unexpected.
    """
    if not isinstance(checkpoint_path, str):
        raise TypeError(f"Expected checkpoint_path as str, got {type(checkpoint_path).__name__}.")

    clean_path = checkpoint_path.strip()
    if not clean_path:
        raise ValueError("Checkpoint path cannot be empty or whitespace.")

    if "\x00" in clean_path:
        raise CheckpointCorruptError(
            "Checkpoint path contains invalid null byte character.",
            operation="load_eye_model_checkpoint",
        )

    target_device = resolve_device(device)

    with measure_ml_latency() as timer:
        clean_name = sanitize_path(clean_path) or "unknown"

        if not os.path.exists(clean_path):
            timer.stop()
            log_ml_event(
                logger,
                event_type="CHECKPOINT_LOAD",
                operation="load_eye_model_checkpoint",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                device=str(target_device),
                request_id=request_id,
                details={"checkpoint": clean_name, "error": "Checkpoint file not found"},
            )
            raise CheckpointNotFoundError(
                f"Model checkpoint not found at: {clean_name}. "
                "Ensure the model has been trained and saved, or specify a valid checkpoint path.",
                operation="load_eye_model_checkpoint",
            )

        if os.path.isdir(clean_path):
            raise IsADirectoryError(f"Checkpoint path points to a directory, not a file: {clean_name}")

        try:
            file_size = os.path.getsize(clean_path)
        except OSError as err:
            timer.stop()
            log_ml_event(
                logger,
                event_type="CHECKPOINT_LOAD",
                operation="load_eye_model_checkpoint",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                device=str(target_device),
                request_id=request_id,
                details={"checkpoint": clean_name, "error": str(err)},
            )
            raise CheckpointCorruptError(f"Unable to read file metadata for {clean_name}: {err}", operation="load_eye_model_checkpoint") from err

        if file_size == 0:
            timer.stop()
            log_ml_event(
                logger,
                event_type="CHECKPOINT_LOAD",
                operation="load_eye_model_checkpoint",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                device=str(target_device),
                request_id=request_id,
                details={"checkpoint": clean_name, "error": "Checkpoint file is 0 bytes"},
            )
            raise CheckpointCorruptError(f"Model checkpoint file is empty (0 bytes): {clean_name}", operation="load_eye_model_checkpoint")

        try:
            raw_checkpoint = torch.load(clean_path, map_location=target_device, weights_only=True)
        except Exception as err:
            timer.stop()
            log_ml_event(
                logger,
                event_type="CHECKPOINT_LOAD",
                operation="load_eye_model_checkpoint",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                device=str(target_device),
                request_id=request_id,
                details={"checkpoint": clean_name, "error": str(err)},
            )
            raise CheckpointCorruptError(
                f"Failed to load checkpoint file at '{clean_name}'. The file may be corrupt or invalid: {err}",
                operation="load_eye_model_checkpoint",
            ) from err

        state_dict: Dict[str, Any]
        if isinstance(raw_checkpoint, dict):
            # Validate architecture identifier if present in checkpoint metadata
            meta_arch = raw_checkpoint.get("architecture")
            if meta_arch is not None and str(meta_arch).lower() != "densenet201":
                timer.stop()
                log_ml_event(
                    logger,
                    event_type="CHECKPOINT_LOAD",
                    operation="load_eye_model_checkpoint",
                    status="FAILURE",
                    duration_ms=timer.duration_ms,
                    device=str(target_device),
                    request_id=request_id,
                    details={"checkpoint": clean_name, "error": f"Architecture mismatch: {meta_arch}"},
                )
                raise CheckpointIncompatibleError(
                    f"Checkpoint architecture mismatch. Expected DenseNet201, but checkpoint specifies '{meta_arch}'.",
                    operation="load_eye_model_checkpoint",
                )

            meta_classes = raw_checkpoint.get("classes") or raw_checkpoint.get("class_names")
            if meta_classes is not None:
                if list(meta_classes) != list(CLASS_NAMES):
                    timer.stop()
                    log_ml_event(
                        logger,
                        event_type="CHECKPOINT_LOAD",
                        operation="load_eye_model_checkpoint",
                        status="FAILURE",
                        duration_ms=timer.duration_ms,
                        device=str(target_device),
                        request_id=request_id,
                        details={"checkpoint": clean_name, "error": "Class order mismatch"},
                    )
                    raise CheckpointIncompatibleError(
                        f"Checkpoint metadata class order mismatch. Expected {CLASS_NAMES}, but checkpoint specifies {list(meta_classes)}.",
                        operation="load_eye_model_checkpoint",
                    )

            if "state_dict" in raw_checkpoint:
                state_dict = raw_checkpoint["state_dict"]
            elif "model_state_dict" in raw_checkpoint:
                state_dict = raw_checkpoint["model_state_dict"]
            else:
                state_dict = raw_checkpoint
        else:
            timer.stop()
            log_ml_event(
                logger,
                event_type="CHECKPOINT_LOAD",
                operation="load_eye_model_checkpoint",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                device=str(target_device),
                request_id=request_id,
                details={"checkpoint": sanitize_path(clean_path), "error": "Unsupported payload type"},
            )
            raise CheckpointIncompatibleError(
                f"Unsupported checkpoint payload type: {type(raw_checkpoint).__name__}. Expected dictionary.",
                operation="load_eye_model_checkpoint",
            )

        if "classifier.weight" in state_dict:
            cls_weight_shape = state_dict["classifier.weight"].shape
            expected_shape = torch.Size([NUM_CLASSES, DENSENET201_IN_FEATURES])
            if cls_weight_shape != expected_shape:
                timer.stop()
                log_ml_event(
                    logger,
                    event_type="CHECKPOINT_LOAD",
                    operation="load_eye_model_checkpoint",
                    status="FAILURE",
                    duration_ms=timer.duration_ms,
                    device=str(target_device),
                    request_id=request_id,
                    details={"checkpoint": sanitize_path(clean_path), "error": "Classifier weight shape mismatch"},
                )
                raise CheckpointIncompatibleError(
                    f"Checkpoint classifier weight shape mismatch: {cls_weight_shape}. "
                    f"Expected {expected_shape} for {NUM_CLASSES}-class eye model.",
                    operation="load_eye_model_checkpoint",
                )

        if "classifier.bias" in state_dict:
            cls_bias_shape = state_dict["classifier.bias"].shape
            if cls_bias_shape != torch.Size([NUM_CLASSES]):
                timer.stop()
                log_ml_event(
                    logger,
                    event_type="CHECKPOINT_LOAD",
                    operation="load_eye_model_checkpoint",
                    status="FAILURE",
                    duration_ms=timer.duration_ms,
                    device=str(target_device),
                    request_id=request_id,
                    details={"checkpoint": sanitize_path(clean_path), "error": "Classifier bias shape mismatch"},
                )
                raise CheckpointIncompatibleError(
                    f"Checkpoint classifier bias shape mismatch: {cls_bias_shape}. "
                    f"Expected torch.Size([{NUM_CLASSES}]).",
                    operation="load_eye_model_checkpoint",
                )

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as err:
            timer.stop()
            log_ml_event(
                logger,
                event_type="CHECKPOINT_LOAD",
                operation="load_eye_model_checkpoint",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                device=str(target_device),
                request_id=request_id,
                details={"checkpoint": sanitize_path(clean_path), "error": str(err)},
            )
            raise CheckpointIncompatibleError(
                f"Incompatible state_dict keys for DenseNet201 architecture: {err}",
                operation="load_eye_model_checkpoint",
            ) from err

        model.to(target_device)
        model.eval()
        model.requires_grad_(False)

        timer.stop()
        log_ml_event(
            logger,
            event_type="CHECKPOINT_LOAD",
            operation="load_eye_model_checkpoint",
            status="SUCCESS",
            duration_ms=timer.duration_ms,
            device=str(target_device),
            request_id=request_id,
            details={"checkpoint": sanitize_path(clean_path), "epoch": raw_checkpoint.get("epoch") if isinstance(raw_checkpoint, dict) else None},
            level=logging.INFO,
        )

    return model


def get_model(
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    request_id: Optional[str] = None,
) -> nn.Module:
    """
    Instantiates the 4-class DenseNet201 model on the target device,
    optionally loading verified weights from a checkpoint file.

    Args:
        weights_path: Optional path to a trained checkpoint file.
        device: Target device (defaults to CUDA if available, else CPU).
        request_id: Optional correlation identifier.

    Returns:
        models.DenseNet model configured for 4-class retinal disease screening.
    """
    target_device = resolve_device(device)
    model = build_densenet201_model(num_classes=NUM_CLASSES)

    if weights_path is not None:
        model = load_eye_model_checkpoint(model, weights_path, device=target_device, request_id=request_id)
    else:
        model = model.to(target_device)
        model.eval()

    return model


def get_cached_model(
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    request_id: Optional[str] = None,
) -> nn.Module:
    """
    Retrieves or loads a model instance using the device-aware in-memory cache.

    Args:
        weights_path: Optional path to the checkpoint file.
        device: Target device.
        request_id: Optional correlation identifier.

    Returns:
        Cached or freshly initialized nn.Module.
    """
    target_device = resolve_device(device)
    clean_path = os.path.abspath(weights_path) if weights_path is not None else None
    cache_key = (clean_path, str(target_device))

    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        loaded_model = get_model(weights_path=weights_path, device=target_device, request_id=request_id)
        _MODEL_CACHE[cache_key] = loaded_model
        return loaded_model


def compute_risk_level(predicted_class: str, confidence: float) -> str:
    """
    Maps model prediction and confidence score to screening risk level.

    Rules:
        - "normal" class -> "LOW"
        - Non-normal with confidence >= 0.80 -> "HIGH"
        - Non-normal with confidence >= 0.50 -> "MODERATE"
        - Non-normal with confidence < 0.50 -> "LOW"

    Clinical Note:
        This is a preliminary triage indicator based on model classification confidence,
        strictly intended for screening prioritization and decision support.
        It does not represent an authoritative clinical diagnosis.
    """
    if predicted_class == "normal":
        return "LOW"
    if confidence >= 0.80:
        return "HIGH"
    if confidence >= 0.50:
        return "MODERATE"
    return "LOW"


def predict_eye_disease(
    image_input: Union[str, torch.Tensor],
    weights_path: Optional[str] = None,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    use_cache: bool = True,
    validate_quality: bool = False,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes eye disease classification inference on a fundus image.

    Pipeline:
        1. If validate_quality is True, routes image path through screen_eye_image.
        2. Validates input format and processes raw image path to tensor if needed.
        3. Resolves model instance (provided model, cached model, or newly loaded model).
        4. Runs forward pass under torch.inference_mode().
        5. Applies softmax to produce probability distribution across the 4 canonical classes.
        6. Computes predicted class, confidence, and risk level.
        7. Returns structured, JSON-serializable dictionary.

    Args:
        image_input: Path to image file on disk (str) or preprocessed tensor [1, 3, 224, 224].
        weights_path: Path to checkpoint. Defaults to DEFAULT_CHECKPOINT_PATH if model is None.
        model: Pre-loaded model instance. If provided, bypasses checkpoint loading.
        device: Target torch device (CPU or CUDA). Defaults to model device or default device.
        use_cache: Whether to use cached model instance when model argument is None.
        validate_quality: If True, validates image quality first via screen_eye_image.
        request_id: Optional correlation identifier for telemetry.

    Returns:
        dict:
            - prediction (str): Name of predicted class.
            - class_index (int): Zero-based index of predicted class.
            - confidence (float): Probability of predicted class in [0.0, 1.0].
            - risk_level (str): "HIGH" | "MODERATE" | "LOW".
            - all_probabilities (dict): Probabilities for all 4 classes in canonical order.
            (Or quality rejection dict if validate_quality=True and quality gate fails).

    Raises:
        TypeError: If image_input is neither str nor torch.Tensor, or if validate_quality is True with a tensor.
        ValueError: If image_input tensor has incorrect dimensions.
        CheckpointNotFoundError: If checkpoint file does not exist.
        InvalidImageInputError: If image file is corrupted.
        ModelInferenceError: If forward pass fails.
        ResourceExhaustionError: If CUDA memory or system memory is exhausted.
    """
    if validate_quality:
        if not isinstance(image_input, str):
            raise TypeError(
                f"Quality validation requires an image file path (str), got {type(image_input).__name__}."
            )
        return screen_eye_image(
            image_path=image_input,
            weights_path=weights_path,
            model=model,
            device=device,
            use_cache=use_cache,
            request_id=request_id,
        )

    with measure_ml_latency() as timer:
        # 1. Process Input into Preprocessed Tensor
        tensor: torch.Tensor
        if isinstance(image_input, str):
            tensor = preprocess_image(image_input) if request_id is None else preprocess_image(image_input, request_id=request_id)
        elif isinstance(image_input, torch.Tensor):
            if image_input.ndim != 4 or image_input.shape[1] != 3 or image_input.shape[2:] != (224, 224):
                raise ValueError(
                    f"Input tensor must have shape [1, 3, 224, 224] or [B, 3, 224, 224]. Got {image_input.shape}."
                )
            if not image_input.is_floating_point():
                raise TypeError(
                    f"Input tensor must have floating-point dtype, got {image_input.dtype}."
                )
            if not torch.isfinite(image_input).all():
                raise ValueError("Input tensor contains non-finite pixel values (NaN or Inf).")
            tensor = image_input
        else:
            raise TypeError(
                f"Expected image_input as str (path) or torch.Tensor, got {type(image_input).__name__}."
            )

        # 2. Resolve Model Instance
        active_model: nn.Module
        if model is not None:
            active_model = model
            if device is not None:
                target_device = resolve_device(device)
            else:
                param = next(active_model.parameters(), None)
                target_device = param.device if param is not None else resolve_device("auto")
        else:
            target_weights = weights_path if weights_path is not None else DEFAULT_CHECKPOINT_PATH
            target_device = resolve_device(device)
            if use_cache:
                active_model = get_cached_model(weights_path=target_weights, device=target_device, request_id=request_id)
            else:
                active_model = get_model(weights_path=target_weights, device=target_device, request_id=request_id)

        active_model.eval()
        tensor = tensor.to(target_device)

        # 3. Execution under Inference Context
        try:
            with torch.inference_mode():
                logits = active_model(tensor)
                if not torch.isfinite(logits).all():
                    raise ModelInferenceError(
                        "Model output logits contain non-finite values (NaN or Inf).",
                        operation="predict_eye_disease",
                    )
                probs = torch.softmax(logits, dim=1).squeeze(0)
                if not torch.isfinite(probs).all():
                    raise ModelInferenceError(
                        "Computed class probabilities contain non-finite values (NaN or Inf).",
                        operation="predict_eye_disease",
                    )
                conf_tensor, pred_idx_tensor = torch.max(probs, dim=0)
        except (torch.cuda.OutOfMemoryError, MemoryError) as err:
            timer.stop()
            log_ml_event(
                logger,
                event_type="INFERENCE",
                operation="predict_eye_disease",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                device=str(target_device),
                request_id=request_id,
                details={"error": "Resource exhaustion (OOM)", "error_type": type(err).__name__},
            )
            raise ResourceExhaustionError(f"Inference failed due to memory exhaustion: {err}", operation="predict_eye_disease") from err
        except Exception as err:
            timer.stop()
            log_ml_event(
                logger,
                event_type="INFERENCE",
                operation="predict_eye_disease",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                device=str(target_device),
                request_id=request_id,
                details={"error": str(err), "error_type": type(err).__name__},
            )
            raise ModelInferenceError(f"Neural network inference execution failed: {err}", operation="predict_eye_disease") from err

        # 4. Strict Output Contract Validation
        if probs.numel() != NUM_CLASSES:
            raise ModelInferenceError(
                f"Model output dimension mismatch: expected exactly {NUM_CLASSES} class probabilities, got {probs.numel()}.",
                operation="predict_eye_disease",
            )

        prob_vals = [float(p.item()) for p in probs]
        if any(not math.isfinite(pv) or pv < 0.0 or pv > 1.0 for pv in prob_vals):
            raise ModelInferenceError(
                "Class probabilities contain non-finite or out-of-range [0.0, 1.0] values.",
                operation="predict_eye_disease",
            )

        prob_sum = float(torch.sum(probs).item())
        if abs(prob_sum - 1.0) > 1e-3:
            raise ModelInferenceError(
                f"Class probabilities do not sum to 1.0 within tolerance 1e-3 (sum={prob_sum:.6f}).",
                operation="predict_eye_disease",
            )

        pred_idx = int(pred_idx_tensor.item())
        if pred_idx < 0 or pred_idx >= NUM_CLASSES:
            raise ModelInferenceError(
                f"Predicted class index {pred_idx} is out of bounds [0, {NUM_CLASSES - 1}].",
                operation="predict_eye_disease",
            )

        predicted_class = CLASS_NAMES[pred_idx]
        if CLASS_TO_IDX.get(predicted_class) != pred_idx:
            raise ModelInferenceError(
                f"Predicted label mismatch: class '{predicted_class}' index does not match mapping.",
                operation="predict_eye_disease",
            )

        confidence_val = round(float(conf_tensor.item()), 4)
        risk_level_val = compute_risk_level(predicted_class, confidence_val)

        probabilities_dict = {
            cls_name: round(float(probs[i].item()), 4)
            for i, cls_name in enumerate(CLASS_NAMES)
        }

        if abs(confidence_val - probabilities_dict[predicted_class]) > 1e-4:
            raise ModelInferenceError(
                f"Confidence score ({confidence_val}) does not match probability of predicted class ({probabilities_dict[predicted_class]}).",
                operation="predict_eye_disease",
            )

        timer.stop()
        log_ml_event(
            logger,
            event_type="INFERENCE",
            operation="predict_eye_disease",
            status="SUCCESS",
            duration_ms=timer.duration_ms,
            device=str(target_device),
            request_id=request_id,
            details={
                "prediction": predicted_class,
                "confidence": confidence_val,
                "risk_level": risk_level_val,
            },
            level=logging.INFO,
        )

        # 5. Return JSON-serializable Dictionary adhering to contract
        return {
            "prediction": predicted_class,
            "class_index": pred_idx,
            "confidence": confidence_val,
            "risk_level": risk_level_val,
            "all_probabilities": probabilities_dict,
        }


def screen_eye_image(
    image_path: str,
    weights_path: Optional[str] = None,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    use_cache: bool = True,
    include_gradcam: bool = False,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the integrated eye-screening workflow on a retinal fundus image.

    Workflow:
        1. Validates image using the automated quality gate (check_image_quality).
        2. If overall_pass is False:
           IMMEDIATELY stops and returns the quality rejection payload adhering to
           the backend contract:
               {
                   "status": "rejected",
                   "reason": <error_message>,
                   "quality_checks": <quality_result_dict>,
               }
           Neither preprocessing nor disease classification is executed.
        3. If quality validation passes:
           Preprocesses the image via preprocess_image(image_path).
        4. Runs inference using the validated DenseNet201 checkpoint.
        5. If include_gradcam is True:
           Generates Grad-CAM heatmap visualization using the same model instance and
           the exact predicted class index returned by classification.
        6. Returns structured classification output adhering to the backend contract.

    Clinical & Interpretability Note:
        Grad-CAM highlights image regions with the highest gradient activation for
        the predicted class, providing visual interpretability. It does not establish
        a clinical diagnosis or prove model correctness.

    Args:
        image_path: Path to the target image file on disk (str).
        weights_path: Path to checkpoint. Defaults to DEFAULT_CHECKPOINT_PATH if model is None.
        model: Pre-loaded model instance. If provided, bypasses checkpoint loading.
        device: Target torch device (CPU or CUDA). Defaults to model device or default device.
        use_cache: Whether to use cached model instance when model argument is None.
        include_gradcam: Whether to generate and attach Grad-CAM Base64 PNG visualization.
        request_id: Optional correlation identifier for telemetry.

    Returns:
        dict: Either rejection dictionary or success dictionary with complete disease prediction.
    """
    with measure_ml_latency() as timer:
        # 1. Run Automated Quality Gate Check
        quality_result = check_image_quality(image_path) if request_id is None else check_image_quality(image_path, request_id=request_id)

        # 2. Gate Decision: Stop immediately on quality failure
        if not quality_result.get("overall_pass", False):
            timer.stop()
            log_ml_event(
                logger,
                event_type="SCREENING_WORKFLOW",
                operation="screen_eye_image",
                status="REJECTED",
                duration_ms=timer.duration_ms,
                request_id=request_id,
                details={
                    "file": sanitize_path(image_path),
                    "reason": quality_result.get("error_message"),
                },
            )
            return {
                "status": "rejected",
                "reason": quality_result.get("error_message") or "Image failed quality gate validation.",
                "quality_checks": quality_result,
            }

        # 3. Quality Validation Passed: Preprocess retinal fundus image
        preprocessed_tensor = preprocess_image(image_path) if request_id is None else preprocess_image(image_path, request_id=request_id)

        # 4. DenseNet201 Inference
        if request_id is not None:
            inference_result = predict_eye_disease(
                image_input=preprocessed_tensor,
                weights_path=weights_path,
                model=model,
                device=device,
                use_cache=use_cache,
                validate_quality=False,
                request_id=request_id,
            )
        else:
            inference_result = predict_eye_disease(
                image_input=preprocessed_tensor,
                weights_path=weights_path,
                model=model,
                device=device,
                use_cache=use_cache,
                validate_quality=False,
            )

        # 5. Optional Grad-CAM Generation (Consistent model & predicted class index reuse)
        gradcam_b64: Optional[str] = None
        if include_gradcam:
            from ml.gradcam import generate_gradcam

            # Resolve active model instance to avoid duplicate model loading
            active_model = model
            if active_model is None:
                target_weights = weights_path if weights_path is not None else DEFAULT_CHECKPOINT_PATH
                target_device = resolve_device(device)
                if use_cache:
                    active_model = get_cached_model(weights_path=target_weights, device=target_device, request_id=request_id)
                else:
                    active_model = get_model(weights_path=target_weights, device=target_device, request_id=request_id)

            if request_id is not None:
                gradcam_b64 = generate_gradcam(
                    image_path=image_path,
                    predicted_class_idx=inference_result["class_index"],
                    model=active_model,
                    device=device,
                    input_tensor=preprocessed_tensor,
                    request_id=request_id,
                )
            else:
                gradcam_b64 = generate_gradcam(
                    image_path=image_path,
                    predicted_class_idx=inference_result["class_index"],
                    model=active_model,
                    device=device,
                    input_tensor=preprocessed_tensor,
                )

        # 6. Return Unified Success Response adhering to backend contract
        response_data: Dict[str, Any] = {
            "status": "success",
            "prediction": inference_result["prediction"],
            "class_index": inference_result["class_index"],
            "confidence": inference_result["confidence"],
            "risk_level": inference_result["risk_level"],
            "all_probabilities": inference_result["all_probabilities"],
            "quality_checks": quality_result,
        }
        if include_gradcam:
            response_data["gradcam_image"] = gradcam_b64

        timer.stop()
        log_ml_event(
            logger,
            event_type="SCREENING_WORKFLOW",
            operation="screen_eye_image",
            status="SUCCESS",
            duration_ms=timer.duration_ms,
            request_id=request_id,
            details={
                "file": sanitize_path(image_path),
                "prediction": inference_result["prediction"],
                "confidence": inference_result["confidence"],
                "risk_level": inference_result["risk_level"],
                "has_gradcam": include_gradcam,
            },
            level=logging.INFO,
        )

        return response_data
