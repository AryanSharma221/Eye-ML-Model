"""
Physiological Vital Signs Analysis and Screening Module.

Evaluates vital sign measurements (Heart Rate, SpO2, Temperature) independently
from retinal imaging using clinical reference range rules, structured telemetry,
and typed exception handling.

Public Contract:
    analyze_vitals(heart_rate: float, spo2: float, temperature: float, ..., request_id=None) -> dict[str, Any]
    assess_heart_rate(heart_rate: float) -> str
    assess_spo2(spo2: float) -> str
    assess_temperature(temperature: float) -> str
    assess_overall_status(hr_status: str, spo2_status: str, temp_status: str) -> str
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

from ml.exceptions import PhysiologicalTypeValidationError, PhysiologicalValidationError
from ml.monitoring import log_ml_event, measure_ml_latency

logger = logging.getLogger(__name__)

# Plausible physiological measurement ranges (for sanity and artifact rejection)
HR_MIN_PLAUSIBLE: float = 20.0    # Severe extreme lower bound for living human
HR_MAX_PLAUSIBLE: float = 300.0   # Extreme ventricular tachycardia upper bound
SPO2_MIN_PLAUSIBLE: float = 0.0   # Minimum physical percentage
SPO2_MAX_PLAUSIBLE: float = 100.0 # Maximum physical saturation percentage
TEMP_MIN_PLAUSIBLE: float = 25.0  # Profound hypothermia floor (°C)
TEMP_MAX_PLAUSIBLE: float = 45.0  # Lethal hyperthermia ceiling (°C)

# Clinical Thresholds (Adult Resting Baseline)
HR_CRITICAL_LOW: float = 40.0     # Severe bradycardia (<40 bpm) -> Abnormal
HR_NORMAL_MIN: float = 60.0       # Bradycardia (<60 bpm) -> Lower than usual
HR_NORMAL_MAX: float = 100.0      # Normal range: 60 - 100 bpm
HR_CRITICAL_HIGH: float = 120.0   # Severe tachycardia (>120 bpm) -> Abnormal
                                  # 100 < hr <= 120 -> Higher than usual

SPO2_CRITICAL_LOW: float = 90.0   # Severe hypoxemia (<90%) -> Abnormal
SPO2_NORMAL_MIN: float = 95.0     # Mild hypoxemia (90% - 94.9%) -> Lower than usual
                                  # Normal range: 95% - 100%

TEMP_CRITICAL_LOW: float = 35.0   # Severe hypothermia (<35.0°C) -> Abnormal
TEMP_NORMAL_MIN: float = 36.1     # Mild hypothermia (35.0°C - 36.0°C) -> Lower than usual
TEMP_NORMAL_MAX: float = 37.2     # Normal range: 36.1°C - 37.2°C
TEMP_CRITICAL_HIGH: float = 38.0  # Low-grade fever (37.3°C - 38.0°C) -> Higher than usual
                                  # Pyrexia / High fever (>38.0°C) -> Abnormal

# Canonical Status Strings
STATUS_NORMAL: str = "Normal"
STATUS_HIGHER: str = "Higher than usual"
STATUS_LOWER: str = "Lower than usual"
STATUS_ABNORMAL: str = "Abnormal"

# Risk Levels (Screening Triage)
RISK_LOW: str = "LOW"
RISK_MODERATE: str = "MODERATE"
RISK_HIGH: str = "HIGH"

CLINICAL_DISCLAIMER: str = (
    "Preliminary screening assessment based on standard physiological reference ranges. "
    "Normal vital signs do not rule out underlying medical conditions or disease. "
    "This does not constitute an authoritative clinical diagnosis. Consult a licensed medical practitioner."
)


def _validate_measurement(
    val: Any,
    name: str,
    min_val: float,
    max_val: float,
    unit: str,
) -> float:
    """
    Validates that a vital sign reading is numeric, finite, and physiologically plausible.

    Raises:
        PhysiologicalTypeValidationError: If val is None, bool, or non-numeric (subclasses TypeError).
        PhysiologicalValidationError: If val is NaN, infinite, or outside plausible limits (subclasses ValueError).
    """
    if val is None:
        raise PhysiologicalTypeValidationError(f"{name} is missing (None provided).", metric_name=name)

    if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
        raise PhysiologicalTypeValidationError(
            f"{name} must be a numeric value, got {type(val).__name__}.",
            metric_name=name,
        )

    num_val = float(val)

    if math.isnan(num_val):
        raise PhysiologicalValidationError(f"{name} cannot be NaN.", metric_name=name)

    if math.isinf(num_val):
        raise PhysiologicalValidationError(f"{name} cannot be infinite.", metric_name=name)

    if num_val < min_val or num_val > max_val:
        raise PhysiologicalValidationError(
            f"{name} reading ({num_val} {unit}) is outside physiologically plausible range "
            f"[{min_val}, {max_val}] {unit}.",
            metric_name=name,
        )

    return num_val


ALLOWED_STATUSES: Tuple[str, ...] = (STATUS_NORMAL, STATUS_HIGHER, STATUS_LOWER, STATUS_ABNORMAL)


def assess_heart_rate(heart_rate: float) -> str:
    """
    Evaluates resting heart rate in Beats Per Minute (BPM).
    """
    val = _validate_measurement(heart_rate, "Heart rate", HR_MIN_PLAUSIBLE, HR_MAX_PLAUSIBLE, "BPM")
    if val < HR_CRITICAL_LOW:
        return STATUS_ABNORMAL
    if val < HR_NORMAL_MIN:
        return STATUS_LOWER
    if val <= HR_NORMAL_MAX:
        return STATUS_NORMAL
    if val <= HR_CRITICAL_HIGH:
        return STATUS_HIGHER
    return STATUS_ABNORMAL


def assess_spo2(spo2: float) -> str:
    """
    Evaluates peripheral capillary oxygen saturation percentage (% SpO2).
    """
    val = _validate_measurement(spo2, "SpO2", SPO2_MIN_PLAUSIBLE, SPO2_MAX_PLAUSIBLE, "%")
    if val < SPO2_CRITICAL_LOW:
        return STATUS_ABNORMAL
    if val < SPO2_NORMAL_MIN:
        return STATUS_LOWER
    return STATUS_NORMAL


def assess_temperature(temperature: float) -> str:
    """
    Evaluates core body temperature in degrees Celsius (°C).
    """
    val = _validate_measurement(temperature, "Temperature", TEMP_MIN_PLAUSIBLE, TEMP_MAX_PLAUSIBLE, "°C")
    if val < TEMP_CRITICAL_LOW:
        return STATUS_ABNORMAL
    if val < TEMP_NORMAL_MIN:
        return STATUS_LOWER
    if val <= TEMP_NORMAL_MAX:
        return STATUS_NORMAL
    if val <= TEMP_CRITICAL_HIGH:
        return STATUS_HIGHER
    return STATUS_ABNORMAL


def assess_overall_status(hr_status: str, spo2_status: str, temp_status: str) -> str:
    """
    Aggregates individual vital sign statuses into an overall screening status.
    """
    for name, s in [("hr_status", hr_status), ("spo2_status", spo2_status), ("temp_status", temp_status)]:
        if not isinstance(s, str):
            raise PhysiologicalTypeValidationError(f"{name} must be a string, got {type(s).__name__}.")
        if s not in ALLOWED_STATUSES:
            raise PhysiologicalValidationError(f"Invalid status '{s}' for {name}. Must be one of {list(ALLOWED_STATUSES)}.")

    statuses = [hr_status, spo2_status, temp_status]

    if any(s == STATUS_ABNORMAL for s in statuses):
        return STATUS_ABNORMAL

    has_higher = any(s == STATUS_HIGHER for s in statuses)
    has_lower = any(s == STATUS_LOWER for s in statuses)

    if has_higher and has_lower:
        return STATUS_ABNORMAL

    if has_higher:
        return STATUS_HIGHER

    if has_lower:
        return STATUS_LOWER

    return STATUS_NORMAL


def compute_vital_risk_level(overall_status: str) -> str:
    """Maps overall vital status to screening triage risk category."""
    if not isinstance(overall_status, str):
        raise PhysiologicalTypeValidationError(f"overall_status must be a string, got {type(overall_status).__name__}.")
    if overall_status == STATUS_ABNORMAL:
        return RISK_HIGH
    if overall_status in (STATUS_HIGHER, STATUS_LOWER):
        return RISK_MODERATE
    if overall_status == STATUS_NORMAL:
        return RISK_LOW
    if overall_status in ("Invalid", "UNKNOWN"):
        return "UNKNOWN"
    raise PhysiologicalValidationError(f"Invalid overall_status '{overall_status}'. Expected one of {list(ALLOWED_STATUSES)}.")


def analyze_vitals(
    heart_rate: float,
    spo2: float,
    temperature: float,
    raise_errors: bool = True,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyzes physiological vital readings and produces structured clinical status labels.

    Args:
        heart_rate: Heart rate in BPM (physiologically plausible: 20-300)
        spo2: Blood oxygen saturation in % (physiologically plausible: 0-100)
        temperature: Core body temperature in °C (physiologically plausible: 25-45)
        raise_errors: If True (default), invalid inputs raise ValueError/TypeError.
                      If False, returns structured error dictionary.
        request_id: Optional correlation identifier for telemetry.

    Returns:
        dict: Complete vital assessment contract dictionary.
    """
    with measure_ml_latency() as timer:
        try:
            clean_hr = _validate_measurement(heart_rate, "Heart rate", HR_MIN_PLAUSIBLE, HR_MAX_PLAUSIBLE, "BPM")
            clean_spo2 = _validate_measurement(spo2, "SpO2", SPO2_MIN_PLAUSIBLE, SPO2_MAX_PLAUSIBLE, "%")
            clean_temp = _validate_measurement(temperature, "Temperature", TEMP_MIN_PLAUSIBLE, TEMP_MAX_PLAUSIBLE, "°C")
        except (TypeError, ValueError) as err:
            timer.stop()
            log_ml_event(
                logger,
                event_type="PHYSIO_VALIDATION",
                operation="analyze_vitals",
                status="FAILURE",
                duration_ms=timer.duration_ms,
                request_id=request_id,
                details={"error": str(err), "error_type": type(err).__name__},
            )
            if raise_errors:
                raise
            return {
                "status": "error",
                "error_message": str(err),
                "overall_status": "Invalid",
                "risk_level": "UNKNOWN",
                "is_abnormal": True,
                "disclaimer": CLINICAL_DISCLAIMER,
            }

        hr_status = assess_heart_rate(clean_hr)
        spo2_status = assess_spo2(clean_spo2)
        temp_status = assess_temperature(clean_temp)
        overall_status = assess_overall_status(hr_status, spo2_status, temp_status)
        risk_level = compute_vital_risk_level(overall_status)
        is_abnormal = overall_status != STATUS_NORMAL

        # Construct descriptive clinical summary
        abnormalities = []
        if hr_status != STATUS_NORMAL:
            abnormalities.append(f"Heart rate ({clean_hr:.1f} BPM: {hr_status})")
        if spo2_status != STATUS_NORMAL:
            abnormalities.append(f"SpO2 ({clean_spo2:.1f}%: {spo2_status})")
        if temp_status != STATUS_NORMAL:
            abnormalities.append(f"Temperature ({clean_temp:.1f}°C: {temp_status})")

        if not abnormalities:
            summary = "All vital signs are within normal clinical reference ranges."
        else:
            summary = f"Flagged physiological metrics: {', '.join(abnormalities)}."

        timer.stop()
        log_ml_event(
            logger,
            event_type="PHYSIO_ANALYSIS",
            operation="analyze_vitals",
            status="SUCCESS",
            duration_ms=timer.duration_ms,
            request_id=request_id,
            details={
                "overall_status": overall_status,
                "risk_level": risk_level,
                "is_abnormal": is_abnormal,
            },
            level=logging.INFO,
        )

        return {
            "status": "success",
            "heart_rate": {
                "value": round(clean_hr, 1),
                "status": hr_status,
                "unit": "BPM",
            },
            "spo2": {
                "value": round(clean_spo2, 1),
                "status": spo2_status,
                "unit": "%",
            },
            "temperature": {
                "value": round(clean_temp, 1),
                "status": temp_status,
                "unit": "°C",
            },
            # Top-level status aliases for direct DB column persistence
            "hr_status": hr_status,
            "spo2_status": spo2_status,
            "temp_status": temp_status,
            "overall_status": overall_status,
            "risk_level": risk_level,
            "is_abnormal": is_abnormal,
            "summary": summary,
            "disclaimer": CLINICAL_DISCLAIMER,
        }
