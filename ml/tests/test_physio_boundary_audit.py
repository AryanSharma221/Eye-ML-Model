"""
Comprehensive Boundary, Independence, and Semantic Validation for Physiological Analysis.

Audits and validates ml/physio_model.py against:
1. Exact threshold boundaries (immediately below, at, and above every configured threshold).
2. Missing, non-numeric, string, boolean, and complex type inputs.
3. NaN, positive infinity, negative infinity, and extreme out-of-range values.
4. Multiple concurrent abnormal and divergent mild vital-sign readings.
5. Determinism across repeated calls and thread concurrency.
6. Strict independence from ophthalmic inference and zero dependency on checkpoints.
7. Explicit result semantics (units, error distinction vs. clinical abnormality, disclaimers).
"""

import concurrent.futures
import json
import math
import sys
import numpy as np
import pytest

from ml.exceptions import PhysiologicalTypeValidationError, PhysiologicalValidationError
from ml.physio_model import (
    HR_CRITICAL_HIGH,
    HR_CRITICAL_LOW,
    HR_MAX_PLAUSIBLE,
    HR_MIN_PLAUSIBLE,
    HR_NORMAL_MAX,
    HR_NORMAL_MIN,
    RISK_HIGH,
    RISK_LOW,
    RISK_MODERATE,
    SPO2_CRITICAL_LOW,
    SPO2_MAX_PLAUSIBLE,
    SPO2_MIN_PLAUSIBLE,
    SPO2_NORMAL_MIN,
    STATUS_ABNORMAL,
    STATUS_HIGHER,
    STATUS_LOWER,
    STATUS_NORMAL,
    TEMP_CRITICAL_HIGH,
    TEMP_CRITICAL_LOW,
    TEMP_MAX_PLAUSIBLE,
    TEMP_MIN_PLAUSIBLE,
    TEMP_NORMAL_MAX,
    TEMP_NORMAL_MIN,
    analyze_vitals,
    assess_heart_rate,
    assess_overall_status,
    assess_spo2,
    assess_temperature,
    compute_vital_risk_level,
)
from ml.screening_pipeline import run_screening


class TestConfiguredThresholdBoundaries:
    """Verifies behavior immediately below, at, and above each configured threshold."""

    # --------------------------------------------------------------------------
    # Heart Rate Thresholds: [20.0, 40.0, 60.0, 100.0, 120.0, 300.0]
    # --------------------------------------------------------------------------
    def test_hr_min_plausible_boundary(self):
        # Immediately below HR_MIN_PLAUSIBLE (20.0) -> Out of range validation error
        with pytest.raises(PhysiologicalValidationError, match="plausible"):
            assess_heart_rate(19.9)

        # At HR_MIN_PLAUSIBLE (20.0) -> Valid reading, evaluated as severe bradycardia (<40.0)
        assert assess_heart_rate(20.0) == STATUS_ABNORMAL

        # Immediately above HR_MIN_PLAUSIBLE (20.1) -> Evaluated as severe bradycardia (<40.0)
        assert assess_heart_rate(20.1) == STATUS_ABNORMAL

    def test_hr_critical_low_boundary(self):
        # Immediately below HR_CRITICAL_LOW (40.0) -> Severe bradycardia (<40.0)
        assert assess_heart_rate(39.9) == STATUS_ABNORMAL

        # At HR_CRITICAL_LOW (40.0) -> Mild bradycardia [40.0, 60.0)
        assert assess_heart_rate(40.0) == STATUS_LOWER

        # Immediately above HR_CRITICAL_LOW (40.1) -> Mild bradycardia [40.0, 60.0)
        assert assess_heart_rate(40.1) == STATUS_LOWER

    def test_hr_normal_min_boundary(self):
        # Immediately below HR_NORMAL_MIN (60.0) -> Mild bradycardia [40.0, 60.0)
        assert assess_heart_rate(59.9) == STATUS_LOWER

        # At HR_NORMAL_MIN (60.0) -> Normal resting adult [60.0, 100.0]
        assert assess_heart_rate(60.0) == STATUS_NORMAL

        # Immediately above HR_NORMAL_MIN (60.1) -> Normal resting adult [60.0, 100.0]
        assert assess_heart_rate(60.1) == STATUS_NORMAL

    def test_hr_normal_max_boundary(self):
        # Immediately below HR_NORMAL_MAX (100.0) -> Normal resting adult [60.0, 100.0]
        assert assess_heart_rate(99.9) == STATUS_NORMAL

        # At HR_NORMAL_MAX (100.0) -> Normal resting adult [60.0, 100.0]
        assert assess_heart_rate(100.0) == STATUS_NORMAL

        # Immediately above HR_NORMAL_MAX (100.1) -> Mild tachycardia (100.0, 120.0]
        assert assess_heart_rate(100.1) == STATUS_HIGHER

    def test_hr_critical_high_boundary(self):
        # Immediately below HR_CRITICAL_HIGH (120.0) -> Mild tachycardia (100.0, 120.0]
        assert assess_heart_rate(119.9) == STATUS_HIGHER

        # At HR_CRITICAL_HIGH (120.0) -> Mild tachycardia (100.0, 120.0]
        assert assess_heart_rate(120.0) == STATUS_HIGHER

        # Immediately above HR_CRITICAL_HIGH (120.1) -> Severe tachycardia (>120.0)
        assert assess_heart_rate(120.1) == STATUS_ABNORMAL

    def test_hr_max_plausible_boundary(self):
        # Immediately below HR_MAX_PLAUSIBLE (300.0) -> Severe tachycardia (>120.0)
        assert assess_heart_rate(299.9) == STATUS_ABNORMAL

        # At HR_MAX_PLAUSIBLE (300.0) -> Severe tachycardia (>120.0)
        assert assess_heart_rate(300.0) == STATUS_ABNORMAL

        # Immediately above HR_MAX_PLAUSIBLE (300.1) -> Out of range validation error
        with pytest.raises(PhysiologicalValidationError, match="plausible"):
            assess_heart_rate(300.1)

    # --------------------------------------------------------------------------
    # SpO2 Thresholds: [0.0, 90.0, 95.0, 100.0]
    # --------------------------------------------------------------------------
    def test_spo2_min_plausible_boundary(self):
        # Immediately below SPO2_MIN_PLAUSIBLE (0.0) -> Out of range validation error
        with pytest.raises(PhysiologicalValidationError, match="plausible"):
            assess_spo2(-0.1)

        # At SPO2_MIN_PLAUSIBLE (0.0) -> Severe hypoxemia (<90.0)
        assert assess_spo2(0.0) == STATUS_ABNORMAL

        # Immediately above SPO2_MIN_PLAUSIBLE (0.1) -> Severe hypoxemia (<90.0)
        assert assess_spo2(0.1) == STATUS_ABNORMAL

    def test_spo2_critical_low_boundary(self):
        # Immediately below SPO2_CRITICAL_LOW (90.0) -> Severe hypoxemia (<90.0)
        assert assess_spo2(89.9) == STATUS_ABNORMAL

        # At SPO2_CRITICAL_LOW (90.0) -> Mild hypoxemia [90.0, 95.0)
        assert assess_spo2(90.0) == STATUS_LOWER

        # Immediately above SPO2_CRITICAL_LOW (90.1) -> Mild hypoxemia [90.0, 95.0)
        assert assess_spo2(90.1) == STATUS_LOWER

    def test_spo2_normal_min_boundary(self):
        # Immediately below SPO2_NORMAL_MIN (95.0) -> Mild hypoxemia [90.0, 95.0)
        assert assess_spo2(94.9) == STATUS_LOWER

        # At SPO2_NORMAL_MIN (95.0) -> Normal saturation [95.0, 100.0]
        assert assess_spo2(95.0) == STATUS_NORMAL

        # Immediately above SPO2_NORMAL_MIN (95.1) -> Normal saturation [95.0, 100.0]
        assert assess_spo2(95.1) == STATUS_NORMAL

    def test_spo2_max_plausible_boundary(self):
        # Immediately below SPO2_MAX_PLAUSIBLE (100.0) -> Normal saturation [95.0, 100.0]
        assert assess_spo2(99.9) == STATUS_NORMAL

        # At SPO2_MAX_PLAUSIBLE (100.0) -> Normal saturation [95.0, 100.0]
        assert assess_spo2(100.0) == STATUS_NORMAL

        # Immediately above SPO2_MAX_PLAUSIBLE (100.1) -> Out of range validation error
        with pytest.raises(PhysiologicalValidationError, match="plausible"):
            assess_spo2(100.1)

    # --------------------------------------------------------------------------
    # Temperature Thresholds: [25.0, 35.0, 36.1, 37.2, 38.0, 45.0]
    # --------------------------------------------------------------------------
    def test_temp_min_plausible_boundary(self):
        # Immediately below TEMP_MIN_PLAUSIBLE (25.0) -> Out of range validation error
        with pytest.raises(PhysiologicalValidationError, match="plausible"):
            assess_temperature(24.9)

        # At TEMP_MIN_PLAUSIBLE (25.0) -> Severe hypothermia (<35.0)
        assert assess_temperature(25.0) == STATUS_ABNORMAL

        # Immediately above TEMP_MIN_PLAUSIBLE (25.1) -> Severe hypothermia (<35.0)
        assert assess_temperature(25.1) == STATUS_ABNORMAL

    def test_temp_critical_low_boundary(self):
        # Immediately below TEMP_CRITICAL_LOW (35.0) -> Severe hypothermia (<35.0)
        assert assess_temperature(34.9) == STATUS_ABNORMAL

        # At TEMP_CRITICAL_LOW (35.0) -> Mild hypothermia [35.0, 36.1)
        assert assess_temperature(35.0) == STATUS_LOWER

        # Immediately above TEMP_CRITICAL_LOW (35.1) -> Mild hypothermia [35.0, 36.1)
        assert assess_temperature(35.1) == STATUS_LOWER

    def test_temp_normal_min_boundary(self):
        # Immediately below TEMP_NORMAL_MIN (36.1) -> Mild hypothermia [35.0, 36.1)
        assert assess_temperature(36.0) == STATUS_LOWER

        # At TEMP_NORMAL_MIN (36.1) -> Normal adult baseline [36.1, 37.2]
        assert assess_temperature(36.1) == STATUS_NORMAL

        # Immediately above TEMP_NORMAL_MIN (36.2) -> Normal adult baseline [36.1, 37.2]
        assert assess_temperature(36.2) == STATUS_NORMAL

    def test_temp_normal_max_boundary(self):
        # Immediately below TEMP_NORMAL_MAX (37.2) -> Normal adult baseline [36.1, 37.2]
        assert assess_temperature(37.1) == STATUS_NORMAL

        # At TEMP_NORMAL_MAX (37.2) -> Normal adult baseline [36.1, 37.2]
        assert assess_temperature(37.2) == STATUS_NORMAL

        # Immediately above TEMP_NORMAL_MAX (37.3) -> Low-grade fever (37.2, 38.0]
        assert assess_temperature(37.3) == STATUS_HIGHER

    def test_temp_critical_high_boundary(self):
        # Immediately below TEMP_CRITICAL_HIGH (38.0) -> Low-grade fever (37.2, 38.0]
        assert assess_temperature(37.9) == STATUS_HIGHER

        # At TEMP_CRITICAL_HIGH (38.0) -> Low-grade fever (37.2, 38.0]
        assert assess_temperature(38.0) == STATUS_HIGHER

        # Immediately above TEMP_CRITICAL_HIGH (38.1) -> Pyrexia / High fever (>38.0)
        assert assess_temperature(38.1) == STATUS_ABNORMAL

    def test_temp_max_plausible_boundary(self):
        # Immediately below TEMP_MAX_PLAUSIBLE (45.0) -> High fever (>38.0)
        assert assess_temperature(44.9) == STATUS_ABNORMAL

        # At TEMP_MAX_PLAUSIBLE (45.0) -> High fever (>38.0)
        assert assess_temperature(45.0) == STATUS_ABNORMAL

        # Immediately above TEMP_MAX_PLAUSIBLE (45.1) -> Out of range validation error
        with pytest.raises(PhysiologicalValidationError, match="plausible"):
            assess_temperature(45.1)


class TestMissingAndIncompatibleInputs:
    """Verifies that missing values, invalid types, and booleans are cleanly rejected."""

    def test_missing_each_parameter_raises_typed_error(self):
        with pytest.raises(PhysiologicalTypeValidationError, match="Heart rate is missing"):
            analyze_vitals(heart_rate=None, spo2=98.0, temperature=36.8)

        with pytest.raises(PhysiologicalTypeValidationError, match="SpO2 is missing"):
            analyze_vitals(heart_rate=72.0, spo2=None, temperature=36.8)

        with pytest.raises(PhysiologicalTypeValidationError, match="Temperature is missing"):
            analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=None)

    def test_string_inputs_rejected(self):
        for bad_str in ["72", "72.0", "normal", "N/A", ""]:
            with pytest.raises(PhysiologicalTypeValidationError, match="numeric"):
                analyze_vitals(heart_rate=bad_str, spo2=98.0, temperature=36.8)  # type: ignore

            with pytest.raises(PhysiologicalTypeValidationError, match="numeric"):
                analyze_vitals(heart_rate=72.0, spo2=bad_str, temperature=36.8)  # type: ignore

            with pytest.raises(PhysiologicalTypeValidationError, match="numeric"):
                analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=bad_str)  # type: ignore

    def test_boolean_inputs_strictly_rejected(self):
        # Python bool inherits from int; must be explicitly rejected to avoid True->1.0 coercion
        for bool_val in [True, False, np.bool_(True), np.bool_(False)]:
            with pytest.raises(PhysiologicalTypeValidationError, match="numeric"):
                analyze_vitals(heart_rate=bool_val, spo2=98.0, temperature=36.8)  # type: ignore

            with pytest.raises(PhysiologicalTypeValidationError, match="numeric"):
                analyze_vitals(heart_rate=72.0, spo2=bool_val, temperature=36.8)  # type: ignore

            with pytest.raises(PhysiologicalTypeValidationError, match="numeric"):
                analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=bool_val)  # type: ignore

    def test_complex_and_container_types_rejected(self):
        for container in [[72.0], (72.0,), {"hr": 72.0}, {72.0}, object()]:
            with pytest.raises(PhysiologicalTypeValidationError):
                analyze_vitals(heart_rate=container, spo2=98.0, temperature=36.8)  # type: ignore


class TestNonFiniteAndOutOfRangeValidation:
    """Verifies that NaN, infinite, and out-of-range readings raise clear validation errors."""

    def test_nan_inputs_rejected_for_all_metrics(self):
        for nan_val in [float("nan"), math.nan, np.nan]:
            with pytest.raises(PhysiologicalValidationError, match="cannot be NaN"):
                analyze_vitals(heart_rate=nan_val, spo2=98.0, temperature=36.8)

            with pytest.raises(PhysiologicalValidationError, match="cannot be NaN"):
                analyze_vitals(heart_rate=72.0, spo2=nan_val, temperature=36.8)

            with pytest.raises(PhysiologicalValidationError, match="cannot be NaN"):
                analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=nan_val)

    def test_infinite_inputs_rejected_for_all_metrics(self):
        for inf_val in [float("inf"), float("-inf"), np.inf, -np.inf]:
            with pytest.raises(PhysiologicalValidationError, match="cannot be infinite"):
                analyze_vitals(heart_rate=inf_val, spo2=98.0, temperature=36.8)

            with pytest.raises(PhysiologicalValidationError, match="cannot be infinite"):
                analyze_vitals(heart_rate=72.0, spo2=inf_val, temperature=36.8)

            with pytest.raises(PhysiologicalValidationError, match="cannot be infinite"):
                analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=inf_val)

    def test_out_of_range_values_rejected(self):
        # HR bounds: [20, 300]
        with pytest.raises(PhysiologicalValidationError, match="plausible"):
            analyze_vitals(heart_rate=19.5, spo2=98.0, temperature=36.8)
        with pytest.raises(PhysiologicalValidationError, match="plausible"):
            analyze_vitals(heart_rate=300.5, spo2=98.0, temperature=36.8)

        # SpO2 bounds: [0, 100]
        with pytest.raises(PhysiologicalValidationError, match="plausible"):
            analyze_vitals(heart_rate=72.0, spo2=-0.5, temperature=36.8)
        with pytest.raises(PhysiologicalValidationError, match="plausible"):
            analyze_vitals(heart_rate=72.0, spo2=100.5, temperature=36.8)

        # Temperature bounds: [25, 45]
        with pytest.raises(PhysiologicalValidationError, match="plausible"):
            analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=24.5)
        with pytest.raises(PhysiologicalValidationError, match="plausible"):
            analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=45.5)


class TestMultipleAbnormalReadingsAndEscalation:
    """Verifies multi-metric aggregation, conflict escalation, and absence of cancellation."""

    def test_two_abnormal_metrics(self):
        # Tachycardia + Fever
        res = analyze_vitals(heart_rate=135.0, spo2=98.0, temperature=38.6)
        assert res["heart_rate"]["status"] == STATUS_ABNORMAL
        assert res["temperature"]["status"] == STATUS_ABNORMAL
        assert res["spo2"]["status"] == STATUS_NORMAL
        assert res["overall_status"] == STATUS_ABNORMAL
        assert res["risk_level"] == RISK_HIGH
        assert res["is_abnormal"] is True

    def test_three_critical_metrics(self):
        # Severe bradycardia + Severe hypoxemia + Hypothermia
        res = analyze_vitals(heart_rate=32.0, spo2=84.0, temperature=34.1)
        assert res["heart_rate"]["status"] == STATUS_ABNORMAL
        assert res["spo2"]["status"] == STATUS_ABNORMAL
        assert res["temperature"]["status"] == STATUS_ABNORMAL
        assert res["overall_status"] == STATUS_ABNORMAL
        assert res["risk_level"] == RISK_HIGH
        assert res["is_abnormal"] is True

    def test_divergent_mild_deviations_escalate_to_abnormal(self):
        # HR elevated (110 -> Higher than usual) and SpO2 depressed (92% -> Lower than usual)
        res = analyze_vitals(heart_rate=110.0, spo2=92.0, temperature=36.8)
        assert res["heart_rate"]["status"] == STATUS_HIGHER
        assert res["spo2"]["status"] == STATUS_LOWER
        assert res["temp_status"] == STATUS_NORMAL
        assert res["overall_status"] == STATUS_ABNORMAL
        assert res["risk_level"] == RISK_HIGH

        # HR depressed (52 -> Lower than usual) and Temp elevated (37.5 -> Higher than usual)
        res2 = analyze_vitals(heart_rate=52.0, spo2=98.0, temperature=37.5)
        assert res2["heart_rate"]["status"] == STATUS_LOWER
        assert res2["temp_status"] == STATUS_HIGHER
        assert res2["overall_status"] == STATUS_ABNORMAL
        assert res2["risk_level"] == RISK_HIGH

    def test_isolated_mild_elevation_remains_moderate(self):
        # Single elevated metric
        res = analyze_vitals(heart_rate=112.0, spo2=98.0, temperature=36.8)
        assert res["overall_status"] == STATUS_HIGHER
        assert res["risk_level"] == RISK_MODERATE

        # Single depressed metric
        res2 = analyze_vitals(heart_rate=72.0, spo2=93.0, temperature=36.8)
        assert res2["overall_status"] == STATUS_LOWER
        assert res2["risk_level"] == RISK_MODERATE


class TestDeterminismAndConcurrency:
    """Verifies that analysis is purely deterministic and safe across concurrent executions."""

    def test_repeated_calls_are_bitwise_identical(self):
        runs = [analyze_vitals(heart_rate=78.4, spo2=97.2, temperature=36.75) for _ in range(50)]
        first = runs[0]
        for run in runs[1:]:
            assert run == first

    def test_multithreaded_concurrency(self):
        test_inputs = [
            (72.0, 98.0, 36.8),
            (130.0, 98.0, 36.8),
            (72.0, 88.0, 36.8),
            (72.0, 98.0, 39.0),
            (110.0, 93.0, 36.8),
        ] * 10

        def _worker(inp):
            hr, spo2, temp = inp
            return analyze_vitals(heart_rate=hr, spo2=spo2, temperature=temp)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(_worker, test_inputs))

        assert len(results) == len(test_inputs)
        for res, inp in zip(results, test_inputs):
            assert res["heart_rate"]["value"] == round(inp[0], 1)


class TestModalityIndependence:
    """Verifies complete independence between vital-sign analysis and eye-disease inference."""

    def test_pure_vitals_execution_without_torch_model(self):
        # Can execute on raw numbers without any model checkpoints or images
        res = analyze_vitals(heart_rate=75.0, spo2=99.0, temperature=36.6)
        assert res["overall_status"] == STATUS_NORMAL
        assert res["risk_level"] == RISK_LOW

    def test_pipeline_vitals_survive_image_absence(self):
        # When image_path is None, screening pipeline executes vitals and skips eye model
        res = run_screening(image_path=None, heart_rate=74.0, spo2=98.0, temperature=36.8)
        assert res["status"] == "partial"
        assert res["vitals_screening"]["status"] == "success"
        assert res["vitals_screening"]["overall_status"] == "Normal"
        assert res["eye_screening"]["status"] == "skipped"

    def test_pipeline_vitals_survive_image_absence_and_error(self):
        # A nonexistent image file triggers eye error but preserves valid vital signs
        res = run_screening(image_path="nonexistent_retina_image.png", heart_rate=76.0, spo2=98.5, temperature=36.7)
        assert res["vitals_screening"]["status"] == "success"
        assert res["vitals_screening"]["overall_status"] == "Normal"
        assert res["eye_screening"]["status"] == "error"

    def test_pipeline_vitals_survive_image_quality_rejection(self, tmp_path):
        # Create an intentionally blank/flat image that fails the quality gate (low contrast/blur)
        from PIL import Image
        flat_img = Image.new("RGB", (224, 224), color=(50, 50, 50))
        flat_path = str(tmp_path / "blurred_flat.png")
        flat_img.save(flat_path)

        res = run_screening(image_path=flat_path, heart_rate=76.0, spo2=98.5, temperature=36.7)
        assert res["vitals_screening"]["status"] == "success"
        assert res["vitals_screening"]["overall_status"] == "Normal"
        assert res["eye_screening"]["status"] == "rejected"

    def test_pipeline_eye_survives_vitals_error(self):
        # An invalid vital sign does not prevent eye-screening if image is valid or skipped
        res = run_screening(image_path=None, heart_rate=500.0, spo2=98.0, temperature=36.8)
        assert res["vitals_screening"]["status"] == "error"
        assert res["vitals_screening"]["risk_level"] == "UNKNOWN"
        assert "plausible" in res["vitals_screening"]["reason"]


class TestResultSemanticsAndUnits:
    """Verifies that units are explicit, errors are distinct from abnormalities, and disclaimers are clear."""

    def test_explicit_units_in_output(self):
        res = analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=36.8)
        assert res["heart_rate"]["unit"] == "BPM"
        assert res["spo2"]["unit"] == "%"
        assert res["temperature"]["unit"] == "°C"

    def test_validation_errors_distinct_from_abnormalities(self):
        # Validation failure with raise_errors=False
        err_res = analyze_vitals(heart_rate=500.0, spo2=98.0, temperature=36.8, raise_errors=False)
        assert err_res["status"] == "error"
        assert err_res["overall_status"] == "Invalid"
        assert err_res["risk_level"] == "UNKNOWN"

        # Legitimate physiological abnormality
        abnormal_res = analyze_vitals(heart_rate=140.0, spo2=98.0, temperature=36.8, raise_errors=False)
        assert abnormal_res["status"] == "success"
        assert abnormal_res["overall_status"] == STATUS_ABNORMAL
        assert abnormal_res["risk_level"] == RISK_HIGH

    def test_disclaimer_clarifies_normal_does_not_rule_out_disease(self):
        res = analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=36.8)
        disclaimer = res["disclaimer"]
        assert "Preliminary screening assessment" in disclaimer
        assert "Normal vital signs do not rule out" in disclaimer
        assert "does not constitute an authoritative clinical diagnosis" in disclaimer
