"""
Tests for Physiological Analysis Module (ml/physio_model.py).

Verifies:
1. Representative normal readings across standard reference baselines.
2. Individual abnormal and elevated measurements.
3. Multiple simultaneous abnormal measurements (no cancellation by normal vitals).
4. Strict threshold boundary behaviors.
5. Missing, non-numeric, NaN, infinite, and impossible inputs.
6. Schema contract, data types, and JSON serializability.
7. Pure determinism across repeated invocations.
8. Module independence and isolation from the vision/eye model pipeline.
"""

import json
import math
import sys
import pytest

from ml.physio_model import (
    HR_CRITICAL_HIGH,
    HR_CRITICAL_LOW,
    HR_NORMAL_MAX,
    HR_NORMAL_MIN,
    RISK_HIGH,
    RISK_LOW,
    RISK_MODERATE,
    SPO2_CRITICAL_LOW,
    SPO2_NORMAL_MIN,
    STATUS_ABNORMAL,
    STATUS_HIGHER,
    STATUS_LOWER,
    STATUS_NORMAL,
    TEMP_CRITICAL_HIGH,
    TEMP_CRITICAL_LOW,
    TEMP_NORMAL_MAX,
    TEMP_NORMAL_MIN,
    analyze_vitals,
    assess_heart_rate,
    assess_overall_status,
    assess_spo2,
    assess_temperature,
)


class TestRepresentativeNormalReadings:
    """Verifies standard healthy adult vital sign screening profiles."""

    def test_standard_normal_vitals(self):
        res = analyze_vitals(heart_rate=72.0, spo2=98.0, temperature=36.8)

        assert res["overall_status"] == STATUS_NORMAL
        assert res["risk_level"] == RISK_LOW
        assert res["is_abnormal"] is False
        assert res["heart_rate"]["status"] == STATUS_NORMAL
        assert res["spo2"]["status"] == STATUS_NORMAL
        assert res["temperature"]["status"] == STATUS_NORMAL
        assert res["heart_rate"]["value"] == 72.0
        assert res["heart_rate"]["unit"] == "BPM"
        assert res["spo2"]["value"] == 98.0
        assert res["spo2"]["unit"] == "%"
        assert res["temperature"]["value"] == 36.8
        assert res["temperature"]["unit"] == "°C"
        assert "within normal" in res["summary"]

    def test_integer_inputs_coerced_properly(self):
        res = analyze_vitals(heart_rate=80, spo2=97, temperature=37)
        assert res["overall_status"] == STATUS_NORMAL
        assert isinstance(res["heart_rate"]["value"], float)


class TestIndividualAbnormalMeasurements:
    """Verifies that an individual abnormal vital sign triggers appropriate status and risk."""

    def test_severe_bradycardia(self):
        # HR < 40 -> Abnormal
        res = analyze_vitals(heart_rate=35.0, spo2=98.0, temperature=36.8)
        assert res["heart_rate"]["status"] == STATUS_ABNORMAL
        assert res["overall_status"] == STATUS_ABNORMAL
        assert res["risk_level"] == RISK_HIGH
        assert res["is_abnormal"] is True

    def test_severe_tachycardia(self):
        # HR > 120 -> Abnormal
        res = analyze_vitals(heart_rate=135.0, spo2=98.0, temperature=36.8)
        assert res["heart_rate"]["status"] == STATUS_ABNORMAL
        assert res["overall_status"] == STATUS_ABNORMAL
        assert res["risk_level"] == RISK_HIGH

    def test_severe_hypoxemia(self):
        # SpO2 < 90 -> Abnormal
        res = analyze_vitals(heart_rate=75.0, spo2=86.0, temperature=36.8)
        assert res["spo2"]["status"] == STATUS_ABNORMAL
        assert res["overall_status"] == STATUS_ABNORMAL
        assert res["risk_level"] == RISK_HIGH

    def test_severe_hypothermia(self):
        # Temp < 35.0 -> Abnormal
        res = analyze_vitals(heart_rate=75.0, spo2=98.0, temperature=34.2)
        assert res["temperature"]["status"] == STATUS_ABNORMAL
        assert res["overall_status"] == STATUS_ABNORMAL
        assert res["risk_level"] == RISK_HIGH

    def test_pyrexia_high_fever(self):
        # Temp > 38.0 -> Abnormal
        res = analyze_vitals(heart_rate=75.0, spo2=98.0, temperature=39.1)
        assert res["temperature"]["status"] == STATUS_ABNORMAL
        assert res["overall_status"] == STATUS_ABNORMAL
        assert res["risk_level"] == RISK_HIGH

    def test_mild_elevations_warning_statuses(self):
        # Mild tachycardia: 100 < hr <= 120
        res_hr = analyze_vitals(heart_rate=112.0, spo2=98.0, temperature=36.8)
        assert res_hr["heart_rate"]["status"] == STATUS_HIGHER
        assert res_hr["overall_status"] == STATUS_HIGHER
        assert res_hr["risk_level"] == RISK_MODERATE

        # Mild bradycardia: 40 <= hr < 60
        res_brady = analyze_vitals(heart_rate=52.0, spo2=98.0, temperature=36.8)
        assert res_brady["heart_rate"]["status"] == STATUS_LOWER
        assert res_brady["overall_status"] == STATUS_LOWER
        assert res_brady["risk_level"] == RISK_MODERATE

        # Mild hypoxemia: 90 <= spo2 < 95
        res_spo2 = analyze_vitals(heart_rate=75.0, spo2=92.0, temperature=36.8)
        assert res_spo2["spo2"]["status"] == STATUS_LOWER
        assert res_spo2["overall_status"] == STATUS_LOWER
        assert res_spo2["risk_level"] == RISK_MODERATE

        # Low-grade fever: 37.2 < temp <= 38.0
        res_temp = analyze_vitals(heart_rate=75.0, spo2=98.0, temperature=37.6)
        assert res_temp["temperature"]["status"] == STATUS_HIGHER
        assert res_temp["overall_status"] == STATUS_HIGHER
        assert res_temp["risk_level"] == RISK_MODERATE


class TestMultipleAbnormalMeasurements:
    """Verifies that normal vitals never mask abnormalities and multiple issues aggregate safely."""

    def test_abnormal_not_cancelled_by_normal(self):
        # 1 Abnormal (SpO2=87%), 2 Normal (HR=72, Temp=36.8)
        res = analyze_vitals(heart_rate=72.0, spo2=87.0, temperature=36.8)
        assert res["overall_status"] == STATUS_ABNORMAL
        assert res["risk_level"] == RISK_HIGH
        # Individual normal measurements retain their individual normal label
        assert res["heart_rate"]["status"] == STATUS_NORMAL
        assert res["temperature"]["status"] == STATUS_NORMAL

    def test_multiple_critical_vitals(self):
        # Tachycardia + Hypoxemia + Fever
        res = analyze_vitals(heart_rate=130.0, spo2=88.0, temperature=38.9)
        assert res["heart_rate"]["status"] == STATUS_ABNORMAL
        assert res["spo2"]["status"] == STATUS_ABNORMAL
        assert res["temperature"]["status"] == STATUS_ABNORMAL
        assert res["overall_status"] == STATUS_ABNORMAL
        assert res["risk_level"] == RISK_HIGH

    def test_conflicting_mild_deviations_escalate(self):
        # HR elevated (Higher than usual) and SpO2 depressed (Lower than usual)
        res = analyze_vitals(heart_rate=110.0, spo2=92.0, temperature=36.8)
        assert res["heart_rate"]["status"] == STATUS_HIGHER
        assert res["spo2"]["status"] == STATUS_LOWER
        assert res["overall_status"] == STATUS_ABNORMAL


class TestThresholdBoundaries:
    """Verifies exact inclusive/exclusive boundary conditions for all metrics."""

    def test_heart_rate_exact_boundaries(self):
        # Critical low: < 40.0
        assert assess_heart_rate(39.9) == STATUS_ABNORMAL
        assert assess_heart_rate(40.0) == STATUS_LOWER
        # Lower than usual: [40.0, 60.0)
        assert assess_heart_rate(59.9) == STATUS_LOWER
        assert assess_heart_rate(60.0) == STATUS_NORMAL
        # Normal: [60.0, 100.0]
        assert assess_heart_rate(100.0) == STATUS_NORMAL
        assert assess_heart_rate(100.1) == STATUS_HIGHER
        # Higher than usual: (100.0, 120.0]
        assert assess_heart_rate(120.0) == STATUS_HIGHER
        assert assess_heart_rate(120.1) == STATUS_ABNORMAL

    def test_spo2_exact_boundaries(self):
        # Critical low: < 90.0
        assert assess_spo2(89.9) == STATUS_ABNORMAL
        assert assess_spo2(90.0) == STATUS_LOWER
        # Lower than usual: [90.0, 95.0)
        assert assess_spo2(94.9) == STATUS_LOWER
        assert assess_spo2(95.0) == STATUS_NORMAL
        # Normal: [95.0, 100.0]
        assert assess_spo2(100.0) == STATUS_NORMAL

    def test_temperature_exact_boundaries(self):
        # Critical low: < 35.0
        assert assess_temperature(34.9) == STATUS_ABNORMAL
        assert assess_temperature(35.0) == STATUS_LOWER
        # Lower than usual: [35.0, 36.1)
        assert assess_temperature(36.0) == STATUS_LOWER
        assert assess_temperature(36.1) == STATUS_NORMAL
        # Normal: [36.1, 37.2]
        assert assess_temperature(37.2) == STATUS_NORMAL
        assert assess_temperature(37.3) == STATUS_HIGHER
        # Higher than usual: (37.2, 38.0]
        assert assess_temperature(38.0) == STATUS_HIGHER
        assert assess_temperature(38.1) == STATUS_ABNORMAL


class TestInvalidAndMissingInputs:
    """Verifies robust validation against bad types, missing data, and invalid numbers."""

    def test_missing_none_raises_typeerror(self):
        with pytest.raises(TypeError, match="missing"):
            analyze_vitals(heart_rate=None, spo2=98.0, temperature=36.8)
        with pytest.raises(TypeError, match="missing"):
            analyze_vitals(heart_rate=75.0, spo2=None, temperature=36.8)
        with pytest.raises(TypeError, match="missing"):
            analyze_vitals(heart_rate=75.0, spo2=98.0, temperature=None)

    def test_string_raises_typeerror(self):
        with pytest.raises(TypeError, match="numeric"):
            analyze_vitals(heart_rate="75", spo2=98.0, temperature=36.8)

    def test_boolean_raises_typeerror(self):
        # Python bool is an int subclass; must reject bool specifically
        with pytest.raises(TypeError, match="numeric"):
            analyze_vitals(heart_rate=True, spo2=98.0, temperature=36.8)

    def test_nan_values_raise_valueerror(self):
        with pytest.raises(ValueError, match="NaN"):
            analyze_vitals(heart_rate=float("nan"), spo2=98.0, temperature=36.8)
        with pytest.raises(ValueError, match="NaN"):
            analyze_vitals(heart_rate=75.0, spo2=float("nan"), temperature=36.8)

    def test_infinite_values_raise_valueerror(self):
        with pytest.raises(ValueError, match="infinite"):
            analyze_vitals(heart_rate=float("inf"), spo2=98.0, temperature=36.8)
        with pytest.raises(ValueError, match="infinite"):
            analyze_vitals(heart_rate=75.0, spo2=98.0, temperature=float("-inf"))

    def test_impossible_readings_raise_valueerror(self):
        # Heart rate: outside [20, 300]
        with pytest.raises(ValueError, match="plausible"):
            analyze_vitals(heart_rate=10.0, spo2=98.0, temperature=36.8)
        with pytest.raises(ValueError, match="plausible"):
            analyze_vitals(heart_rate=350.0, spo2=98.0, temperature=36.8)

        # SpO2: outside [0, 100]
        with pytest.raises(ValueError, match="plausible"):
            analyze_vitals(heart_rate=75.0, spo2=-5.0, temperature=36.8)
        with pytest.raises(ValueError, match="plausible"):
            analyze_vitals(heart_rate=75.0, spo2=105.0, temperature=36.8)

        # Temperature: outside [25, 45]
        with pytest.raises(ValueError, match="plausible"):
            analyze_vitals(heart_rate=75.0, spo2=98.0, temperature=15.0)
        with pytest.raises(ValueError, match="plausible"):
            analyze_vitals(heart_rate=75.0, spo2=98.0, temperature=50.0)

    def test_safe_mode_without_raising_errors(self):
        res = analyze_vitals(heart_rate=-10.0, spo2=98.0, temperature=36.8, raise_errors=False)
        assert res["status"] == "error"
        assert res["overall_status"] == "Invalid"
        assert "error_message" in res


class TestReturnSchemaAndSerialization:
    """Verifies contract keys, units, and JSON serialization."""

    def test_contract_schema_keys(self):
        res = analyze_vitals(heart_rate=75.0, spo2=97.0, temperature=36.7)
        required_keys = [
            "heart_rate",
            "spo2",
            "temperature",
            "hr_status",
            "spo2_status",
            "temp_status",
            "overall_status",
            "risk_level",
            "is_abnormal",
            "summary",
            "disclaimer",
        ]
        for k in required_keys:
            assert k in res, f"Missing key: {k}"

        assert "value" in res["heart_rate"]
        assert "status" in res["heart_rate"]
        assert "unit" in res["heart_rate"]

    def test_json_serializability(self):
        res = analyze_vitals(heart_rate=76.4, spo2=98.2, temperature=36.9)
        json_str = json.dumps(res)
        deserialized = json.loads(json_str)
        assert deserialized["overall_status"] == STATUS_NORMAL
        assert deserialized["heart_rate"]["value"] == 76.4

    def test_deterministic_repeated_calls(self):
        res1 = analyze_vitals(heart_rate=88.5, spo2=94.2, temperature=37.7)
        res2 = analyze_vitals(heart_rate=88.5, spo2=94.2, temperature=37.7)
        assert res1 == res2


class TestModuleIndependence:
    """Verifies that physio_model is strictly decoupled from the vision/eye model pipeline."""

    def test_no_torch_or_cv2_dependency_in_physio_module(self):
        import ml.physio_model as pm
        # Check module dictionary does not import heavy vision modules
        assert not hasattr(pm, "torch")
        assert not hasattr(pm, "cv2")
        assert not hasattr(pm, "torchvision")

    def test_analyze_vitals_callable_without_checkpoints_or_images(self):
        # Should execute purely on numeric primitives without file I/O or checkpoints
        res = analyze_vitals(heart_rate=70.0, spo2=99.0, temperature=36.6)
        assert res["overall_status"] == STATUS_NORMAL
