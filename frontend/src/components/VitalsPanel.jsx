import React from 'react';
import {
  IconHeartPulse,
  IconOxygen,
  IconThermometer,
  IconActivity,
  IconAlertCircle,
  IconCheck,
} from './MedicalIcons';

export default function VitalsPanel({ vitalsData, isHardwareConnected = true }) {
  const vitals = vitalsData?.analysis || {
    heart_rate: { value: 78.5, unit: 'BPM', status: 'Normal' },
    spo2: { value: 97.2, unit: '%', status: 'Normal' },
    temperature: { value: 36.8, unit: '°C', status: 'Normal' },
    overall_status: 'Normal',
  };

  const hrValue = vitals.heart_rate?.value ?? '--';
  const hrStatus = vitals.heart_rate?.status ?? 'Pending';
  const spo2Value = vitals.spo2?.value ?? '--';
  const spo2Status = vitals.spo2?.status ?? 'Pending';
  const tempValue = vitals.temperature?.value ?? '--';
  const tempStatus = vitals.temperature?.status ?? 'Pending';
  const overall = vitals.overall_status ?? 'Normal';

  const getStatusBadge = (status) => {
    if (status === 'Normal') {
      return (
        <span className="inline-flex items-center text-[11px] font-semibold px-2 py-0.5 rounded border bg-emerald-50 text-emerald-800 border-emerald-200">
          <IconCheck className="w-3 h-3 mr-1 text-emerald-600" />
          Normal
        </span>
      );
    }
    return (
      <span className="inline-flex items-center text-[11px] font-semibold px-2 py-0.5 rounded border bg-rose-50 text-rose-800 border-rose-200">
        <IconAlertCircle className="w-3 h-3 mr-1 text-rose-600" />
        {status}
      </span>
    );
  };

  return (
    <div className="bg-white border border-slate-200 rounded p-5">
      {/* Header with Hardware Telemetry Status */}
      <div className="flex flex-wrap items-center justify-between pb-3 mb-4 border-b border-slate-100 gap-2">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center">
            <IconActivity className="w-4 h-4 mr-2 text-sky-600" />
            3. Physiological Telemetry Monitoring
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Real-time biometric data stream with automated XGBoost status classification
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1.5 text-xs text-slate-600">
            <span
              className={`w-2 h-2 rounded-full ${
                isHardwareConnected ? 'bg-emerald-500' : 'bg-slate-400'
              }`}
            />
            <span>
              ESP32 Sensors: {isHardwareConnected ? 'Connected (MAX30102 / MLX90614)' : 'Offline'}
            </span>
          </div>

          <div className="text-xs font-semibold">
            Overall: {getStatusBadge(overall)}
          </div>
        </div>
      </div>

      {/* Structured Telemetry Grid with Clinical Reference Standards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Heart Rate Metric */}
        <div className="p-3.5 bg-slate-50 border border-slate-200 rounded">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-600 flex items-center">
              <IconHeartPulse className="w-4 h-4 mr-1.5 text-rose-600" />
              Heart Rate (PPG)
            </span>
            {getStatusBadge(hrStatus)}
          </div>
          <div className="mt-2 flex items-baseline">
            <span className="text-2xl font-bold font-mono text-slate-900">{hrValue}</span>
            <span className="text-xs font-semibold text-slate-500 ml-1.5">BPM</span>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-200 text-[10px] text-slate-500 flex justify-between">
            <span>Clinical Range: 60 to 100</span>
            <span className="font-mono">Resting Baseline</span>
          </div>
        </div>

        {/* SpO2 Oxygen Saturation */}
        <div className="p-3.5 bg-slate-50 border border-slate-200 rounded">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-600 flex items-center">
              <IconOxygen className="w-4 h-4 mr-1.5 text-sky-600" />
              Blood Oxygen (SpO2)
            </span>
            {getStatusBadge(spo2Status)}
          </div>
          <div className="mt-2 flex items-baseline">
            <span className="text-2xl font-bold font-mono text-slate-900">{spo2Value}</span>
            <span className="text-xs font-semibold text-slate-500 ml-1.5">%</span>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-200 text-[10px] text-slate-500 flex justify-between">
            <span>Normal Target: ≥95%</span>
            <span className="font-mono">Pulse Oximetry</span>
          </div>
        </div>

        {/* Body Temperature */}
        <div className="p-3.5 bg-slate-50 border border-slate-200 rounded">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-600 flex items-center">
              <IconThermometer className="w-4 h-4 mr-1.5 text-amber-600" />
              Body Temperature
            </span>
            {getStatusBadge(tempStatus)}
          </div>
          <div className="mt-2 flex items-baseline">
            <span className="text-2xl font-bold font-mono text-slate-900">{tempValue}</span>
            <span className="text-xs font-semibold text-slate-500 ml-1.5">°C</span>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-200 text-[10px] text-slate-500 flex justify-between">
            <span>Afebrile: 36.1 to 37.2°C</span>
            <span className="font-mono">Infrared Forehead</span>
          </div>
        </div>
      </div>
    </div>
  );
}
