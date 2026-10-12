import React, { useState } from 'react';
import { IconActivity, IconRefresh, IconCheck } from './MedicalIcons';

export default function ManualVitals({ onSubmit, isSubmitting }) {
  const [hr, setHr] = useState('78');
  const [spo2, setSpo2] = useState('98');
  const [temp, setTemp] = useState('36.8');
  const [successNotice, setSuccessNotice] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!hr || !spo2 || !temp) return;
    onSubmit(hr, spo2, temp);
    setSuccessNotice(true);
    setTimeout(() => setSuccessNotice(false), 3000);
  };

  const handleApplyPreset = (presetHr, presetSpo2, presetTemp) => {
    setHr(presetHr);
    setSpo2(presetSpo2);
    setTemp(presetTemp);
  };

  return (
    <div className="bg-white border border-slate-200 rounded p-5">
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center">
            <IconActivity className="w-4 h-4 mr-2 text-sky-600" />
            4. Manual Physiological Intake Fallback
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Operator manual override when hardware sensors are disconnected
          </p>
        </div>
        {successNotice && (
          <span className="text-[11px] font-semibold text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-300 flex items-center">
            <IconCheck className="w-3.5 h-3.5 mr-1" />
            Telemetry Updated
          </span>
        )}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* Heart Rate Field */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Heart Rate (BPM)
            </label>
            <input
              type="number"
              step="0.1"
              min="30"
              max="220"
              required
              value={hr}
              onChange={(e) => setHr(e.target.value)}
              placeholder="e.g. 78"
              className="w-full px-3 py-1.5 text-xs bg-slate-50 border border-slate-300 rounded text-slate-900 focus:outline-none focus:border-sky-600 font-mono"
            />
            <span className="text-[10px] text-slate-400 mt-0.5 block">Standard: 60-100 BPM</span>
          </div>

          {/* SpO2 Field */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              SpO2 Saturation (%)
            </label>
            <input
              type="number"
              step="0.1"
              min="60"
              max="100"
              required
              value={spo2}
              onChange={(e) => setSpo2(e.target.value)}
              placeholder="e.g. 98"
              className="w-full px-3 py-1.5 text-xs bg-slate-50 border border-slate-300 rounded text-slate-900 focus:outline-none focus:border-sky-600 font-mono"
            />
            <span className="text-[10px] text-slate-400 mt-0.5 block">Standard: ≥95%</span>
          </div>

          {/* Temperature Field */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Temperature (°C)
            </label>
            <input
              type="number"
              step="0.1"
              min="33"
              max="43"
              required
              value={temp}
              onChange={(e) => setTemp(e.target.value)}
              placeholder="e.g. 36.8"
              className="w-full px-3 py-1.5 text-xs bg-slate-50 border border-slate-300 rounded text-slate-900 focus:outline-none focus:border-sky-600 font-mono"
            />
            <span className="text-[10px] text-slate-400 mt-0.5 block">Standard: 36.1-37.2°C</span>
          </div>
        </div>

        {/* Action Row & Quick Presets */}
        <div className="flex flex-wrap items-center justify-between pt-2 border-t border-slate-100 gap-2">
          <div className="flex items-center space-x-1.5 text-xs">
            <span className="text-slate-500 text-[11px]">Quick Clinical Presets:</span>
            <button
              type="button"
              onClick={() => handleApplyPreset('74', '98', '36.6')}
              className="px-2 py-0.5 text-[11px] rounded bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition-colors"
            >
              Baseline Normal
            </button>
            <button
              type="button"
              onClick={() => handleApplyPreset('108', '93', '37.1')}
              className="px-2 py-0.5 text-[11px] rounded bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition-colors"
            >
              Hypoxia & Tachycardia
            </button>
            <button
              type="button"
              onClick={() => handleApplyPreset('88', '97', '38.6')}
              className="px-2 py-0.5 text-[11px] rounded bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition-colors"
            >
              Febrile Pyrexia
            </button>
          </div>

          <button
            type="submit"
            disabled={isSubmitting}
            className={`px-4 py-1.5 text-xs font-semibold rounded text-white transition-colors flex items-center ${
              isSubmitting
                ? 'bg-slate-400 cursor-not-allowed'
                : 'bg-slate-800 hover:bg-slate-900'
            }`}
          >
            {isSubmitting ? (
              <>
                <IconRefresh className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                Submitting Vitals...
              </>
            ) : (
              'Ingest Manual Vitals'
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
