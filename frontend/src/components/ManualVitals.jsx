import React from "react";

export default function ManualVitals({
  heartRate,
  setHeartRate,
  spo2,
  setSpo2,
  temperature,
  setTemperature,
  disabled,
}) {
  const applyPreset = (hr, ox, temp) => {
    setHeartRate(hr);
    setSpo2(ox);
    setTemperature(temp);
  };

  const clearVitals = () => {
    setHeartRate("");
    setSpo2("");
    setTemperature("");
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
      <div className="flex justify-between items-center mb-3">
        <div>
          <h2 className="font-semibold text-slate-900 text-base flex items-center space-x-2">
            <span>💓 Physiological Vital Measurements</span>
          </h2>
          <p className="text-xs text-slate-500">
            Manual triage inputs or ESP32 sensor simulation. Analyzed independently of retinal imaging.
          </p>
        </div>
        {(heartRate || spo2 || temperature) && !disabled && (
          <button
            onClick={clearVitals}
            className="text-xs text-slate-500 hover:text-slate-700 font-medium px-2 py-1 rounded bg-slate-100"
          >
            Clear Vitals
          </button>
        )}
      </div>

      {/* Input Fields */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        <div>
          <label className="block text-xs font-semibold text-slate-700 mb-1">
            Heart Rate (BPM)
          </label>
          <div className="relative">
            <input
              type="number"
              min="30"
              max="220"
              step="1"
              placeholder="e.g. 78"
              value={heartRate}
              onChange={(e) => setHeartRate(e.target.value)}
              disabled={disabled}
              className="w-full px-3 py-2 text-sm bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white font-mono"
            />
            <span className="absolute right-3 top-2 text-xs text-slate-400 pointer-events-none">
              BPM
            </span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Normal: 60 - 100 BPM</span>
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-700 mb-1">
            SpO2 Saturation (%)
          </label>
          <div className="relative">
            <input
              type="number"
              min="50"
              max="100"
              step="0.5"
              placeholder="e.g. 98"
              value={spo2}
              onChange={(e) => setSpo2(e.target.value)}
              disabled={disabled}
              className="w-full px-3 py-2 text-sm bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white font-mono"
            />
            <span className="absolute right-3 top-2 text-xs text-slate-400 pointer-events-none">
              %
            </span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Normal: 95 - 100%</span>
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-700 mb-1">
            Body Temperature (°C)
          </label>
          <div className="relative">
            <input
              type="number"
              min="30"
              max="45"
              step="0.1"
              placeholder="e.g. 36.8"
              value={temperature}
              onChange={(e) => setTemperature(e.target.value)}
              disabled={disabled}
              className="w-full px-3 py-2 text-sm bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white font-mono"
            />
            <span className="absolute right-3 top-2 text-xs text-slate-400 pointer-events-none">
              °C
            </span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Normal: 36.1 - 37.2°C</span>
        </div>
      </div>

      {/* Preset Quick Fill Buttons */}
      <div className="pt-2 border-t border-slate-100 flex flex-wrap items-center gap-2">
        <span className="text-[11px] font-medium text-slate-500 mr-1">Presets:</span>
        <button
          type="button"
          onClick={() => applyPreset("75", "98", "36.8")}
          disabled={disabled}
          className="text-xs px-2.5 py-1 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors"
        >
          Normal (75 BPM, 98%, 36.8°C)
        </button>
        <button
          type="button"
          onClick={() => applyPreset("115", "97", "38.4")}
          disabled={disabled}
          className="text-xs px-2.5 py-1 rounded bg-amber-50 hover:bg-amber-100 text-amber-800 transition-colors"
        >
          Elevated HR & Fever (115 BPM, 38.4°C)
        </button>
        <button
          type="button"
          onClick={() => applyPreset("88", "88", "37.0")}
          disabled={disabled}
          className="text-xs px-2.5 py-1 rounded bg-rose-50 hover:bg-rose-100 text-rose-800 transition-colors"
        >
          Hypoxia Alert (SpO2 88%)
        </button>
      </div>
    </div>
  );
}
