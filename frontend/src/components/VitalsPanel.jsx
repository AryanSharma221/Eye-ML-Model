import React from "react";

export default function VitalsPanel({ vitalsData }) {
  if (!vitalsData || vitalsData.status !== "success") {
    return null;
  }

  const {
    heart_rate,
    spo2,
    temperature,
    hr_status,
    spo2_status,
    temp_status,
    overall_status,
    risk_level,
    summary,
  } = vitalsData;

  const hrValue = typeof heart_rate === "object" ? heart_rate?.value : heart_rate;
  const spo2Value = typeof spo2 === "object" ? spo2?.value : spo2;
  const tempValue = typeof temperature === "object" ? temperature?.value : temperature;

  const getStatusBadge = (statusStr) => {
    const s = (statusStr || "").toLowerCase();
    if (s.includes("normal")) {
      return { bg: "bg-emerald-100 text-emerald-800 border-emerald-200", icon: "✓" };
    }
    if (s.includes("higher") || s.includes("elevated") || s.includes("warning")) {
      return { bg: "bg-amber-100 text-amber-800 border-amber-200", icon: "⚠️" };
    }
    if (s.includes("lower") || s.includes("low")) {
      return { bg: "bg-blue-100 text-blue-800 border-blue-200", icon: "↓" };
    }
    return { bg: "bg-rose-100 text-rose-800 border-rose-200", icon: "!" };
  };

  const getOverallRiskBadge = (risk) => {
    switch (risk) {
      case "HIGH":
        return "bg-rose-100 text-rose-800 border-rose-300";
      case "MODERATE":
        return "bg-amber-100 text-amber-800 border-amber-300";
      case "LOW":
        return "bg-emerald-100 text-emerald-800 border-emerald-300";
      default:
        return "bg-slate-100 text-slate-800 border-slate-300";
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
      <div className="flex justify-between items-start mb-4">
        <div>
          <span className="text-xs font-semibold text-blue-600 uppercase tracking-wider block">
            Independent Physiological Stream
          </span>
          <h3 className="text-xl font-bold text-slate-900 mt-0.5">
            Vital Signs Assessment
          </h3>
        </div>
        <span
          className={`px-3 py-1 rounded-full text-xs font-bold border tracking-wide ${getOverallRiskBadge(
            risk_level
          )}`}
        >
          {overall_status || "NORMAL"}
        </span>
      </div>

      {/* 3 Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        {/* Heart Rate */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-3.5 flex flex-col justify-between">
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-medium text-slate-500 uppercase">Heart Rate</span>
            <span className="text-sm">💓</span>
          </div>
          <div className="my-1">
            <span className="text-2xl font-bold font-mono text-slate-900">{hrValue ?? "--"}</span>
            <span className="text-xs text-slate-500 ml-1">BPM</span>
          </div>
          <div>
            <span
              className={`inline-flex items-center text-[11px] font-semibold px-2 py-0.5 rounded-full border ${
                getStatusBadge(hr_status).bg
              }`}
            >
              {hr_status || "Normal"}
            </span>
          </div>
        </div>

        {/* SpO2 */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-3.5 flex flex-col justify-between">
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-medium text-slate-500 uppercase">SpO2 Oxygen</span>
            <span className="text-sm">🫁</span>
          </div>
          <div className="my-1">
            <span className="text-2xl font-bold font-mono text-slate-900">{spo2Value ?? "--"}</span>
            <span className="text-xs text-slate-500 ml-1">%</span>
          </div>
          <div>
            <span
              className={`inline-flex items-center text-[11px] font-semibold px-2 py-0.5 rounded-full border ${
                getStatusBadge(spo2_status).bg
              }`}
            >
              {spo2_status || "Normal"}
            </span>
          </div>
        </div>

        {/* Temperature */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-3.5 flex flex-col justify-between">
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-medium text-slate-500 uppercase">Core Temp</span>
            <span className="text-sm">🌡️</span>
          </div>
          <div className="my-1">
            <span className="text-2xl font-bold font-mono text-slate-900">{tempValue ?? "--"}</span>
            <span className="text-xs text-slate-500 ml-1">°C</span>
          </div>
          <div>
            <span
              className={`inline-flex items-center text-[11px] font-semibold px-2 py-0.5 rounded-full border ${
                getStatusBadge(temp_status).bg
              }`}
            >
              {temp_status || "Normal"}
            </span>
          </div>
        </div>
      </div>

      {/* Clinical Summary Narrative */}
      {summary && (
        <div className="bg-slate-50 rounded-lg p-3 text-xs text-slate-700 border border-slate-100">
          <span className="font-semibold text-slate-900">Triage Summary: </span>
          {summary}
        </div>
      )}
    </div>
  );
}
