import React from "react";
import { api } from "../api";

export default function ScreeningSummary({ screeningResult }) {
  if (!screeningResult) return null;

  const {
    status,
    screening_id,
    duration_ms,
    session_id,
    overall_risk_level,
    eye_screening = {},
    vitals_screening = {},
    disclaimer,
  } = screeningResult;

  const getRiskBannerStyle = (risk) => {
    switch (risk) {
      case "HIGH":
        return {
          bg: "bg-rose-50 border-rose-300 text-rose-900",
          badge: "bg-rose-600 text-white",
          label: "HIGH CLINICAL RISK",
          description: "One or more modalities indicated high-risk findings requiring medical attention.",
        };
      case "MODERATE":
        return {
          bg: "bg-amber-50 border-amber-300 text-amber-900",
          badge: "bg-amber-600 text-white",
          label: "MODERATE CLINICAL RISK",
          description: "Elevated or borderline parameters detected. Clinical monitoring recommended.",
        };
      case "LOW":
        return {
          bg: "bg-emerald-50 border-emerald-300 text-emerald-900",
          badge: "bg-emerald-600 text-white",
          label: "LOW CLINICAL RISK",
          description: "All evaluated screening streams returned normal or low-risk parameters.",
        };
      default:
        return {
          bg: "bg-slate-50 border-slate-300 text-slate-800",
          badge: "bg-slate-600 text-white",
          label: "INCOMPLETE / UNKNOWN",
          description: "Screening assessment incomplete or inconclusive.",
        };
    }
  };

  const banner = getRiskBannerStyle(overall_risk_level);

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
      {/* Overall Risk Header */}
      <div className={`p-4 rounded-xl border ${banner.bg}`}>
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mb-2">
          <div className="flex items-center space-x-2.5">
            <span className={`px-3 py-1 rounded-full text-xs font-bold tracking-wide ${banner.badge}`}>
              {banner.label}
            </span>
            <span className="text-xs font-medium opacity-80">
              Pipeline Status: <span className="font-bold uppercase">{status}</span>
            </span>
          </div>
          {session_id && (
            <a
              href={api.getSessionReportUrl(session_id)}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center space-x-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white shadow-sm transition-colors"
            >
              <span>📄 View / Print Clinical Report</span>
            </a>
          )}
        </div>
        <p className="text-xs opacity-90">{banner.description}</p>
      </div>

      {/* Synthesis Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
        <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
          <span className="text-slate-500 font-semibold block uppercase mb-1">
            Visual Stream (Eye Model)
          </span>
          <div className="flex justify-between items-center">
            <span className="text-slate-800 font-medium">
              {eye_screening.prediction
                ? eye_screening.prediction.replace(/_/g, " ").toUpperCase()
                : eye_screening.status === "rejected"
                ? "Quality Gate Rejected"
                : eye_screening.status === "skipped"
                ? "Skipped (No image)"
                : "Not evaluated"}
            </span>
            <span className="font-semibold text-slate-600">
              {eye_screening.confidence
                ? `${Math.round(eye_screening.confidence * 100)}% Conf.`
                : "--"}
            </span>
          </div>
        </div>

        <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
          <span className="text-slate-500 font-semibold block uppercase mb-1">
            Physiological Stream (Vitals)
          </span>
          <div className="flex justify-between items-center">
            <span className="text-slate-800 font-medium">
              {vitals_screening.overall_status ||
                (vitals_screening.status === "skipped" ? "Skipped (No vitals)" : "Not evaluated")}
            </span>
            <span className="font-semibold text-slate-600">
              {vitals_screening.risk_level ? `${vitals_screening.risk_level} Risk` : "--"}
            </span>
          </div>
        </div>
      </div>

      {/* Latency & Correlation Metadata */}
      <div className="flex flex-wrap justify-between items-center text-[11px] text-slate-400 pt-2 border-t border-slate-100">
        <span>Request ID: <span className="font-mono text-slate-500">{screening_id || "N/A"}</span></span>
        <span>Latency: <span className="font-mono font-medium text-slate-600">{duration_ms ?? "--"} ms</span></span>
      </div>

      {/* Clinical Disclaimer */}
      <div className="p-3 bg-amber-50/70 border border-amber-200/80 rounded-lg text-amber-900 text-xs">
        <div className="flex items-start space-x-2">
          <span className="text-amber-600 font-bold text-sm mt-0.5">⚠️</span>
          <div>
            <span className="font-semibold block mb-0.5">Clinical Decision Support Disclaimer</span>
            <p className="text-[11px] leading-relaxed opacity-90">
              {disclaimer ||
                "This system is strictly intended as a preliminary screening and decision support tool, not a diagnostic medical device. Outputs must never supersede clinical judgment or evaluation by certified medical practitioners."}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
