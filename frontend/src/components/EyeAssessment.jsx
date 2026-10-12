import React, { useState } from "react";

export default function EyeAssessment({ eyeData, previewUrl }) {
  const [showGradCam, setShowGradCam] = useState(true);

  if (!eyeData || eyeData.status !== "success") {
    return null;
  }

  const {
    prediction,
    confidence,
    risk_level,
    all_probabilities = {},
    gradcam_image,
  } = eyeData;

  const displayPrediction = (prediction || "Unknown").replace(/_/g, " ").toUpperCase();
  const confidencePct = Math.round((confidence || 0) * 1000) / 10;

  const getRiskBadge = (level) => {
    switch (level) {
      case "HIGH":
        return {
          bg: "bg-rose-100 text-rose-800 border-rose-300",
          text: "HIGH RISK",
        };
      case "MODERATE":
        return {
          bg: "bg-amber-100 text-amber-800 border-amber-300",
          text: "MODERATE RISK",
        };
      case "LOW":
        return {
          bg: "bg-emerald-100 text-emerald-800 border-emerald-300",
          text: "LOW RISK",
        };
      default:
        return {
          bg: "bg-slate-100 text-slate-800 border-slate-300",
          text: level || "UNKNOWN",
        };
    }
  };

  const riskBadge = getRiskBadge(risk_level);

  // Grad-CAM image src resolution
  const gradcamSrc = gradcam_image
    ? gradcam_image.startsWith("data:")
      ? gradcam_image
      : `data:image/png;base64,${gradcam_image}`
    : eyeData.gradcam_url || null;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
      <div className="flex justify-between items-start mb-4">
        <div>
          <span className="text-xs font-semibold text-blue-600 uppercase tracking-wider block">
            DenseNet201 Deep Learning Analysis
          </span>
          <h3 className="text-xl font-bold text-slate-900 mt-0.5">
            {displayPrediction}
          </h3>
        </div>
        <span
          className={`px-3 py-1 rounded-full text-xs font-bold border tracking-wide ${riskBadge.bg}`}
        >
          {riskBadge.text}
        </span>
      </div>

      {/* Confidence Bar */}
      <div className="mb-5 bg-slate-50 p-3.5 rounded-lg border border-slate-100">
        <div className="flex justify-between items-center text-xs font-medium text-slate-700 mb-1.5">
          <span>Model Diagnostic Confidence</span>
          <span className="font-bold text-sm text-slate-900">{confidencePct}%</span>
        </div>
        <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
          <div
            className={`h-2.5 rounded-full transition-all duration-500 ${
              risk_level === "HIGH"
                ? "bg-rose-500"
                : risk_level === "MODERATE"
                ? "bg-amber-500"
                : "bg-emerald-500"
            }`}
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>

      {/* Visual Saliency / Heatmap Overlay Toggle */}
      <div className="mb-5">
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
            Visual Explainability (Grad-CAM)
          </span>
          {gradcamSrc && previewUrl && (
            <div className="flex bg-slate-100 p-0.5 rounded-lg border border-slate-200 text-xs">
              <button
                onClick={() => setShowGradCam(false)}
                className={`px-2.5 py-1 rounded-md transition-colors font-medium ${
                  !showGradCam
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-500 hover:text-slate-900"
                }`}
              >
                Original Fundus
              </button>
              <button
                onClick={() => setShowGradCam(true)}
                className={`px-2.5 py-1 rounded-md transition-colors font-medium ${
                  showGradCam
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-500 hover:text-slate-900"
                }`}
              >
                Grad-CAM Heatmap
              </button>
            </div>
          )}
        </div>

        <div className="relative rounded-lg overflow-hidden bg-slate-900 border border-slate-200 flex items-center justify-center min-h-[220px]">
          {showGradCam && gradcamSrc ? (
            <img
              src={gradcamSrc}
              alt="Grad-CAM Saliency Map"
              className="object-contain max-h-72 w-full"
            />
          ) : previewUrl ? (
            <img
              src={previewUrl}
              alt="Original Fundus Photograph"
              className="object-contain max-h-72 w-full"
            />
          ) : (
            <span className="text-slate-400 text-xs">No visual preview available</span>
          )}

          {showGradCam && gradcamSrc && (
            <div className="absolute bottom-2 left-2 bg-slate-900/85 backdrop-blur-sm text-white text-[11px] px-2.5 py-1 rounded-md flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              <span>Warmer colors highlight pathological retinal cues driving inference</span>
            </div>
          )}
        </div>
      </div>

      {/* 4-Class Probability Breakdown */}
      <div>
        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2.5">
          Four-Class Probability Distribution
        </h4>
        <div className="space-y-2">
          {Object.entries(all_probabilities).map(([cls, prob]) => {
            const pct = Math.round(prob * 1000) / 10;
            const isTop = (prediction || "").toLowerCase() === cls.toLowerCase();
            return (
              <div key={cls} className="text-xs">
                <div className="flex justify-between items-center text-slate-600 mb-1">
                  <span className={`capitalize ${isTop ? "font-bold text-slate-900" : ""}`}>
                    {cls.replace(/_/g, " ")}
                  </span>
                  <span className={`font-mono ${isTop ? "font-bold text-blue-600" : ""}`}>
                    {pct}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                  <div
                    className={`h-1.5 rounded-full ${
                      isTop ? "bg-blue-600" : "bg-slate-300"
                    }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
