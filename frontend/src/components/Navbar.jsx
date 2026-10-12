import React from "react";

export default function Navbar({ health, activeTab, setActiveTab }) {
  const isReady = health && health.model_loaded;
  const device = health ? health.device : "Unknown";

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Brand */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="font-bold text-lg text-slate-900 leading-tight">
                  Multimodal Health Screening
                </h1>
                <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-blue-100 text-blue-700">
                  v1.0
                </span>
              </div>
              <p className="text-xs text-slate-500">
                DenseNet201 Retinal AI &bull; Physiological Triage &bull; Decision Support
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex items-center space-x-1 sm:space-x-2">
            <button
              onClick={() => setActiveTab("screening")}
              className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
                activeTab === "screening"
                  ? "bg-blue-50 text-blue-700"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              Live Screening
            </button>
            <button
              onClick={() => setActiveTab("patients")}
              className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
                activeTab === "patients"
                  ? "bg-blue-50 text-blue-700"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              Patients
            </button>
            <button
              onClick={() => setActiveTab("sessions")}
              className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
                activeTab === "sessions"
                  ? "bg-blue-50 text-blue-700"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              Session History
            </button>
          </div>

          {/* System Status Pill */}
          <div className="hidden md:flex items-center space-x-2">
            <div
              className={`flex items-center space-x-1.5 text-xs font-medium px-2.5 py-1 rounded-full border ${
                isReady
                  ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                  : "bg-amber-50 text-amber-700 border-amber-200"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isReady ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
                }`}
              />
              <span>
                {isReady ? `AI Model Ready (${device})` : "Model Initializing..."}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
