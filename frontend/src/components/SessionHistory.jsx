import React, { useEffect, useState } from "react";
import { api } from "../api";

export default function SessionHistory() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadSessions = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getSessions();
      setSessions(data || []);
    } catch (err) {
      setError(err.message || "Failed to load session history.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSessions();
  }, []);

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-lg font-bold text-slate-900">Longitudinal Screening Sessions</h2>
          <p className="text-xs text-slate-500">
            Historical multimodal intake records, AI predictions, and archived diagnostic reports.
          </p>
        </div>
        <button
          onClick={loadSessions}
          disabled={loading}
          className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700"
        >
          {loading ? "Refreshing..." : "↻ Refresh"}
        </button>
      </div>

      {error && (
        <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-lg mb-4">
          {error}
        </div>
      )}

      {loading && sessions.length === 0 ? (
        <div className="text-center py-12 text-slate-400 text-sm">
          Loading screening session records...
        </div>
      ) : sessions.length === 0 ? (
        <div className="text-center py-12 border-2 border-dashed border-slate-200 rounded-xl">
          <p className="text-sm font-medium text-slate-500 mb-1">No screening sessions recorded yet.</p>
          <p className="text-xs text-slate-400">Run a live screening to create persistent session records.</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-400 uppercase tracking-wider font-semibold">
                <th className="py-3 px-3">Session</th>
                <th className="py-3 px-3">Date (UTC)</th>
                <th className="py-3 px-3">Overall Risk</th>
                <th className="py-3 px-3">Eye Classification</th>
                <th className="py-3 px-3">Vitals Status</th>
                <th className="py-3 px-3 text-right">Clinical Report</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {sessions.map((s) => {
                const eye = s.eye_records && s.eye_records[0];
                const physio = s.physio_records && s.physio_records[0];

                return (
                  <tr key={s.id} className="hover:bg-slate-50/80 transition-colors">
                    <td className="py-3 px-3 font-mono font-semibold text-slate-800">
                      #{s.id}
                    </td>
                    <td className="py-3 px-3 text-slate-500">
                      {new Date(s.timestamp).toLocaleDateString()} {new Date(s.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </td>
                    <td className="py-3 px-3">
                      <span
                        className={`inline-block px-2 py-0.5 rounded-full font-bold text-[10px] ${
                          s.overall_risk_level === "HIGH"
                            ? "bg-rose-100 text-rose-700"
                            : s.overall_risk_level === "MODERATE"
                            ? "bg-amber-100 text-amber-700"
                            : s.overall_risk_level === "LOW"
                            ? "bg-emerald-100 text-emerald-700"
                            : "bg-slate-100 text-slate-700"
                        }`}
                      >
                        {s.overall_risk_level || "UNKNOWN"}
                      </span>
                    </td>
                    <td className="py-3 px-3">
                      {eye ? (
                        <div>
                          <span className="font-semibold text-slate-800 uppercase">
                            {eye.prediction.replace(/_/g, " ")}
                          </span>
                          <span className="text-slate-400 ml-1.5 font-mono">
                            ({Math.round((eye.confidence || 0) * 100)}%)
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-400">None</span>
                      )}
                    </td>
                    <td className="py-3 px-3">
                      {physio ? (
                        <span className="text-slate-700 font-medium">
                          {physio.overall_status || "Recorded"} ({physio.heart_rate_bpm} BPM, {physio.spo2_percent}%)
                        </span>
                      ) : (
                        <span className="text-slate-400">None</span>
                      )}
                    </td>
                    <td className="py-3 px-3 text-right">
                      <a
                        href={api.getSessionReportUrl(s.id)}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center text-xs font-semibold text-blue-600 hover:text-blue-800 bg-blue-50 px-2.5 py-1 rounded"
                      >
                        Report &rarr;
                      </a>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
