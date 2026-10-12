import React, { useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import {
  IconFileText,
  IconDownload,
  IconHeartPulse,
  IconEye,
  IconAlertCircle,
  IconCheck,
} from './MedicalIcons';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function PatientHistory({
  patient,
  sessions = [],
  onSelectSessionForReport,
}) {
  const [activeTab, setActiveTab] = useState('timeline'); // 'timeline' or 'trends'

  if (!patient) {
    return (
      <div className="bg-white border border-slate-200 rounded p-6 text-center text-xs text-slate-500">
        No patient selected for longitudinal tracking.
      </div>
    );
  }

  const getSessionDate = (s) => s.date || (s.timestamp ? new Date(s.timestamp).toLocaleDateString() : 'Undated');
  const getSessionTime = (s) => s.time || (s.timestamp ? new Date(s.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '');
  const getEyePred = (s) => s.eye_record?.prediction || s.eye_prediction || 'Screening Pending';
  const getEyeConfidence = (s) => s.eye_record?.confidence ?? s.confidence;
  const getRisk = (s) => s.overall_risk_level || s.eye_record?.risk_level || s.risk_level || 'LOW';
  const getVitals = (s) => {
    if (s.vitals_records && s.vitals_records.length > 0) {
      const v = s.vitals_records[0];
      return {
        hr: v.heart_rate_bpm,
        spo2: v.spo2_percent,
        temp: v.temperature_celsius,
        status: v.overall_status,
      };
    }
    return s.vitals || { hr: '--', spo2: '--', temp: '--', status: 'Pending' };
  };

  // Build trend chart data
  const sortedSessions = [...sessions].sort(
    (a, b) => new Date(a.timestamp || a.date || 0).getTime() - new Date(b.timestamp || b.date || 0).getTime()
  );

  const dates = sortedSessions.map(getSessionDate);
  const hrData = sortedSessions.map((s) => {
    const val = getVitals(s).hr;
    return typeof val === 'number' ? val : 75;
  });
  const spo2Data = sortedSessions.map((s) => {
    const val = getVitals(s).spo2;
    return typeof val === 'number' ? val : 98;
  });
  const confidenceData = sortedSessions.map((s) => {
    const c = getEyeConfidence(s);
    return c ? c * 100 : 90;
  });

  const trendChartData = {
    labels: dates.length > 0 ? dates : ['No Records'],
    datasets: [
      {
        label: 'Heart Rate (BPM)',
        data: hrData.length > 0 ? hrData : [75],
        borderColor: '#dc2626',
        backgroundColor: '#dc2626',
        tension: 0.2,
        yAxisID: 'y',
      },
      {
        label: 'SpO2 Oxygen (%)',
        data: spo2Data.length > 0 ? spo2Data : [98],
        borderColor: '#0284c7',
        backgroundColor: '#0284c7',
        tension: 0.2,
        yAxisID: 'y1',
      },
      {
        label: 'AI Diagnostic Confidence (%)',
        data: confidenceData.length > 0 ? confidenceData : [90],
        borderColor: '#059669',
        backgroundColor: '#059669',
        tension: 0.2,
        borderDash: [4, 4],
        yAxisID: 'y1',
      },
    ],
  };

  const trendChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          boxWidth: 12,
          font: { size: 11, family: 'Plus Jakarta Sans' },
          color: '#334155',
        },
      },
      tooltip: {
        backgroundColor: '#0f172a',
        titleFont: { size: 11 },
        bodyFont: { size: 11 },
        cornerRadius: 4,
      },
    },
    scales: {
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: { display: true, text: 'Heart Rate (BPM)', font: { size: 10 } },
        ticks: { font: { size: 10 } },
        grid: { color: '#f1f5f9' },
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        min: 60,
        max: 100,
        title: { display: true, text: 'Percentage (%)', font: { size: 10 } },
        ticks: { font: { size: 10 } },
        grid: { drawOnChartArea: false },
      },
      x: {
        ticks: { font: { size: 10 } },
        grid: { color: '#f8fafc' },
      },
    },
  };

  const getRiskBadge = (level) => {
    const l = (level || 'LOW').toUpperCase();
    if (l === 'HIGH') {
      return (
        <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-rose-50 text-rose-800 border border-rose-300">
          HIGH RISK
        </span>
      );
    }
    if (l === 'MODERATE') {
      return (
        <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-300">
          MODERATE RISK
        </span>
      );
    }
    return (
      <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-300">
        LOW RISK
      </span>
    );
  };

  return (
    <div className="bg-white border border-slate-200 rounded p-5">
      {/* Header and Toggle */}
      <div className="flex flex-wrap items-center justify-between pb-3 mb-4 border-b border-slate-100 gap-2">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center">
            <IconFileText className="w-4 h-4 mr-2 text-sky-600" />
            5. Longitudinal Patient Screening History
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Historical trajectory for {patient.name} ({patient.id})
          </p>
        </div>

        <div className="inline-flex rounded border border-slate-300 bg-slate-100 text-xs overflow-hidden">
          <button
            type="button"
            onClick={() => setActiveTab('timeline')}
            className={`px-3 py-1 font-medium transition-colors ${
              activeTab === 'timeline'
                ? 'bg-white text-slate-900 font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Chronological Sessions ({sessions.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('trends')}
            className={`px-3 py-1 font-medium transition-colors ${
              activeTab === 'trends'
                ? 'bg-white text-slate-900 font-bold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Longitudinal Charts
          </button>
        </div>
      </div>

      {/* Tab: Chronological Timeline */}
      {activeTab === 'timeline' && (
        <div className="space-y-3">
          {sessions.length === 0 ? (
            <div className="py-6 text-center text-xs text-slate-400">
              No previous screening encounters recorded for this patient.
            </div>
          ) : (
            sessions.map((sess, idx) => (
              <div
                key={sess.id || idx}
                className="p-3.5 bg-slate-50 border border-slate-200 rounded hover:border-slate-300 transition-colors"
              >
                <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-bold text-slate-900">
                      Session #{sessions.length - idx}
                    </span>
                    <span className="text-[11px] text-slate-500">
                      Recorded: {getSessionDate(sess)} at {getSessionTime(sess)}
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.2 bg-slate-200 text-slate-700 rounded">
                      ID: {sess.id}
                    </span>
                  </div>

                  <div className="flex items-center space-x-2">
                    {getRiskBadge(getRisk(sess))}
                    <button
                      type="button"
                      onClick={() => onSelectSessionForReport(sess)}
                      className="px-2.5 py-1 text-[11px] font-semibold rounded bg-white hover:bg-slate-100 text-slate-700 border border-slate-300 transition-colors flex items-center"
                    >
                      <IconDownload className="w-3 h-3 mr-1" />
                      View Clinical Report
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                  <div className="bg-white p-2 rounded border border-slate-200">
                    <span className="text-slate-400 text-[10px] block">Eye Assessment:</span>
                    <span className="font-semibold text-slate-800">
                      {(getEyePred(sess) || 'Screening Pending').replace(/_/g, ' ')}
                    </span>
                    {getEyeConfidence(sess) && (
                      <span className="text-[10px] text-slate-500 block font-mono">
                        {(getEyeConfidence(sess) * 100).toFixed(1)}% Confidence
                      </span>
                    )}
                  </div>

                  <div className="bg-white p-2 rounded border border-slate-200">
                    <span className="text-slate-400 text-[10px] block">Heart Rate:</span>
                    <span className="font-mono font-semibold text-slate-800">
                      {getVitals(sess).hr ?? '--'} BPM
                    </span>
                    <span className="text-[10px] text-slate-500 block">
                      Status: {getVitals(sess).status ?? 'Normal'}
                    </span>
                  </div>

                  <div className="bg-white p-2 rounded border border-slate-200">
                    <span className="text-slate-400 text-[10px] block">SpO2 Oxygen:</span>
                    <span className="font-mono font-semibold text-slate-800">
                      {getVitals(sess).spo2 ?? '--'}%
                    </span>
                    <span className="text-[10px] text-slate-500 block">Pulse Oximetry</span>
                  </div>

                  <div className="bg-white p-2 rounded border border-slate-200">
                    <span className="text-slate-400 text-[10px] block">Temperature:</span>
                    <span className="font-mono font-semibold text-slate-800">
                      {getVitals(sess).temp ?? '--'}°C
                    </span>
                    <span className="text-[10px] text-slate-500 block">Infrared Forehead</span>
                  </div>
                </div>

                {sess.notes && (
                  <div className="mt-2 text-[11px] text-slate-600 bg-white p-2 rounded border border-slate-200">
                    <strong className="text-slate-700">Clinical Observations:</strong> {sess.notes}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {/* Tab: Trend Charts */}
      {activeTab === 'trends' && (
        <div className="h-64 bg-slate-50 border border-slate-200 rounded p-3">
          <Line data={trendChartData} options={trendChartOptions} />
        </div>
      )}
    </div>
  );
}
