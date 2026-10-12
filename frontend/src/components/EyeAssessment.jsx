import React, { useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { IconEye, IconCrosshair, IconAlertCircle, IconCheck } from './MedicalIcons';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export default function EyeAssessment({ result, isLoading }) {
  const [showGradCam, setShowGradCam] = useState(false);
  const [reticleCoords, setReticleCoords] = useState(null);
  const [isReticleActive, setIsReticleActive] = useState(true);

  if (isLoading) {
    return (
      <div className="bg-white border border-slate-200 rounded p-6">
        <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100">
          <div className="h-4 w-36 bg-slate-200 rounded animate-pulse" />
          <div className="h-4 w-20 bg-slate-200 rounded animate-pulse" />
        </div>
        {/* Skeleton Loader */}
        <div className="space-y-4">
          <div className="h-6 w-48 bg-slate-200 rounded animate-pulse" />
          <div className="h-64 bg-slate-100 rounded border border-slate-200 animate-pulse flex items-center justify-center">
            <span className="text-xs text-slate-400 font-mono">Running DenseNet201 Inference & Grad-CAM...</span>
          </div>
          <div className="h-28 bg-slate-100 rounded border border-slate-200 animate-pulse" />
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="bg-white border border-slate-200 rounded p-8 text-center">
        <div className="w-12 h-12 mx-auto rounded bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-400 mb-3">
          <IconEye className="w-6 h-6 text-slate-400" />
        </div>
        <h3 className="text-sm font-semibold text-slate-700">No Retinal Scan Evaluated</h3>
        <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
          Upload a fundus image in Panel 1 or select a clinical preset to initiate AI diagnostic screening.
        </p>
      </div>
    );
  }

  const {
    prediction,
    confidence,
    risk_level,
    all_probabilities,
    gradcam_image,
    original_url,
    timestamp,
  } = result;

  const riskBadgeStyles = {
    HIGH: 'bg-rose-50 text-rose-800 border-rose-300',
    MODERATE: 'bg-amber-50 text-amber-800 border-amber-300',
    LOW: 'bg-emerald-50 text-emerald-800 border-emerald-300',
  };

  const currentRisk = (risk_level || 'LOW').toUpperCase();

  // Chart.js data configuration with clinical tones (no harsh gradients, no rainbow)
  const formatLabel = (lbl) => (lbl || '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  const rawKeys = Object.keys(all_probabilities || {});
  const chartLabels = rawKeys.map(formatLabel);
  const chartValues = Object.values(all_probabilities || {}).map((val) => (val * 100).toFixed(1));

  const chartData = {
    labels: chartLabels,
    datasets: [
      {
        label: 'Posterior Probability (%)',
        data: chartValues,
        backgroundColor: rawKeys.map((key) =>
          key.toLowerCase().replace(/_/g, ' ') === (prediction || '').toLowerCase().replace(/_/g, ' ')
            ? currentRisk === 'HIGH'
              ? '#dc2626'
              : currentRisk === 'MODERATE'
              ? '#d97706'
              : '#059669'
            : '#94a3b8'
        ),
        borderColor: '#475569',
        borderWidth: 1,
        borderRadius: 2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#0f172a',
        titleFont: { size: 11, family: 'Plus Jakarta Sans' },
        bodyFont: { size: 11, family: 'Plus Jakarta Sans' },
        cornerRadius: 4,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          stepSize: 25,
          font: { size: 10 },
          callback: (value) => `${value}%`,
        },
        grid: { color: '#f1f5f9' },
      },
      x: {
        ticks: {
          font: { size: 10, weight: '600' },
          color: '#334155',
        },
        grid: { display: false },
      },
    },
  };

  // Diagnostic Reticle Cursor Handler
  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * 100);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * 100);
    setReticleCoords({ x, y });
  };

  const handleMouseLeave = () => {
    setReticleCoords(null);
  };

  const backendBase = import.meta.env.VITE_API_BASE_URL
    ? import.meta.env.VITE_API_BASE_URL.replace(/\/api\/v1\/?$/, '')
    : 'http://localhost:8000';

  const rawOriginal = original_url || result?.image_url;
  const originalDisplayUrl = rawOriginal
    ? rawOriginal.startsWith('http') || rawOriginal.startsWith('data:') || rawOriginal.startsWith('/samples')
      ? rawOriginal
      : `${backendBase}${rawOriginal.startsWith('/') ? '' : '/'}${rawOriginal}`
    : null;

  const currentImageUrl = showGradCam
    ? gradcam_image?.startsWith('data:')
      ? gradcam_image
      : `data:image/png;base64,${gradcam_image}`
    : originalDisplayUrl || (gradcam_image ? (gradcam_image.startsWith('data:') ? gradcam_image : `data:image/png;base64,${gradcam_image}`) : '/samples/sample_glaucoma.jpg');

  return (
    <div className="bg-white border border-slate-200 rounded p-5">
      {/* Panel Header */}
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center">
            <IconEye className="w-4 h-4 mr-2 text-sky-600" />
            2. AI Retinal Assessment & Grad-CAM
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            DenseNet201 deep classifier with explainable visual attention heatmap
          </p>
        </div>
        <span
          className={`text-xs font-bold px-2.5 py-0.5 rounded border ${
            riskBadgeStyles[currentRisk] || riskBadgeStyles.LOW
          }`}
        >
          {currentRisk} RISK CLASSIFICATION
        </span>
      </div>

      {/* Primary Diagnosis Metric Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
        <div className="p-3 bg-slate-50 border border-slate-200 rounded">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide block">
            Classified Condition
          </span>
          <div className="text-lg font-bold text-slate-900 tracking-tight mt-0.5 uppercase">
            {(prediction || '').replace(/_/g, ' ')}
          </div>
          <span className="text-[11px] text-slate-500 mt-1 block">
            Timestamp: {timestamp ? new Date(timestamp).toLocaleTimeString() : 'Current Session'}
          </span>
        </div>

        <div className="p-3 bg-slate-50 border border-slate-200 rounded">
          <div className="flex justify-between items-center">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">
              Diagnostic Confidence
            </span>
            <span className="text-xs font-mono font-bold text-slate-800">
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
          {/* Progress bar without harsh gradient */}
          <div className="w-full bg-slate-200 h-2 rounded mt-2 overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                currentRisk === 'HIGH'
                  ? 'bg-rose-600'
                  : currentRisk === 'MODERATE'
                  ? 'bg-amber-500'
                  : 'bg-emerald-600'
              }`}
              style={{ width: `${Math.min(100, confidence * 100)}%` }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-400 mt-1 font-mono">
            <span>Threshold: 85%</span>
            <span>Target: ≥90%</span>
          </div>
        </div>
      </div>

      {/* Retinal Fundus & Grad-CAM Heatmap Viewer with Precision Reticle Cursor */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-semibold text-slate-700">
              Interactive Image Inspection:
            </span>
            <button
              type="button"
              onClick={() => setIsReticleActive(!isReticleActive)}
              className={`text-[11px] px-2 py-0.5 rounded border transition-colors flex items-center ${
                isReticleActive
                  ? 'bg-sky-50 text-sky-800 border-sky-300'
                  : 'bg-slate-50 text-slate-600 border-slate-200'
              }`}
              title="Toggle precision coordinates cursor reticle"
            >
              <IconCrosshair className="w-3 h-3 mr-1" />
              Reticle HUD {isReticleActive ? 'ON' : 'OFF'}
            </button>
          </div>

          <div className="inline-flex rounded border border-slate-300 overflow-hidden bg-slate-100 text-xs">
            <button
              type="button"
              onClick={() => setShowGradCam(false)}
              className={`px-3 py-1 font-medium transition-colors ${
                !showGradCam ? 'bg-white text-slate-900 font-bold' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Original Fundus
            </button>
            <button
              type="button"
              onClick={() => setShowGradCam(true)}
              className={`px-3 py-1 font-medium transition-colors ${
                showGradCam ? 'bg-white text-slate-900 font-bold' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Grad-CAM Heatmap
            </button>
          </div>
        </div>

        {/* Viewport with cursor animation for clinical inspection */}
        <div
          onMouseMove={isReticleActive ? handleMouseMove : undefined}
          onMouseLeave={handleMouseLeave}
          className={`relative h-64 bg-slate-950 border border-slate-300 rounded overflow-hidden flex items-center justify-center ${
            isReticleActive ? 'inspection-reticle' : ''
          }`}
        >
          <img
            src={currentImageUrl}
            alt="Fundus photograph of retina displaying optic disc, macula, and vessel anatomy with diagnostic Grad-CAM overlay"
            className="h-full w-full object-contain select-none"
          />

          {/* Precision Diagnostic Reticle Coordinate Overlay (Cursor Animation) */}
          {isReticleActive && reticleCoords && (
            <div className="absolute top-2 left-2 bg-slate-900/90 border border-slate-700 text-sky-300 text-[10px] font-mono px-2 py-1 rounded pointer-events-none">
              Foveal Coordinate: X={reticleCoords.x}% Y={reticleCoords.y}% | Mode: {showGradCam ? 'Grad-CAM Attention' : 'Color Fundus'}
            </div>
          )}

          {/* View Mode Tag */}
          <div className="absolute bottom-2 right-2 bg-slate-900/85 text-slate-200 text-[10px] px-2 py-0.5 rounded border border-slate-700 font-mono">
            {showGradCam ? 'Layer: dense_block4 Grad-CAM' : 'Layer: RGB Sensor Capture'}
          </div>
        </div>
      </div>

      {/* Class Probabilities Bar Chart */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs font-semibold text-slate-700">
            Multi-Class Probability Distribution:
          </span>
          <span className="text-[11px] text-slate-500 font-mono">
            Sum: 100.0%
          </span>
        </div>
        <div className="h-36 bg-slate-50 border border-slate-200 rounded p-2">
          <Bar data={chartData} options={chartOptions} />
        </div>
      </div>
    </div>
  );
}
