import React, { useState, useRef } from 'react';
import { IconUpload, IconCheck, IconAlertCircle, IconEye, IconRefresh } from './MedicalIcons';
import { DEMO_SAMPLES } from '../api';

export default function ImageUpload({ onEvaluate, isAnalyzing, qualityResults }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedPreview, setSelectedPreview] = useState(DEMO_SAMPLES[0].originalImage);
  const [selectedFileName, setSelectedFileName] = useState('sample_glaucoma.jpg');
  const [selectedFileObj, setSelectedFileObj] = useState(null);
  const [isAiSample, setIsAiSample] = useState(false);
  const [aiDisclosure, setAiDisclosure] = useState(DEMO_SAMPLES[0].disclosure);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0]);
    }
  };

  const processSelectedFile = (file) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      setSelectedPreview(event.target.result);
      setSelectedFileName(file.name);
      setSelectedFileObj(file);
      setIsAiSample(false);
      setAiDisclosure('Operator-Supplied Clinical Capture');
    };
    reader.readAsDataURL(file);
  };

  const handleSelectDemo = async (sample) => {
    setSelectedPreview(sample.originalImage);
    const ext = sample.originalImage.split('.').pop() || 'jpg';
    const filename = `${sample.id}.${ext}`;
    setSelectedFileName(filename);
    setIsAiSample(false);
    setAiDisclosure(sample.disclosure || 'Clinical Benchmark Fundus Sample');
    try {
      const response = await fetch(sample.originalImage);
      const blob = await response.blob();
      const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
      setSelectedFileObj(file);
    } catch (e) {
      console.warn('Could not load sample blob:', e);
    }
  };

  const handleTriggerAnalysis = async () => {
    if (isAnalyzing) return;
    let file = selectedFileObj;
    if (!file && selectedPreview) {
      try {
        const response = await fetch(selectedPreview);
        const blob = await response.blob();
        const ext = (selectedFileName || 'sample.jpg').split('.').pop() || 'jpg';
        file = new File([blob], selectedFileName || 'sample.jpg', { type: blob.type || 'image/jpeg' });
        setSelectedFileObj(file);
      } catch (e) {
        console.warn('Could not convert selectedPreview to File:', e);
      }
    }
    onEvaluate(file || selectedPreview, selectedFileName);
  };

  return (
    <div className="bg-white border border-slate-200 rounded p-5">
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-100">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center">
            <IconEye className="w-4 h-4 mr-2 text-sky-600" />
            1. Fundus Image Ingestion
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Color retinal photography upload with automated quality gating
          </p>
        </div>
        <span className="text-[11px] font-mono px-2 py-0.5 bg-slate-100 text-slate-600 rounded border border-slate-200">
          Modality: Retinal Fundus
        </span>
      </div>

      {/* Upload Zone */}
      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded p-5 text-center cursor-pointer transition-colors ${
          dragActive
            ? 'border-sky-500 bg-sky-50/50'
            : 'border-slate-300 hover:border-slate-400 bg-slate-50/60'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="hidden"
          onChange={handleChange}
          aria-label="Upload retinal fundus scan"
        />

        <div className="flex flex-col items-center justify-center space-y-2">
          <div className="w-10 h-10 rounded bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-600">
            <IconUpload className="w-5 h-5 text-slate-600" />
          </div>
          <div className="text-xs text-slate-700">
            <span className="font-semibold text-sky-700">Click to upload fundus file</span> or drag and drop
          </div>
          <div className="text-[11px] text-slate-500">
            Supported formats: PNG, JPG, JPEG (Minimum resolution: 512x512)
          </div>
        </div>
      </div>

      {/* Selected Image Preview & Metadata */}
      {selectedPreview && (
        <div className="mt-4 p-3 bg-slate-50 border border-slate-200 rounded">
          <div className="flex items-start space-x-3">
            <div className="relative w-20 h-20 bg-slate-900 rounded overflow-hidden flex-shrink-0 border border-slate-300">
              <img
                src={selectedPreview}
                alt="Selected retinal fundus scan preview showing optic disc and retinal vascular pattern"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold text-slate-900 truncate">
                  {selectedFileName}
                </p>
                {isAiSample && (
                  <span className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded bg-sky-100 text-sky-800 border border-sky-200">
                    AI Test Sample
                  </span>
                )}
              </div>
              <p className="text-[11px] text-slate-500 mt-1">
                {aiDisclosure}
              </p>
              <div className="mt-2 flex items-center space-x-2">
                <button
                  type="button"
                  onClick={handleTriggerAnalysis}
                  disabled={isAnalyzing}
                  className={`px-3 py-1.5 text-xs font-semibold rounded text-white transition-colors flex items-center ${
                    isAnalyzing
                      ? 'bg-slate-400 cursor-not-allowed'
                      : 'bg-sky-700 hover:bg-sky-800'
                  }`}
                >
                  {isAnalyzing ? (
                    <>
                      <IconRefresh className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                      Evaluating Quality & Model...
                    </>
                  ) : (
                    'Analyze Fundus Image'
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Pre-configured Clinical Demo Presets for Real Testing */}
      <div className="mt-4 pt-3 border-t border-slate-100">
        <label className="text-[11px] font-semibold text-slate-700 block mb-2">
          Or Load Verified Clinical Evaluation Samples:
        </label>
        <div className="grid grid-cols-2 gap-2">
          {DEMO_SAMPLES.map((sample) => (
            <button
              key={sample.id}
              type="button"
              onClick={() => handleSelectDemo(sample)}
              className="text-left p-2 rounded border border-slate-200 bg-white hover:bg-slate-50 transition-colors text-xs"
            >
              <div className="font-semibold text-slate-800">{sample.label}</div>
              <div className="text-[10px] text-slate-500 truncate">Condition: {sample.condition}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Image Quality Gate Telemetry */}
      {qualityResults && (
        <div className="mt-4 pt-3 border-t border-slate-100">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-800">
              Automated Image Quality Gate Telemetry:
            </span>
            <span
              className={`text-[11px] font-semibold px-2 py-0.5 rounded border ${
                qualityResults.overall_pass
                  ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                  : 'bg-rose-50 text-rose-800 border-rose-300'
              }`}
            >
              {qualityResults.overall_pass ? 'PASSED QUALITY GATE' : 'QUALITY REJECTED'}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2 text-[11px]">
            <div className="p-2 bg-slate-50 border border-slate-200 rounded">
              <span className="text-slate-500 block">Blur Metric:</span>
              <span className="font-mono font-semibold text-slate-800">
                {qualityResults.blur_score?.toFixed(1)} / 100 min
              </span>
            </div>
            <div className="p-2 bg-slate-50 border border-slate-200 rounded">
              <span className="text-slate-500 block">Mean Brightness:</span>
              <span className="font-mono font-semibold text-slate-800">
                {qualityResults.brightness_mean?.toFixed(1)} / 255
              </span>
            </div>
            <div className="p-2 bg-slate-50 border border-slate-200 rounded">
              <span className="text-slate-500 block">Contrast STD:</span>
              <span className="font-mono font-semibold text-slate-800">
                {qualityResults.contrast_std?.toFixed(1)}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
