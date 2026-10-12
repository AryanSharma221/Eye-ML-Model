import React, { useRef, useState } from "react";

export default function ImageUpload({
  selectedFile,
  setSelectedFile,
  previewUrl,
  setPreviewUrl,
  qualityChecks,
  isRejected,
  rejectionReason,
  disabled,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    const validExtensions = ["image/jpeg", "image/jpg", "image/png"];
    if (!validExtensions.includes(file.type) && !/\.(jpe?g|png)$/i.test(file.name)) {
      alert("Invalid format: Please upload a JPG or PNG retinal image.");
      return;
    }
    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    if (previewUrl && previewUrl.startsWith("blob:")) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
      <div className="flex justify-between items-center mb-3">
        <div>
          <h2 className="font-semibold text-slate-900 text-base flex items-center space-x-2">
            <span>📷 Retinal Fundus Photograph</span>
          </h2>
          <p className="text-xs text-slate-500">
            JPG, JPEG, or PNG up to 30 MB. Evaluated for focus, illumination, and retinal geometry.
          </p>
        </div>
        {selectedFile && !disabled && (
          <button
            onClick={handleClear}
            className="text-xs text-rose-600 hover:text-rose-700 font-medium px-2 py-1 rounded bg-rose-50"
          >
            Clear Image
          </button>
        )}
      </div>

      {/* Dropzone / Preview Area */}
      {!previewUrl ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
            isDragging
              ? "border-blue-500 bg-blue-50/50"
              : "border-slate-300 hover:border-blue-400 bg-slate-50/50"
          } ${disabled ? "opacity-60 cursor-not-allowed" : ""}`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png"
            className="hidden"
            onChange={(e) => handleFile(e.target.files?.[0])}
            disabled={disabled}
          />
          <div className="w-12 h-12 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center mx-auto mb-3">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <p className="text-sm font-medium text-slate-700 mb-1">
            Drag & drop patient fundus image here, or{" "}
            <span className="text-blue-600 font-semibold underline">browse</span>
          </p>
          <p className="text-xs text-slate-400">
            Recommended: High-resolution macula- or optic-disc-centered fundus photo
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="relative rounded-lg overflow-hidden bg-slate-900 border border-slate-200 flex items-center justify-center max-h-72">
            <img
              src={previewUrl}
              alt="Uploaded retinal fundus preview"
              className="object-contain max-h-72 w-full"
            />
            <div className="absolute bottom-2 left-2 bg-slate-900/80 backdrop-blur-sm text-white text-xs px-2.5 py-1 rounded-md">
              {selectedFile ? `${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)` : "Loaded Sample"}
            </div>
          </div>
        </div>
      )}

      {/* Quality Gate Rejection Banner */}
      {isRejected && (
        <div className="mt-4 p-3.5 bg-rose-50 border border-rose-200 rounded-lg flex items-start space-x-3">
          <div className="text-rose-600 mt-0.5">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <div className="flex-1 text-xs">
            <h4 className="font-semibold text-rose-800 text-sm mb-0.5">Image Rejected by Quality Gate</h4>
            <p className="text-rose-700 mb-2">{rejectionReason || "The submitted photograph does not satisfy clinical quality requirements."}</p>
            <p className="text-rose-600 font-medium">Action: Please acquire a clearer, well-illuminated retinal photograph and retry.</p>
          </div>
        </div>
      )}

      {/* Quality Checks Breakdown Indicator */}
      {qualityChecks && (
        <div className="mt-4 pt-3 border-t border-slate-100">
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
            Quality Gate Metrics
          </h4>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <div className="bg-slate-50 p-2 rounded-lg border border-slate-200 text-center">
              <span className="text-[10px] text-slate-500 uppercase block">Blur (Laplacian)</span>
              <span className={`text-xs font-bold ${qualityChecks.blur_score >= 100 ? "text-emerald-600" : "text-rose-600"}`}>
                {qualityChecks.blur_score?.toFixed(1) ?? "N/A"}
              </span>
              <span className="text-[9px] text-slate-400 block">&ge; 100 req.</span>
            </div>

            <div className="bg-slate-50 p-2 rounded-lg border border-slate-200 text-center">
              <span className="text-[10px] text-slate-500 uppercase block">Brightness</span>
              <span className={`text-xs font-bold ${
                ((qualityChecks.brightness_mean ?? qualityChecks.brightness) >= 30 &&
                 (qualityChecks.brightness_mean ?? qualityChecks.brightness) <= 225)
                  ? "text-emerald-600"
                  : "text-rose-600"
              }`}>
                {(qualityChecks.brightness_mean ?? qualityChecks.brightness)?.toFixed(1) ?? "N/A"}
              </span>
              <span className="text-[9px] text-slate-400 block">[30, 225]</span>
            </div>

            <div className="bg-slate-50 p-2 rounded-lg border border-slate-200 text-center">
              <span className="text-[10px] text-slate-500 uppercase block">Contrast (StdDev)</span>
              <span className={`text-xs font-bold ${
                (qualityChecks.contrast_std ?? qualityChecks.contrast) >= 20 ? "text-emerald-600" : "text-rose-600"
              }`}>
                {(qualityChecks.contrast_std ?? qualityChecks.contrast)?.toFixed(1) ?? "N/A"}
              </span>
              <span className="text-[9px] text-slate-400 block">&ge; 20 req.</span>
            </div>

            <div className="bg-slate-50 p-2 rounded-lg border border-slate-200 text-center">
              <span className="text-[10px] text-slate-500 uppercase block">Retinal Region</span>
              <span className={`text-xs font-bold ${qualityChecks.retinal_region_detected ? "text-emerald-600" : "text-rose-600"}`}>
                {qualityChecks.retinal_region_detected ? "Detected" : "Missing"}
              </span>
              <span className="text-[9px] text-slate-400 block">Contour gate</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
