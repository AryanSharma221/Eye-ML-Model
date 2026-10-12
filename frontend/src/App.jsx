import React, { useEffect, useState } from "react";
import { api } from "./api";
import Navbar from "./components/Navbar";
import ImageUpload from "./components/ImageUpload";
import ManualVitals from "./components/ManualVitals";
import EyeAssessment from "./components/EyeAssessment";
import VitalsPanel from "./components/VitalsPanel";
import ScreeningSummary from "./components/ScreeningSummary";
import PatientModal from "./components/PatientModal";
import SessionHistory from "./components/SessionHistory";

export default function App() {
  const [activeTab, setActiveTab] = useState("screening");
  const [health, setHealth] = useState(null);

  // Patient State
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [isPatientModalOpen, setIsPatientModalOpen] = useState(false);

  // Input Modalities State
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [heartRate, setHeartRate] = useState("75");
  const [spo2, setSpo2] = useState("98");
  const [temperature, setTemperature] = useState("36.8");

  // Execution & Results State
  const [isScreening, setIsScreening] = useState(false);
  const [screeningStatusText, setScreeningStatusText] = useState("");
  const [screeningResult, setScreeningResult] = useState(null);
  const [errorBanner, setErrorBanner] = useState(null);

  // Load backend health and patients on startup
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const h = await api.getHealth();
        setHealth(h);
      } catch (err) {
        console.warn("Backend health check warning:", err);
      }

      try {
        const pList = await api.getPatients();
        setPatients(pList || []);
      } catch (err) {
        console.warn("Patient fetch warning:", err);
      }
    };
    fetchInitialData();
  }, []);

  const selectedPatient = patients.find((p) => p.id === selectedPatientId) || null;

  // Execute Multimodal Screening
  const handleRunScreening = async () => {
    if (!selectedFile && !heartRate && !spo2 && !temperature) {
      setErrorBanner("Please select a retinal fundus image and/or enter vital signs to run screening.");
      return;
    }

    setIsScreening(true);
    setErrorBanner(null);
    setScreeningStatusText("Initiating multimodal evaluation...");

    try {
      setScreeningStatusText("Evaluating image quality & vitals...");
      const result = await api.screenMultimodal({
        imageFile: selectedFile,
        heartRate: heartRate !== "" ? heartRate : null,
        spo2: spo2 !== "" ? spo2 : null,
        temperature: temperature !== "" ? temperature : null,
        patientId: selectedPatientId,
        includeGradcam: true,
      });

      setScreeningResult(result);
    } catch (err) {
      console.error("Screening failure:", err);
      if (err.data && (err.data.status === "rejected" || err.data.eye_screening?.status === "rejected")) {
        // Quality Gate Rejected response payload
        setScreeningResult(err.data);
      } else {
        setErrorBanner(err.message || "An unexpected error occurred during screening.");
      }
    } finally {
      setIsScreening(false);
      setScreeningStatusText("");
    }
  };

  const handlePatientCreated = (newPatient) => {
    setPatients((prev) => [newPatient, ...prev]);
    setSelectedPatientId(newPatient.id);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      <Navbar health={health} activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* TAB 1: LIVE SCREENING */}
        {activeTab === "screening" && (
          <div className="space-y-6">
            {/* Patient Header Strip */}
            <div className="bg-white rounded-xl border border-slate-200 px-5 py-3.5 shadow-sm flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center text-slate-600 font-bold text-sm">
                  👤
                </div>
                <div>
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                    Active Patient
                  </span>
                  <div className="text-sm font-bold text-slate-900">
                    {selectedPatient ? (
                      <span>
                        {selectedPatient.name} &bull;{" "}
                        <span className="font-normal text-slate-500">
                          {selectedPatient.age ? `${selectedPatient.age} yrs` : "Age N/A"},{" "}
                          {selectedPatient.gender || "Gender N/A"} (ID #{selectedPatient.id})
                        </span>
                      </span>
                    ) : (
                      <span className="text-slate-500 font-normal italic">
                        Anonymous / Walk-in Session
                      </span>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                {selectedPatient && (
                  <button
                    onClick={() => setSelectedPatientId(null)}
                    className="text-xs text-slate-500 hover:text-slate-700 px-2 py-1"
                  >
                    Deselect
                  </button>
                )}
                <button
                  onClick={() => setIsPatientModalOpen(true)}
                  className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700 transition-colors"
                >
                  {selectedPatient ? "Change Patient" : "+ Select / Register Patient"}
                </button>
              </div>
            </div>

            {/* Error Notification Banner */}
            {errorBanner && (
              <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 text-xs flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="text-rose-600 font-bold">⚠️</span>
                  <span>{errorBanner}</span>
                </div>
                <button
                  onClick={() => setErrorBanner(null)}
                  className="font-bold text-rose-500 hover:text-rose-700 px-2"
                >
                  &times;
                </button>
              </div>
            )}

            {/* Input Modalities Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left Column: Retinal Fundus Input */}
              <ImageUpload
                selectedFile={selectedFile}
                setSelectedFile={setSelectedFile}
                previewUrl={previewUrl}
                setPreviewUrl={setPreviewUrl}
                qualityChecks={screeningResult?.eye_screening?.quality_checks}
                isRejected={screeningResult?.eye_screening?.status === "rejected"}
                rejectionReason={screeningResult?.eye_screening?.reason}
                disabled={isScreening}
              />

              {/* Right Column: Physiological Input */}
              <ManualVitals
                heartRate={heartRate}
                setHeartRate={setHeartRate}
                spo2={spo2}
                setSpo2={setSpo2}
                temperature={temperature}
                setTemperature={setTemperature}
                disabled={isScreening}
              />
            </div>

            {/* Action Bar */}
            <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="text-xs text-slate-500">
                <span className="font-semibold text-slate-700">Modality Independence: </span>
                Image and vitals are evaluated separately. Either stream can be processed individually.
              </div>

              <button
                onClick={handleRunScreening}
                disabled={isScreening}
                className={`w-full sm:w-auto px-6 py-2.5 rounded-xl font-semibold text-sm text-white shadow-md transition-all flex items-center justify-center space-x-2 ${
                  isScreening
                    ? "bg-blue-400 cursor-not-allowed"
                    : "bg-blue-600 hover:bg-blue-700 shadow-blue-500/20 active:scale-[0.99]"
                }`}
              >
                {isScreening ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>{screeningStatusText || "Processing Screening..."}</span>
                  </>
                ) : (
                  <>
                    <span>⚡ Run Multimodal Screening</span>
                  </>
                )}
              </button>
            </div>

            {/* Results Section */}
            {screeningResult && (
              <div className="space-y-6 pt-2">
                {/* Overall Synthesis Banner */}
                <ScreeningSummary screeningResult={screeningResult} />

                {/* Modality Findings Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <EyeAssessment
                    eyeData={screeningResult.eye_screening}
                    previewUrl={previewUrl}
                  />
                  <VitalsPanel
                    vitalsData={screeningResult.vitals_screening}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: PATIENTS LIST */}
        {activeTab === "patients" && (
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-lg font-bold text-slate-900">Registered Patients</h2>
                <p className="text-xs text-slate-500">
                  Manage patient demographic identities and view longitudinal intake history.
                </p>
              </div>
              <button
                onClick={() => setIsPatientModalOpen(true)}
                className="text-xs font-semibold px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white shadow-sm"
              >
                + Register New Patient
              </button>
            </div>

            {patients.length === 0 ? (
              <div className="text-center py-12 border-2 border-dashed border-slate-200 rounded-xl">
                <p className="text-sm font-medium text-slate-500 mb-1">No patients registered yet.</p>
                <button
                  onClick={() => setIsPatientModalOpen(true)}
                  className="text-xs text-blue-600 font-semibold"
                >
                  Create your first patient record
                </button>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-400 uppercase tracking-wider font-semibold">
                      <th className="py-3 px-3">ID</th>
                      <th className="py-3 px-3">Full Name</th>
                      <th className="py-3 px-3">Age</th>
                      <th className="py-3 px-3">Gender</th>
                      <th className="py-3 px-3">Contact</th>
                      <th className="py-3 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {patients.map((p) => (
                      <tr key={p.id} className="hover:bg-slate-50/80 transition-colors">
                        <td className="py-3 px-3 font-mono font-bold text-slate-700">#{p.id}</td>
                        <td className="py-3 px-3 font-semibold text-slate-900">{p.name}</td>
                        <td className="py-3 px-3 text-slate-600">{p.age ? `${p.age} yrs` : "--"}</td>
                        <td className="py-3 px-3 text-slate-600">{p.gender || "--"}</td>
                        <td className="py-3 px-3 text-slate-600">{p.contact || "--"}</td>
                        <td className="py-3 px-3 text-right">
                          <button
                            onClick={() => {
                              setSelectedPatientId(p.id);
                              setActiveTab("screening");
                            }}
                            className="text-xs font-semibold text-blue-600 hover:text-blue-800 bg-blue-50 px-2.5 py-1 rounded"
                          >
                            Select for Screening
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: SESSION HISTORY */}
        {activeTab === "sessions" && <SessionHistory />}
      </main>

      {/* Patient Registration / Picker Modal */}
      <PatientModal
        isOpen={isPatientModalOpen}
        onClose={() => setIsPatientModalOpen(false)}
        onPatientCreated={handlePatientCreated}
        patients={patients}
        selectedPatientId={selectedPatientId}
        onSelectPatient={(id) => setSelectedPatientId(id)}
      />
    </div>
  );
}
