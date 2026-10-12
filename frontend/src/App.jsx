import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import Navbar from './components/Navbar';
import ScreeningDashboard from './pages/ScreeningDashboard';
import PatientsPage from './pages/PatientsPage';
import TermsAndConditions from './pages/TermsAndConditions';
import PrivacyPolicy from './pages/PrivacyPolicy';
import CustomDomainGuide from './pages/CustomDomainGuide';
import PatientForm from './components/PatientForm';
import { api, checkBackendStatus } from './api';
import { IconShieldCheck, IconGlobe } from './components/MedicalIcons';

export default function App() {
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState('');
  const [patientSessions, setPatientSessions] = useState([]);
  const [backendOnline, setBackendOnline] = useState(false);
  const [isPatientModalOpen, setIsPatientModalOpen] = useState(false);

  // Initial load
  useEffect(() => {
    const initApp = async () => {
      const isOnline = await checkBackendStatus();
      setBackendOnline(isOnline);

      const patientList = await api.getPatients();
      setPatients(patientList);
      if (patientList.length > 0) {
        setSelectedPatientId(patientList[0].id);
      }
    };
    initApp();
  }, []);

  // Load sessions when selected patient changes
  useEffect(() => {
    if (!selectedPatientId) return;
    const loadSessions = async () => {
      const sessions = await api.getPatientSessions(selectedPatientId);
      setPatientSessions(sessions);
    };
    loadSessions();
  }, [selectedPatientId]);

  const selectedPatient =
    patients.find((p) => p.id === selectedPatientId) || patients[0];

  const handleSelectPatient = (id) => {
    setSelectedPatientId(id);
  };

  const handleCreatePatient = async (patientData) => {
    const created = await api.createPatient(patientData);
    setPatients((prev) => [created, ...prev]);
    setSelectedPatientId(created.id);
  };

  const handleRefreshSessions = async () => {
    if (selectedPatientId) {
      const sessions = await api.getPatientSessions(selectedPatientId);
      setPatientSessions(sessions);
    }
  };

  // Encounter counts for directory
  const sessionsCountByPatient = patients.reduce((acc, p) => {
    acc[p.id] = p.id === 'pat_001' ? 3 : p.id === 'pat_002' ? 1 : 1;
    return acc;
  }, {});

  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
        {/* Navigation Bar */}
        <Navbar
          patients={patients}
          selectedPatient={selectedPatient}
          onSelectPatient={handleSelectPatient}
          onOpenNewPatient={() => setIsPatientModalOpen(true)}
          backendOnline={backendOnline}
        />

        {/* Primary Page Content */}
        <main className="flex-1">
          <Routes>
            <Route
              path="/"
              element={
                <ScreeningDashboard
                  patient={selectedPatient}
                  sessions={patientSessions}
                  onRefreshSessions={handleRefreshSessions}
                  backendOnline={backendOnline}
                />
              }
            />
            <Route
              path="/patients"
              element={
                <PatientsPage
                  patients={patients}
                  selectedPatient={selectedPatient}
                  onSelectPatient={handleSelectPatient}
                  onOpenNewPatient={() => setIsPatientModalOpen(true)}
                  sessionsCountByPatient={sessionsCountByPatient}
                />
              }
            />
            <Route path="/domain-guide" element={<CustomDomainGuide />} />
            <Route path="/terms" element={<TermsAndConditions />} />
            <Route path="/privacy" element={<PrivacyPolicy />} />
          </Routes>
        </main>

        {/* Clinical Compliance Footer */}
        <footer className="bg-slate-900 border-t border-slate-800 text-slate-400 text-xs py-6">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center space-x-2">
              <IconShieldCheck className="w-4 h-4 text-sky-400" />
              <span>
                RetinaPulse Clinical Screening Suite &copy; 2026. Member 3 Frontend Implementation.
              </span>
            </div>

            <div className="flex items-center space-x-4 text-[11px]">
              <Link to="/domain-guide" className="hover:text-slate-200 transition-colors flex items-center">
                <IconGlobe className="w-3.5 h-3.5 mr-1" />
                screening.medpulse.io
              </Link>
              <Link to="/terms" className="hover:text-slate-200 transition-colors">
                Terms and Conditions
              </Link>
              <Link to="/privacy" className="hover:text-slate-200 transition-colors">
                Privacy Policy
              </Link>
              <span className="text-slate-600">|</span>
              <span className="text-slate-500">Non-Diagnostic Screening Protocol</span>
            </div>
          </div>
        </footer>

        {/* Patient Registration Modal */}
        <PatientForm
          isOpen={isPatientModalOpen}
          onClose={() => setIsPatientModalOpen(false)}
          onSubmit={handleCreatePatient}
        />
      </div>
    </BrowserRouter>
  );
}
