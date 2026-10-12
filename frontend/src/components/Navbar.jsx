import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { IconEye, IconUser, IconGlobe, IconShieldCheck } from './MedicalIcons';

export default function Navbar({
  patients = [],
  selectedPatient,
  onSelectPatient,
  onOpenNewPatient,
  backendOnline,
}) {
  const location = useLocation();

  const navLinks = [
    { to: '/', label: 'Screening Dashboard' },
    { to: '/patients', label: 'Patient Records' },
    { to: '/domain-guide', label: 'Domain Config' },
    { to: '/terms', label: 'Terms and Conditions' },
    { to: '/privacy', label: 'Privacy Policy' },
  ];

  return (
    <header className="bg-slate-900 border-b border-slate-800 text-slate-100">
      {/* Top Clinical Utility Bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2 flex flex-wrap items-center justify-between border-b border-slate-800/80 text-xs">
        <div className="flex items-center space-x-3">
          <span className="flex items-center text-slate-300 font-medium">
            <IconGlobe className="w-3.5 h-3.5 mr-1.5 text-sky-400" />
            Custom Domain: <strong className="text-white ml-1">screening.medpulse.io</strong>
          </span>
          <span className="inline-flex items-center px-1.5 py-0.5 rounded border border-emerald-500/40 bg-emerald-950/60 text-emerald-300 text-[11px]">
            DNS & SSL Verified
          </span>
        </div>

        <div className="flex items-center space-x-4">
          {/* Backend Connection Indicator */}
          <div className="flex items-center space-x-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                backendOnline ? 'bg-emerald-400' : 'bg-amber-400'
              }`}
            />
            <span className="text-slate-300">
              API Status: {backendOnline ? 'FastAPI Online' : 'Simulation Mode'}
            </span>
          </div>

          {/* Regulatory Clinical Disclaimer */}
          <div className="hidden md:flex items-center text-slate-400">
            <IconShieldCheck className="w-3.5 h-3.5 mr-1 text-slate-400" />
            <span>Clinical Screening Assessment (Investigational)</span>
          </div>
        </div>
      </div>

      {/* Main Navigation Bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex flex-wrap items-center justify-between gap-4">
        {/* Brand Identity */}
        <Link to="/" className="flex items-center space-x-3 group">
          <div className="w-9 h-9 rounded bg-sky-600 flex items-center justify-center text-white font-bold">
            <IconEye className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="text-base font-bold tracking-tight text-white flex items-center">
              RetinaPulse
              <span className="ml-2 text-[10px] tracking-wider uppercase px-1.5 py-0.5 rounded bg-slate-800 text-sky-300 border border-slate-700">
                v1.0 Clinical
              </span>
            </div>
            <div className="text-[11px] text-slate-400">
              Multimodal Retinal & Physiological Screening
            </div>
          </div>
        </Link>

        {/* Primary Nav Links */}
        <nav className="flex flex-wrap items-center gap-1 sm:gap-2">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.to;
            return (
              <Link
                key={link.to}
                to={link.to}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  isActive
                    ? 'bg-slate-800 text-sky-400 border border-slate-700'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        {/* Patient Selection Bar */}
        <div className="flex flex-wrap items-center gap-2 max-w-full">
          <div className="flex items-center bg-slate-800 border border-slate-700 rounded px-2.5 py-1 max-w-[200px] sm:max-w-none">
            <IconUser className="w-4 h-4 text-slate-400 mr-2 flex-shrink-0" />
            <select
              value={selectedPatient?.id || ''}
              onChange={(e) => onSelectPatient(e.target.value)}
              className="bg-transparent text-xs text-white focus:outline-none cursor-pointer pr-2 truncate max-w-[150px] sm:max-w-none"
              aria-label="Active patient selector"
            >
              {patients.map((p) => (
                <option key={p.id} value={p.id} className="bg-slate-900 text-white">
                  {p.name} ({p.gender ? p.gender[0] : 'U'}, {p.age || '--'}y)
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={onOpenNewPatient}
            className="px-2.5 py-1 text-xs font-semibold rounded bg-sky-600 hover:bg-sky-500 text-white transition-colors flex-shrink-0"
            title="Register a new patient"
          >
            + New Patient
          </button>
        </div>
      </div>
    </header>
  );
}
