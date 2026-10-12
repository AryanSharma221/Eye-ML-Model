import React, { useState } from 'react';
import { IconUser, IconFileText, IconCheck } from '../components/MedicalIcons';

export default function PatientsPage({
  patients = [],
  selectedPatient,
  onSelectPatient,
  onOpenNewPatient,
  sessionsCountByPatient = {},
}) {
  const [searchTerm, setSearchTerm] = useState('');

  const filteredPatients = patients.filter(
    (p) =>
      p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <div className="flex flex-wrap items-center justify-between pb-4 mb-6 border-b border-slate-200 gap-3">
        <div>
          <h1 className="text-lg font-bold text-slate-900 uppercase tracking-wide">
            Patient Registry & Longitudinal Directory
          </h1>
          <p className="text-xs text-slate-500">
            Registered screening cohorts and longitudinal session links
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by name or ID..."
            className="px-3 py-1.5 text-xs bg-white border border-slate-300 rounded text-slate-900 focus:outline-none focus:border-sky-600 w-56"
          />
          <button
            type="button"
            onClick={onOpenNewPatient}
            className="px-3 py-1.5 text-xs font-semibold rounded bg-sky-700 hover:bg-sky-800 text-white transition-colors"
          >
            + Register Patient
          </button>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded overflow-hidden">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="bg-slate-100 border-b border-slate-200 text-slate-700 font-semibold">
              <th className="p-3">Patient Name</th>
              <th className="p-3">Record ID</th>
              <th className="p-3">Age / Sex</th>
              <th className="p-3">Contact Details</th>
              <th className="p-3">Encounter Count</th>
              <th className="p-3">Current Status</th>
              <th className="p-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {filteredPatients.length === 0 ? (
              <tr>
                <td colSpan="7" className="p-6 text-center text-slate-400">
                  No matching patients found.
                </td>
              </tr>
            ) : (
              filteredPatients.map((patient) => {
                const isSelected = selectedPatient?.id === patient.id;
                return (
                  <tr
                    key={patient.id}
                    className={`transition-colors ${
                      isSelected ? 'bg-sky-50/60' : 'hover:bg-slate-50'
                    }`}
                  >
                    <td className="p-3 font-semibold text-slate-900 flex items-center">
                      <div className="w-7 h-7 rounded bg-slate-200 text-slate-700 flex items-center justify-center mr-2.5 font-bold text-[11px]">
                        {patient.name.charAt(0)}
                      </div>
                      {patient.name}
                    </td>
                    <td className="p-3 font-mono text-slate-600">{patient.id}</td>
                    <td className="p-3 text-slate-700">
                      {patient.age}y / {patient.gender}
                    </td>
                    <td className="p-3 text-slate-600">{patient.contact || 'None'}</td>
                    <td className="p-3">
                      <span className="font-mono font-semibold px-2 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-700 text-[11px]">
                        {sessionsCountByPatient[patient.id] || 1} encounters
                      </span>
                    </td>
                    <td className="p-3">
                      {isSelected ? (
                        <span className="inline-flex items-center text-[11px] font-semibold text-sky-800 bg-sky-100 px-2 py-0.5 rounded border border-sky-300">
                          <IconCheck className="w-3 h-3 mr-1" />
                          Active Session
                        </span>
                      ) : (
                        <span className="text-slate-400 text-[11px]">Standby</span>
                      )}
                    </td>
                    <td className="p-3 text-right">
                      <button
                        type="button"
                        onClick={() => onSelectPatient(patient.id)}
                        className={`px-2.5 py-1 text-[11px] font-semibold rounded border transition-colors ${
                          isSelected
                            ? 'bg-sky-700 text-white border-sky-700'
                            : 'bg-white hover:bg-slate-100 text-slate-700 border-slate-300'
                        }`}
                      >
                        {isSelected ? 'Active' : 'Select Patient'}
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
