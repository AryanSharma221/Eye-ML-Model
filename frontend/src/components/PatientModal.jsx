import React, { useState } from "react";
import { api } from "../api";

export default function PatientModal({ isOpen, onClose, onPatientCreated, patients, selectedPatientId, onSelectPatient }) {
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("Unspecified");
  const [contact, setContact] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim()) {
      setError("Patient name is required.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const newPatient = await api.createPatient({ name, age, gender, contact });
      onPatientCreated(newPatient);
      setName("");
      setAge("");
      setGender("Unspecified");
      setContact("");
      onClose();
    } catch (err) {
      setError(err.message || "Failed to create patient.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/50 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-xl border border-slate-200">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-bold text-slate-900">Patient Management</h3>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 text-lg font-bold px-2 py-1"
          >
            &times;
          </button>
        </div>

        {/* Existing Patients Picker */}
        {patients && patients.length > 0 && (
          <div className="mb-6 pb-5 border-b border-slate-100">
            <label className="block text-xs font-semibold text-slate-700 uppercase mb-2">
              Select Existing Patient
            </label>
            <div className="max-h-36 overflow-y-auto space-y-1.5 pr-1">
              {patients.map((p) => (
                <div
                  key={p.id}
                  onClick={() => {
                    onSelectPatient(p.id);
                    onClose();
                  }}
                  className={`p-2.5 rounded-lg border text-xs cursor-pointer flex justify-between items-center transition-colors ${
                    selectedPatientId === p.id
                      ? "border-blue-500 bg-blue-50/60 font-semibold text-blue-900"
                      : "border-slate-200 hover:bg-slate-50 text-slate-700"
                  }`}
                >
                  <div>
                    <span className="font-bold">{p.name}</span>
                    <span className="text-slate-400 ml-2">
                      ({p.age ? `${p.age}y` : "Age N/A"}, {p.gender || "N/A"})
                    </span>
                  </div>
                  <span className="text-[11px] text-slate-400">ID #{p.id}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Register New Patient */}
        <form onSubmit={handleSubmit} className="space-y-3">
          <h4 className="text-xs font-semibold text-slate-700 uppercase">Register New Patient</h4>
          {error && <div className="text-xs text-rose-600 bg-rose-50 p-2 rounded">{error}</div>}

          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Full Name *</label>
            <input
              type="text"
              required
              placeholder="e.g. Eleanor Vance"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full text-xs px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Age</label>
              <input
                type="number"
                min="0"
                max="130"
                placeholder="e.g. 54"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                className="w-full text-xs px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Gender</label>
              <select
                value={gender}
                onChange={(e) => setGender(e.target.value)}
                className="w-full text-xs px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
              >
                <option value="Female">Female</option>
                <option value="Male">Male</option>
                <option value="Other">Other</option>
                <option value="Unspecified">Unspecified</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Contact / Email</label>
            <input
              type="text"
              placeholder="e.g. patient@example.com"
              value={contact}
              onChange={(e) => setContact(e.target.value)}
              className="w-full text-xs px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="pt-2 flex justify-end space-x-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-xs font-medium text-slate-600 hover:bg-slate-100 rounded-lg"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm disabled:opacity-60"
            >
              {submitting ? "Registering..." : "Register & Select"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
