import React from 'react';
import { IconShieldCheck, IconFileText } from '../components/MedicalIcons';

export default function PrivacyPolicy() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="bg-white border border-slate-200 rounded p-6 sm:p-8">
        {/* Header */}
        <div className="border-b border-slate-200 pb-4 mb-6">
          <div className="flex items-center space-x-2 text-sky-700 mb-1">
            <IconShieldCheck className="w-5 h-5" />
            <span className="text-xs font-bold uppercase tracking-wider">
              Health Information Protection
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900">
            Privacy Policy and Patient Data Security
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Effective Date: October 12, 2026 | Document Reference: RP-PRIV-2026-V1
          </p>
        </div>

        {/* Content Sections */}
        <div className="space-y-6 text-xs text-slate-700 leading-relaxed">
          {/* Section 1 */}
          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              1. Protected Health Information (PHI) Commitment
            </h2>
            <p>
              RetinaPulse is engineered around the principle of strict data minimization. We recognize that retinal fundus photography contains unique vascular biometric geometry and that physiological telemetry constitutes sensitive health information. This privacy policy defines how patient information is acquired, processed, stored, and protected during screening sessions.
            </p>
          </section>

          {/* Section 2 */}
          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              2. Categories of Information Processed
            </h2>
            <p>The platform processes the following explicit data categories:</p>
            <ul className="list-disc pl-5 mt-1.5 space-y-1 text-slate-600">
              <li><strong>Patient Demographics:</strong> Name, Age, Biological Sex, Contact or Medical Record Identifier (MRN).</li>
              <li><strong>Retinal Imaging Data:</strong> High-resolution digital fundus photography, automated quality gate diagnostic metrics, and synthesized Grad-CAM attention maps.</li>
              <li><strong>Physiological Telemetry:</strong> Heart Rate (BPM), Blood Oxygen Saturation (SpO2 percentage), and Infrared Body Temperature (°C) received via hardware sensors or manual clinician entry.</li>
              <li><strong>Screening Metadata:</strong> Session timestamps, operator clinical notes, and generated triage classifications.</li>
            </ul>
          </section>

          {/* Section 3 */}
          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              3. Data Security and Cryptographic Standards
            </h2>
            <p>
              All information handled by the system adheres to industry standard encryption and security mechanisms:
            </p>
            <ul className="list-disc pl-5 mt-1.5 space-y-1 text-slate-600">
              <li><strong>Data in Transit:</strong> Transport Layer Security (TLS 1.3) cryptographic protocols for all API exchanges between the React frontend, FastAPI backend, and ESP32 hardware clients.</li>
              <li><strong>Data at Rest:</strong> AES-256 bit encryption applied to persistent databases containing patient identifiers and uploaded image artifacts.</li>
              <li><strong>Biometric Anonymization:</strong> De-identification pipelines strip EXIF metadata from uploaded photographic files before persistence.</li>
            </ul>
          </section>

          {/* Section 4 */}
          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              4. Strict Non-Commercial Data Policy
            </h2>
            <p>
              Patient data collected through this screening interface is never sold, leased, rented, or distributed to third-party commercial entities, advertisers, or insurance underwriters. De-identified statistical telemetry is retained solely for model validation and clinical performance audits with explicit institutional approval.
            </p>
          </section>

          {/* Section 5 */}
          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              5. Patient Rights and Audit Logs
            </h2>
            <p>
              Under applicable healthcare regulations, patients retain the right to inspect their stored screening history, request corrections to demographic records, or request complete removal of their biometric records from the system database. Requests can be initiated directly by the operating clinician via the patient management portal.
            </p>
          </section>

          {/* Section 6 */}
          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              6. Privacy Officer Contact
            </h2>
            <p>
              For privacy verification inquiries or compliance audits, contact the Data Protection Officer at: dpo@medpulse.io.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
