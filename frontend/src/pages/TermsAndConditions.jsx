import { IconShieldCheck } from '../components/MedicalIcons';

export default function TermsAndConditions() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="bg-white border border-slate-200 rounded p-6 sm:p-8">
        {/* Header */}
        <div className="border-b border-slate-200 pb-4 mb-6">
          <div className="flex items-center space-x-2 text-sky-700 mb-1">
            <IconShieldCheck className="w-5 h-5" />
            <span className="text-xs font-bold uppercase tracking-wider">
              Legal & Clinical Framework
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900">
            Terms and Conditions of Clinical Use
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Last Updated: October 12, 2026 | Document Reference: RP-TC-2026-V1
          </p>
        </div>

        {/* Content Sections */}
        <div className="space-y-6 text-xs text-slate-700 leading-relaxed">
          {/* Section 1 */}
          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              1. Non-Diagnostic Medical Screening Classification
            </h2>
            <p>
              The RetinaPulse Multimodal Health Screening Platform is classified strictly as an assistive computerized triage and risk stratifying decision-support tool. It does not provide definitive medical diagnoses, does not prescribe therapeutic regimens, and does not replace the comprehensive clinical judgment of a licensed ophthalmologist, optometrist, or general practitioner.
            </p>
            <p className="mt-2">
              All classifications produced by the DenseNet-201 convolutional neural network and XGBoost physiological vital sign pipelines represent statistical probability distributions. Clinicians must independently confirm any indicated pathology through standard ophthalmic examinations (including indirect ophthalmoscopy, slit lamp biomicroscopy, and applanation tonometry).
            </p>
          </section>

          {/* Section 2 */}
          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              2. Operator Qualifications and Protocols
            </h2>
            <p>
              Use of this software is restricted to certified healthcare personnel, trained clinical triage technicians, and authorized medical researchers. Operators are responsible for verifying that:
            </p>
            <ul className="list-disc pl-5 mt-1.5 space-y-1 text-slate-600">
              <li>Patient consent has been obtained in accordance with relevant national health regulations prior to fundus capture and vitals telemetry acquisition.</li>
              <li>Fundus cameras and vitals sensors (MAX30102 pulse oximeter, MLX90614 infrared sensor) are operated within approved calibration tolerances.</li>
              <li>Images failing the automated Quality Gate (insufficient contrast, blur score under 100, or absence of clear retinal landmarks) are systematically re-acquired.</li>
            </ul>
          </section>

          {/* Section 3 */}
          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              3. Telemetry and Explainability Disclosures
            </h2>
            <p>
              Gradient-weighted Class Activation Mapping (Grad-CAM) visualizations highlight salient spatial regions within the final convolutional layer of the eye disease classifier. These visual heatmaps are intended solely to provide explainability for operator audit and must not be used as anatomical boundary delineations for surgical or therapeutic planning.
            </p>
          </section>

          {/* Section 4 */}
          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              4. Data Retention and Electronic Health Records
            </h2>
            <p>
              Session dossiers and longitudinal trend records stored within the platform database are intended for continuity of screening workflows. Operators must ensure export to validated institutional Electronic Health Record (EHR) systems where permanent medical archiving is mandated by local law.
            </p>
          </section>

          {/* Section 5 */}
          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              5. Limitation of Liability
            </h2>
            <p>
              Under no circumstances shall the development team, project contributors, or affiliated educational and clinical organizations be liable for any direct, indirect, incidental, or consequential damages resulting from clinical decisions, delayed referrals, or diagnostic discrepancies arising from software outputs.
            </p>
          </section>

          {/* Section 6 */}
          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              6. Governance and Inquiries
            </h2>
            <p>
              Questions regarding clinical validation protocols or system terms may be directed to the technical coordination committee at: compliance@medpulse.io.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
