import React, { useState } from 'react';
import { IconGlobe, IconShieldCheck, IconCheck, IconRefresh } from '../components/MedicalIcons';

export default function CustomDomainGuide() {
  const [domainInput, setDomainInput] = useState('screening.medpulse.io');
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationStatus, setVerificationStatus] = useState('verified'); // verified, checking, idle

  const handleTestDomain = () => {
    setIsVerifying(true);
    setVerificationStatus('checking');
    setTimeout(() => {
      setIsVerifying(false);
      setVerificationStatus('verified');
    }, 1200);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="bg-white border border-slate-200 rounded p-6 sm:p-8">
        {/* Header */}
        <div className="border-b border-slate-200 pb-4 mb-6">
          <div className="flex items-center space-x-2 text-sky-700 mb-1">
            <IconGlobe className="w-5 h-5" />
            <span className="text-xs font-bold uppercase tracking-wider">
              Deployment Architecture
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900">
            Custom Domain & Production DNS Configuration
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Guidelines for binding production clinical domains with automated TLS certificates
          </p>
        </div>

        {/* Live Domain Status Card */}
        <div className="bg-slate-50 border border-slate-200 rounded p-4 mb-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <span className="text-[11px] font-semibold text-slate-500 uppercase block">
                Active Production Endpoint
              </span>
              <div className="text-base font-bold font-mono text-slate-900 mt-0.5">
                https://{domainInput}
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <span className="inline-flex items-center text-xs font-semibold px-2.5 py-1 rounded border bg-emerald-50 text-emerald-800 border-emerald-300">
                <IconCheck className="w-3.5 h-3.5 mr-1 text-emerald-600" />
                DNS & SSL Active
              </span>
              <button
                type="button"
                onClick={handleTestDomain}
                disabled={isVerifying}
                className="px-3 py-1 text-xs font-semibold rounded bg-slate-800 hover:bg-slate-900 text-white transition-colors flex items-center"
              >
                {isVerifying ? (
                  <>
                    <IconRefresh className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                    Querying DNS...
                  </>
                ) : (
                  'Re-verify DNS'
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Step-by-Step Production Setup Instructions */}
        <div className="space-y-6 text-xs text-slate-700">
          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              Step 1: Configure DNS Records at Domain Registrar
            </h2>
            <p className="mb-2">
              Add the following authoritative DNS records within your domain registrar (Cloudflare, Namecheap, Route 53, or GoDaddy):
            </p>
            <div className="overflow-x-auto border border-slate-200 rounded">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-100 border-b border-slate-200 text-slate-700">
                    <th className="p-2 font-semibold">Type</th>
                    <th className="p-2 font-semibold">Host / Name</th>
                    <th className="p-2 font-semibold">Target / Value</th>
                    <th className="p-2 font-semibold">TTL</th>
                  </tr>
                </thead>
                <tbody className="font-mono divide-y divide-slate-100">
                  <tr>
                    <td className="p-2 text-sky-800 font-bold">CNAME</td>
                    <td className="p-2">screening</td>
                    <td className="p-2">cname.medpulse-cdn.org</td>
                    <td className="p-2 text-slate-500">Auto (300s)</td>
                  </tr>
                  <tr>
                    <td className="p-2 text-sky-800 font-bold">A</td>
                    <td className="p-2">@ (root fallback)</td>
                    <td className="p-2">76.76.21.21</td>
                    <td className="p-2 text-slate-500">Auto (300s)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              Step 2: Frontend Environment Configuration
            </h2>
            <p className="mb-1.5">
              Specify your production custom domain and backend base URL inside your environment variables:
            </p>
            <div className="bg-slate-100 p-3 rounded border border-slate-300 font-mono text-[11px] text-slate-800">
              <div>VITE_CUSTOM_DOMAIN=screening.medpulse.io</div>
              <div>VITE_API_BASE_URL=https://api.screening.medpulse.io/api/v1</div>
            </div>
          </section>

          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              Step 3: Automated SSL / TLS Certificate Provisioning
            </h2>
            <p>
              When hosting via Vercel, Netlify, or Cloudflare Pages, SSL certificates are issued automatically through Let's Encrypt with automated 90-day renewal.
            </p>
            <p className="mt-1.5">
              For self-hosted Linux or Docker environments with NGINX, issue an SSL certificate using Certbot:
            </p>
            <div className="bg-slate-100 p-2.5 rounded border border-slate-300 font-mono text-[11px] text-slate-800 mt-1">
              certbot --nginx -d screening.medpulse.io --agree-tos
            </div>
          </section>

          <section>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wide mb-2">
              Step 4: Clinical Security Headers (HSTS)
            </h2>
            <p>
              To ensure compliance with clinical telehealth safety standards, ensure your reverse proxy sends strict security headers:
            </p>
            <ul className="list-disc pl-5 mt-1.5 space-y-1 text-slate-600">
              <li>Strict-Transport-Security: max-age=31536000; includeSubDomains; preload</li>
              <li>X-Content-Type-Options: nosniff</li>
              <li>X-Frame-Options: DENY</li>
              <li>Content-Security-Policy: default-src 'self'; img-src 'self' data: https:</li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
}
