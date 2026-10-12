/**
 * Centralized API client for the Multimodal Health Screening Frontend.
 * Interacts with the FastAPI backend across system health, patient intake,
 * eye disease screening, physiological vitals triage, and session reporting.
 */

// Use Vite environment variable with graceful fallback to relative /api/v1
const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL || "").replace(/\/+$/, "");
const API_BASE = `${BACKEND_URL}/api/v1`;

const DEFAULT_TIMEOUT_MS = 30000; // 30s timeout for deep learning inference & Grad-CAM

/**
 * Fetch wrapper with timeout and informative error sanitization.
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(id);
    return response;
  } catch (err) {
    clearTimeout(id);
    if (err.name === "AbortError") {
      const timeoutError = new Error("The screening request timed out after 30 seconds. Please try again.");
      timeoutError.status = 504;
      throw timeoutError;
    }
    // Network failure (server down, CORS blocked, unreachable)
    const netError = new Error(
      `Unable to reach the backend service at ${BACKEND_URL || "the configured server"}. Please ensure the server is running.`
    );
    netError.status = 0;
    netError.isNetworkError = true;
    throw netError;
  }
}

/**
 * Parses response JSON safely or constructs an informative sanitized error object.
 */
async function handleResponse(res) {
  const contentType = res.headers.get("content-type");
  let data = null;

  if (contentType && contentType.includes("application/json")) {
    try {
      data = await res.json();
    } catch {
      data = null;
    }
  } else {
    try {
      data = await res.text();
    } catch {
      data = null;
    }
  }

  if (!res.ok) {
    let errorMsg = `Request failed with status ${res.status}`;
    if (data && typeof data === "object") {
      if (data.reason) {
        errorMsg = data.reason;
      } else if (data.message) {
        errorMsg = data.message;
      } else if (data.detail) {
        errorMsg = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      }
    } else if (typeof data === "string" && data.trim()) {
      errorMsg = data.slice(0, 150);
    }

    const error = new Error(errorMsg);
    error.status = res.status;
    error.data = data;
    throw error;
  }

  return data;
}

export const api = {
  /**
   * Health check for API service and model readiness
   */
  async getHealth() {
    const res = await fetchWithTimeout(`${BACKEND_URL}/health`, {}, 5000);
    return handleResponse(res);
  },

  /**
   * Unified multimodal screening request (retinal image + optional vitals)
   */
  async screenMultimodal({
    imageFile = null,
    heartRate = null,
    spo2 = null,
    temperature = null,
    patientId = null,
    sessionId = null,
    includeGradcam = true,
  }) {
    const formData = new FormData();
    if (imageFile) {
      formData.append("image", imageFile);
    }
    if (heartRate !== null && heartRate !== undefined && heartRate !== "") {
      formData.append("heart_rate", heartRate);
    }
    if (spo2 !== null && spo2 !== undefined && spo2 !== "") {
      formData.append("spo2", spo2);
    }
    if (temperature !== null && temperature !== undefined && temperature !== "") {
      formData.append("temperature", temperature);
    }
    if (patientId) {
      formData.append("patient_id", patientId);
    }
    if (sessionId) {
      formData.append("session_id", sessionId);
    }
    formData.append("include_gradcam", includeGradcam);

    const res = await fetchWithTimeout(`${API_BASE}/screen`, {
      method: "POST",
      body: formData,
    });
    return handleResponse(res);
  },

  /**
   * Independent eye-disease prediction endpoint
   */
  async predictEye({ imageFile, sessionId = null, includeGradcam = true }) {
    const formData = new FormData();
    formData.append("image", imageFile);
    if (sessionId) {
      formData.append("session_id", sessionId);
    }
    formData.append("include_gradcam", includeGradcam);

    const res = await fetchWithTimeout(`${API_BASE}/predict_eye`, {
      method: "POST",
      body: formData,
    });
    return handleResponse(res);
  },

  /**
   * Independent physiological vital signs assessment endpoint
   */
  async submitVitals({ sessionId = null, heartRate, spo2, temperature }) {
    const res = await fetchWithTimeout(`${API_BASE}/submit_vitals`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        heart_rate_bpm: parseFloat(heartRate),
        spo2_percent: parseFloat(spo2),
        temperature_celsius: parseFloat(temperature),
      }),
    });
    return handleResponse(res);
  },

  /**
   * Patient management
   */
  async createPatient({ name, age, gender, contact }) {
    const res = await fetchWithTimeout(`${API_BASE}/patients`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        age: age ? parseInt(age, 10) : null,
        gender: gender || null,
        contact: contact || null,
      }),
    });
    return handleResponse(res);
  },

  async getPatients() {
    const res = await fetchWithTimeout(`${API_BASE}/patients`, {}, 5000);
    return handleResponse(res);
  },

  async getPatientSessions(patientId) {
    const res = await fetchWithTimeout(`${API_BASE}/patients/${patientId}/sessions`);
    return handleResponse(res);
  },

  /**
   * Sessions management
   */
  async getSessions() {
    const res = await fetchWithTimeout(`${API_BASE}/sessions`, {}, 8000);
    return handleResponse(res);
  },

  getSessionReportUrl(sessionId, format = "pdf") {
    return `${API_BASE}/sessions/${sessionId}/report?format=${format}`;
  },
};
