/**
 * Centralized API client for the Multimodal Health Screening Frontend.
 * Interacts with the FastAPI backend across system health, patient intake,
 * eye disease screening, physiological vitals triage, and session reporting.
 *
 * Observability & Error Handling:
 * - Differentiates network timeout (504), service offline (network error),
 *   model not ready (503), quality gate rejections (400), and internal failures (500).
 * - Extracts and propagates correlation request IDs (X-Request-ID).
 * - Provides concise user-facing messages while isolating technical diagnostics.
 */

// Use Vite environment variable with graceful fallback to relative /api/v1
const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL || "").replace(/\/+$/, "");
const API_BASE = `${BACKEND_URL}/api/v1`;

const DEFAULT_TIMEOUT_MS = 30000; // 30s timeout for deep learning inference & Grad-CAM

/**
 * Fetch wrapper with timeout and informative error categorization.
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
      timeoutError.errorType = "NETWORK_TIMEOUT";
      timeoutError.userMessage = "The screening request timed out after 30 seconds. The deep learning service may be under high load. Please try again.";
      throw timeoutError;
    }
    // Network failure (server down, CORS blocked, unreachable)
    const netError = new Error(
      `Unable to reach the backend service at ${BACKEND_URL || "the configured server"}. Please ensure the server is running.`
    );
    netError.status = 0;
    netError.isNetworkError = true;
    netError.errorType = "BACKEND_UNAVAILABLE";
    netError.userMessage = "The health screening service is currently unreachable. Please check your network connection or ensure the backend server is running.";
    throw netError;
  }
}

/**
 * Parses response JSON safely or constructs an informative sanitized error object.
 */
async function handleResponse(res) {
  const contentType = res.headers.get("content-type");
  const requestId = res.headers.get("x-request-id") || null;
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
    let errorType = "API_ERROR";
    let userMessage = "";

    // 1. Model Not Ready / Checkpoint Missing (HTTP 503)
    if (res.status === 503) {
      errorType = "MODEL_NOT_READY";
      userMessage = "The AI screening model is currently initializing or unavailable. Please wait a moment and try again.";
    }
    // 2. Quality Gate Rejection or Bad Input (HTTP 400)
    else if (res.status === 400) {
      if (data && (data.status === "rejected" || data.reason)) {
        errorType = "QUALITY_GATE_REJECTED";
        userMessage = data.reason || "The uploaded image did not meet clinical quality standards. Please submit a clearer fundus photograph.";
      } else {
        errorType = "BAD_REQUEST";
        userMessage = data?.detail || data?.message || "Invalid input provided. Please verify image file and vital values.";
      }
    }
    // 3. Validation Error (HTTP 422)
    else if (res.status === 422) {
      errorType = "VALIDATION_ERROR";
      userMessage = "Input validation failed. Please check your form entries.";
    }
    // 4. Server Internal Error (HTTP 500)
    else if (res.status === 500) {
      errorType = "INFERENCE_FAILURE";
      userMessage = "An unexpected error occurred during retinal analysis. Please try again or contact system support.";
    }
    else {
      userMessage = data?.message || data?.detail || `Request failed with HTTP status ${res.status}.`;
    }

    const error = new Error(userMessage);
    error.status = res.status;
    error.errorType = errorType;
    error.userMessage = userMessage;
    error.requestId = requestId || data?.request_id || null;
    error.data = data;
    throw error;
  }

  // Handle empty or unparseable response when status is OK
  if (data === null && res.status !== 204) {
    const parseError = new Error("Invalid response format received from server.");
    parseError.status = res.status;
    parseError.errorType = "INVALID_RESPONSE_FORMAT";
    parseError.userMessage = "Received an invalid or unparseable response from the server.";
    parseError.requestId = requestId;
    throw parseError;
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
   * Liveness probe for process uptime
   */
  async getLiveness() {
    const res = await fetchWithTimeout(`${BACKEND_URL}/health/live`, {}, 3000);
    return handleResponse(res);
  },

  /**
   * Deep readiness probe checking model pre-warming and database connectivity
   */
  async getReadiness() {
    const res = await fetchWithTimeout(`${BACKEND_URL}/health/ready`, {}, 5000);
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
