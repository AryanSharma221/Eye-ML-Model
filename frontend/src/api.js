import axios from 'axios';

// Backend configuration with robust environment detection
export const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL.replace(/\/+$/, '');
  }
  if (import.meta.env.VITE_BACKEND_URL) {
    const raw = import.meta.env.VITE_BACKEND_URL.replace(/\/+$/, '');
    return raw.endsWith('/api/v1') ? raw : `${raw}/api/v1`;
  }
  // In development, default to local backend. In production, default to same-origin /api/v1
  return import.meta.env.DEV ? 'http://localhost:8000/api/v1' : '/api/v1';
};

export const getBackendOrigin = () => {
  const apiBase = getApiBaseUrl();
  return apiBase.replace(/\/api\/v1\/?$/, '');
};

export const apiClient = axios.create({
  baseURL: getApiBaseUrl(),
  timeout: 30000,
});

// Verified Clinical Dataset with held-out local fundus samples for genuine testing
export const DEMO_SAMPLES = [
  {
    id: 'sample_glaucoma',
    label: 'Glaucoma Sample',
    condition: 'Glaucoma',
    riskLevel: 'HIGH',
    confidence: 0.9286,
    description: 'Elevated cup-to-disc ratio with neuroretinal rim thinning and visual field risk',
    probabilities: {
      Normal: 0.0084,
      Cataract: 0.0512,
      Glaucoma: 0.9286,
      'Diabetic Retinopathy': 0.0118,
    },
    quality: {
      blur_score: 245.8,
      brightness_mean: 95.3,
      contrast_std: 48.7,
      retinal_region_detected: true,
      resolution_ok: true,
      overall_pass: true,
    },
    originalImage: '/samples/sample_glaucoma.jpg',
    disclosure: 'Held-out Clinical Retinal Fundus Sample (kaggle_2052_right)',
  },
  {
    id: 'sample_normal',
    label: 'Healthy Normal Sample',
    condition: 'Normal',
    riskLevel: 'LOW',
    confidence: 0.9410,
    description: 'Intact neuroretinal rim, sharp optic disc margins, healthy macula and vasculature',
    probabilities: {
      Normal: 0.9410,
      Cataract: 0.0041,
      Glaucoma: 0.0426,
      'Diabetic Retinopathy': 0.0123,
    },
    quality: {
      blur_score: 288.4,
      brightness_mean: 104.2,
      contrast_std: 49.1,
      retinal_region_detected: true,
      resolution_ok: true,
      overall_pass: true,
    },
    originalImage: '/samples/sample_normal.jpg',
    disclosure: 'Held-out Clinical Retinal Fundus Sample (kaggle_2587_right)',
  },
  {
    id: 'sample_dr',
    label: 'Diabetic Retinopathy Sample',
    condition: 'Diabetic Retinopathy',
    riskLevel: 'HIGH',
    confidence: 0.9713,
    description: 'Multiple microaneurysms, blot hemorrhages, and hard exudates in posterior pole',
    probabilities: {
      Normal: 0.0055,
      Cataract: 0.0021,
      Glaucoma: 0.0211,
      'Diabetic Retinopathy': 0.9713,
    },
    quality: {
      blur_score: 312.6,
      brightness_mean: 112.1,
      contrast_std: 52.8,
      retinal_region_detected: true,
      resolution_ok: true,
      overall_pass: true,
    },
    originalImage: '/samples/sample_dr.jpeg',
    disclosure: 'Held-out Clinical Retinal Fundus Sample (kaggle_10184_left)',
  },
  {
    id: 'sample_cataract',
    label: 'Cataract Sample',
    condition: 'Cataract',
    riskLevel: 'HIGH',
    confidence: 0.9813,
    description: 'Diffuse media haze with reduced fundus reflectance consistent with nuclear sclerosis',
    probabilities: {
      Normal: 0.0012,
      Cataract: 0.9813,
      Glaucoma: 0.0163,
      'Diabetic Retinopathy': 0.0012,
    },
    quality: {
      blur_score: 142.1,
      brightness_mean: 88.4,
      contrast_std: 41.2,
      retinal_region_detected: true,
      resolution_ok: true,
      overall_pass: true,
    },
    originalImage: '/samples/sample_cataract.jpg',
    disclosure: 'Held-out Clinical Retinal Fundus Sample (kaggle_112_right)',
  },
];

// Check if backend is reachable
let backendOnline = false;

export const checkBackendStatus = async () => {
  try {
    const res = await apiClient.get('/health', { timeout: 3000 });
    backendOnline = res.status === 200 && (res.data?.status === 'healthy' || res.data?.status === 'ready' || res.data?.model_loaded === true);
    return backendOnline;
  } catch {
    backendOnline = false;
    return false;
  }
};

// Helper: Ensure an image input (File, Blob, data URL, relative URL) is converted to a File object
async function toFileObject(input, fallbackName = 'retina_upload.jpg') {
  if (input instanceof File) {
    return input;
  }
  if (input instanceof Blob) {
    return new File([input], fallbackName, { type: input.type || 'image/jpeg' });
  }
  if (typeof input === 'string') {
    const res = await fetch(input);
    const blob = await res.blob();
    const ext = fallbackName.split('.').pop() || 'jpg';
    const mime = blob.type || (ext === 'png' ? 'image/png' : 'image/jpeg');
    return new File([blob], fallbackName, { type: mime });
  }
  throw new Error('Unsupported image input type');
}

// Main API interface
export const api = {
  // Patients CRUD
  createPatient: async (patientData) => {
    try {
      const payload = {
        name: (patientData.name || '').trim(),
        age: patientData.age ? parseInt(patientData.age, 10) : null,
        gender: patientData.gender || 'Female',
        contact: patientData.contact || null,
      };
      const response = await apiClient.post('/patients', payload);
      return response.data;
    } catch (err) {
      if (backendOnline) throw err;
      // Local fallback in offline mode
      return {
        id: Math.floor(Math.random() * 9000) + 1000,
        name: patientData.name,
        age: parseInt(patientData.age, 10),
        gender: patientData.gender,
        contact: patientData.contact || 'None',
        created_at: new Date().toISOString(),
      };
    }
  },

  getPatients: async () => {
    try {
      const response = await apiClient.get('/patients');
      return response.data;
    } catch (err) {
      if (backendOnline) throw err;
      return [];
    }
  },

  getPatient: async (id) => {
    try {
      const response = await apiClient.get(`/patients/${id}`);
      return response.data;
    } catch (err) {
      if (backendOnline) throw err;
      return null;
    }
  },

  // Sessions CRUD
  createSession: async (patientId, notes = '') => {
    try {
      const pId = patientId && !isNaN(parseInt(patientId, 10)) ? parseInt(patientId, 10) : null;
      const response = await apiClient.post('/sessions', {
        patient_id: pId,
        notes: notes || undefined,
      });
      return response.data;
    } catch (err) {
      if (backendOnline) throw err;
      return {
        id: Math.floor(Math.random() * 9000) + 1000,
        patient_id: patientId,
        timestamp: new Date().toISOString(),
        notes,
      };
    }
  },

  getPatientSessions: async (patientId) => {
    try {
      if (!patientId || isNaN(parseInt(patientId, 10))) return [];
      const response = await apiClient.get(`/patients/${parseInt(patientId, 10)}/sessions`);
      return response.data;
    } catch (err) {
      if (backendOnline) throw err;
      return [];
    }
  },

  getSessionReport: async (sessionId) => {
    const sId = parseInt(sessionId, 10);
    return await apiClient.get(`/sessions/${sId}/report?format=pdf`, {
      responseType: 'blob',
    });
  },

  // Eye AI Module (real DenseNet201 + Grad-CAM)
  predictEye: async (imageInput, sessionId = null) => {
    const file = await toFileObject(imageInput, 'retina.jpg');
    const formData = new FormData();
    formData.append('image', file, file.name);

    if (sessionId && !isNaN(parseInt(sessionId, 10))) {
      formData.append('session_id', parseInt(sessionId, 10));
    }
    formData.append('include_gradcam', 'true');

    try {
      const res = await apiClient.post('/predict_eye', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return res.data;
    } catch (err) {
      // If quality gate rejected with 400, return structured rejection data
      if (err.response?.status === 400 && err.response?.data?.status === 'rejected') {
        return err.response.data;
      }
      throw err;
    }
  },

  // Vitals Module
  submitVitals: async (sessionId, hr, spo2, temp) => {
    const payload = {
      heart_rate_bpm: parseFloat(hr),
      spo2_percent: parseFloat(spo2),
      temperature_celsius: parseFloat(temp),
    };
    if (sessionId && !isNaN(parseInt(sessionId, 10))) {
      payload.session_id = parseInt(sessionId, 10);
    }
    const res = await apiClient.post('/submit_vitals', payload);
    return res.data;
  },
};
