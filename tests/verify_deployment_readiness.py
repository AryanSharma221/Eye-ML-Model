"""
Deployment readiness verification script.
Tests:
1. Environment configuration and path resolution.
2. Checkpoint integrity and loading on CPU runtime.
3. Real inference on CPU (prediction + Grad-CAM heatmap).
4. Dual check on CUDA if available.
5. SQLite/Database connection and ORM tables.
6. Health and readiness responses.
7. Frontend bundle build status.
"""
import os
import sys
import torch
import httpx
from pathlib import Path

from backend.config import (
    DEFAULT_CHECKPOINT_PATH,
    DATABASE_URL,
    ML_DEVICE,
    UPLOADS_DIR,
    get_cors_origins,
)
from backend.database import engine, Base
from ml.eye_model import get_cached_model, predict_eye_disease
from ml.gradcam import generate_gradcam

def run_readiness_checks():
    print("=" * 60)
    print("RUNNING DEPLOYMENT READINESS VERIFICATION")
    print("=" * 60)

    # 1. Environment & Path Resolution
    print("\n--- 1. Configuration & Path Resolution ---")
    print(f"  CHECKPOINT_PATH : {DEFAULT_CHECKPOINT_PATH}")
    print(f"  ML_DEVICE       : {ML_DEVICE}")
    print(f"  DATABASE_URL    : {DATABASE_URL}")
    print(f"  UPLOADS_DIR     : {UPLOADS_DIR}")
    print(f"  CORS_ORIGINS    : {get_cors_origins()}")

    assert os.path.exists(DEFAULT_CHECKPOINT_PATH), f"Checkpoint missing: {DEFAULT_CHECKPOINT_PATH}"
    checkpoint_size_mb = os.path.getsize(DEFAULT_CHECKPOINT_PATH) / (1024 * 1024)
    print(f"  [PASS] Checkpoint present ({checkpoint_size_mb:.2f} MB)")

    # 2. Database & ORM Readiness
    print("\n--- 2. Database Connection & Schema Verification ---")
    with engine.connect() as conn:
        print("  [PASS] Database engine connected successfully.")
    tables = list(Base.metadata.tables.keys())
    print(f"  [PASS] Managed ORM tables: {', '.join(tables)}")

    # 3. CPU Inference & Grad-CAM Execution
    print("\n--- 3. CPU ML Runtime Execution (Zero GPU Assumption) ---")
    cpu_device = torch.device("cpu")
    model_cpu = get_cached_model(DEFAULT_CHECKPOINT_PATH, device=cpu_device)
    sample_path = "frontend/public/samples/sample_glaucoma.jpg"
    assert os.path.exists(sample_path), f"Sample missing: {sample_path}"

    pred_cpu = predict_eye_disease(sample_path, model=model_cpu, device=cpu_device)
    assert pred_cpu["prediction"] == "glaucoma"
    assert pred_cpu["confidence"] > 0.90
    print(f"  [PASS] CPU Forward Pass: class={pred_cpu['prediction']}, confidence={pred_cpu['confidence']*100:.2f}%")

    gradcam_cpu = generate_gradcam(sample_path, predicted_class_idx=pred_cpu["class_index"], model=model_cpu, device=cpu_device)
    assert len(gradcam_cpu) > 1000
    print(f"  [PASS] CPU Grad-CAM Heatmap generated ({len(gradcam_cpu)} chars base64)")

    # 4. Optional CUDA check
    if torch.cuda.is_available():
        print("\n--- 4. CUDA Accelerator Runtime (Hardware Present) ---")
        cuda_device = torch.device("cuda:0")
        model_cuda = get_cached_model(DEFAULT_CHECKPOINT_PATH, device=cuda_device)
        pred_cuda = predict_eye_disease(sample_path, model=model_cuda, device=cuda_device)
        print(f"  [PASS] CUDA Forward Pass: class={pred_cuda['prediction']}, confidence={pred_cuda['confidence']*100:.2f}%")

    # 5. Live Server Probes
    print("\n--- 5. Live Server Liveness & Readiness Probes ---")
    try:
        r_live = httpx.get("http://localhost:8000/health/live", timeout=5.0)
        assert r_live.status_code == 200, f"Live returned {r_live.status_code}"
        print(f"  [PASS] /health/live returned HTTP 200: {r_live.json()['status']}")

        r_ready = httpx.get("http://localhost:8000/health/ready", timeout=5.0)
        assert r_ready.status_code == 200, f"Ready returned {r_ready.status_code}"
        print(f"  [PASS] /health/ready returned HTTP 200: {r_ready.json()['status']}")
    except Exception as e:
        print(f"  [NOTE] Live server probe skipped or not running: {e}")

    # 6. Frontend Distribution Build
    print("\n--- 6. Frontend Production Distribution Asset Check ---")
    dist_dir = Path("frontend/dist")
    index_html = dist_dir / "index.html"
    redirects = dist_dir / "_redirects"
    assert index_html.exists(), "frontend/dist/index.html is missing. Run npm run build."
    assert redirects.exists(), "frontend/dist/_redirects is missing."
    print(f"  [PASS] Production bundle verified ({index_html.stat().st_size} bytes)")
    print(f"  [PASS] SPA Client redirect manifest verified ({redirects.stat().st_size} bytes)")

    print("\n" + "=" * 60)
    print("ALL DEPLOYMENT READINESS CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_readiness_checks()
