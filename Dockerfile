# ==============================================================================
# Multimodal Health Screening Platform - Multi-Stage Production Dockerfile
# Stage 1: Builds the React 18 + Vite 5 Frontend
# Stage 2: Sets up Python 3.12, PyTorch DenseNet201 ML Layer, and FastAPI Backend
# ==============================================================================

# ------------------------------------------------------------------------------
# STAGE 1: Frontend Build
# ------------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder
WORKDIR /build

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# Build production bundle (configured for same-origin /api/v1 calls)
ENV VITE_BACKEND_URL=""
RUN npm run build

# ------------------------------------------------------------------------------
# STAGE 2: Backend & ML Runtime
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runner

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    ML_DEVICE=cpu

# Install system dependencies required for OpenCV, Pillow, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install PyTorch CPU wheels (avoids multi-GB CUDA wheels on CPU hosts)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Python backend and ML dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML and Backend modules, configuration, and champion checkpoint
COPY pyproject.toml ./
COPY ml/ ./ml/
COPY backend/ ./backend/

# Copy compiled frontend from Stage 1 into frontend/dist for single-container serving
COPY --from=frontend-builder /build/dist ./frontend/dist

# Install project package in editable mode for module resolution
RUN pip install --no-cache-dir -e .

# Create non-root user and ensure upload directories have write permissions
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    mkdir -p /app/backend/uploads/fundus /app/backend/uploads/gradcam && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Liveness probe: checks whether web server responds within 5 seconds
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Start production ASGI server binding to 0.0.0.0 and configurable $PORT
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
