# ML Service Docker Container
# Patent #4: Predictive Policy Compliance Engine

# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    MLFLOW_TRACKING_URI=sqlite:///app/mlflow.db

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libssl-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy Python requirements
COPY requirements-ml.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir -r requirements-ml.txt

# Install TensorRT for optimization (optional, for production)
# RUN pip3 install --no-cache-dir tensorrt

# Copy ML models and services
COPY backend/services/ml_models/ /app/ml_models/
COPY backend/migrations/create_ml_tables.sql /app/migrations/
COPY backend/services/websocket_server.py /app/

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/checkpoints

# Create non-root user for security
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app

USER mluser

# Expose ports
EXPOSE 8080 8765 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command - start prediction server
CMD ["python3", "-m", "ml_models.prediction_serving"]