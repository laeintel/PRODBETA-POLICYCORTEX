# AI Engine Service Dockerfile
FROM python:3.11-slim

# Cache bust argument to force fresh builds
ARG CACHEBUST=1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SERVICE_NAME=ai_engine
ENV SERVICE_PORT=8002
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements for container builds
COPY backend/services/ai_engine/requirements-minimal.txt /app/requirements.txt

# Install Python dependencies (minimal set for container startup)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy shared modules
COPY backend/shared /app/shared/

# Copy the service application
COPY backend/services/ai_engine /app/

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs

# Create non-root user
RUN useradd --create-home --shell /bin/bash aiengine
RUN chown -R aiengine:aiengine /app
USER aiengine


# Expose port
EXPOSE $SERVICE_PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]