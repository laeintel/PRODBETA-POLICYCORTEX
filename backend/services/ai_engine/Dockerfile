# AI Engine Service Dockerfile
FROM python:3.11-slim

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

# Copy the entire backend directory structure
# Copy shared modules
COPY shared /app/shared/

# Copy the service application
COPY services/ai_engine /app/

# Install Python dependencies using minimal requirements for container builds
# Use minimal requirements by default, with fallback to basic requirements if needed
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f "/app/services/ai_engine/requirements-minimal.txt" ]; then \
        echo "Installing minimal requirements..." && \
        pip install --no-cache-dir -r /app/services/ai_engine/requirements-minimal.txt; \
    else \
        echo "Minimal requirements not found, installing basic FastAPI..." && \
        pip install --no-cache-dir fastapi==0.104.1 uvicorn[standard]==0.24.0 pydantic==2.4.2; \
    fi

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
CMD ["python", "-m", "uvicorn", "services.ai_engine.main:app", "--host", "0.0.0.0", "--port", "8002"]