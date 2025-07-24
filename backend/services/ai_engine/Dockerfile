# AI Engine Service Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SERVICE_NAME=ai_engine
ENV SERVICE_PORT=8002
ENV PYTHONPATH=/app:$PYTHONPATH

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
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/services/ai_engine/requirements.txt

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs

# Create non-root user
RUN useradd --create-home --shell /bin/bash aiengine
RUN chown -R aiengine:aiengine /app
USER aiengine

# Set working directory to the service
WORKDIR /app/services/ai_engine

# Expose port
EXPOSE $SERVICE_PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]