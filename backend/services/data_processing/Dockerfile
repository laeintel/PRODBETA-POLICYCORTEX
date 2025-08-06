# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Cache bust argument to force fresh builds
ARG CACHEBUST=1

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:$PYTHONPATH \
    SERVICE_NAME=data_processing \
    SERVICE_PORT=8003

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    wget \
    ca-certificates \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Java separately (if needed for Spark)
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     openjdk-11-jre-headless \
#     && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable (if Java is installed)
# ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Copy requirements first for better caching
COPY services/data_processing/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy shared modules
COPY shared /app/shared/

# Copy the service application
COPY services/data_processing /app/

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/tmp \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=5 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

# Expose port
EXPOSE ${SERVICE_PORT}

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]