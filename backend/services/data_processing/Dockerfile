# Use Python 3.11 slim image as base
FROM python:3.11-slim

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

# Copy the entire backend directory structure
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/services/data_processing/requirements.txt

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

# Run the application from root directory
CMD ["python", "-m", "uvicorn", "services.data_processing.main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "4"]