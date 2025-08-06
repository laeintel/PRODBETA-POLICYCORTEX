# Notification Service Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:$PYTHONPATH

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
COPY services/notification /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create non-root user
RUN groupadd -r notification && useradd -r -g notification notification
RUN chown -R notification:notification /app
USER notification

# Set default port
ENV PORT=8005
ENV SERVICE_HOST=0.0.0.0

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]