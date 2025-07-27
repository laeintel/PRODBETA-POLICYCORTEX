# API Gateway Service Dockerfile
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
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/services/api_gateway/requirements.txt

# Create non-root user
RUN groupadd -r apigateway && useradd -r -g apigateway apigateway
RUN chown -R apigateway:apigateway /app
USER apigateway

# Set default port
ENV PORT=8000
ENV SERVICE_HOST=0.0.0.0

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the simplified application
CMD ["python", "-m", "uvicorn", "services.api_gateway.main_simple:app", "--host", "0.0.0.0", "--port", "8000"]