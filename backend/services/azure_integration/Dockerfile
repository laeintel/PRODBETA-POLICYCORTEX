# Azure Integration Service Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

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
COPY services/azure_integration /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/services/azure_integration/requirements.txt

# Create non-root user
RUN groupadd -r azureapp && useradd -r -g azureapp azureapp
RUN chown -R azureapp:azureapp /app
USER azureapp

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application from root directory
CMD ["python", "-m", "uvicorn", "services.azure_integration.main:app", "--host", "0.0.0.0", "--port", "8001"]