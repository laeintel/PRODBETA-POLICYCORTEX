FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY explainability_service.py .

# Create non-root user
RUN useradd -m -u 1000 explainer && chown -R explainer:explainer /app
USER explainer

# Expose port
EXPOSE 8000

# Run service
CMD ["uvicorn", "explainability_service:app", "--host", "0.0.0.0", "--port", "8000"]