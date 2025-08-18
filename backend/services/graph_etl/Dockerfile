FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY etl_processor.py .
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 etluser && chown -R etluser:etluser /app
USER etluser

# Run ETL processor
CMD ["python", "etl_processor.py"]