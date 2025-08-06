FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# Copy shared modules
COPY ../../shared /app/shared

# Copy service code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV SERVICE_NAME=compliance_engine

# Expose port
EXPOSE 8006

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]