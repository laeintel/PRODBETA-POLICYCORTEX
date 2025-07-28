"""
Simplified Data Processing Service for Container Apps deployment.
Basic health checks and data pipeline endpoints without heavy dependencies.
"""

import os
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Simple configuration from environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
SERVICE_NAME = os.getenv("SERVICE_NAME", "data-processing")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8003"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# FastAPI app
app = FastAPI(
    title="PolicyCortex Data Processing Service",
    description="Data Processing and ETL Pipeline Service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": SERVICE_NAME,
        "environment": ENVIRONMENT,
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "service": SERVICE_NAME,
        "environment": ENVIRONMENT
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PolicyCortex Data Processing Service",
        "status": "running",
        "environment": ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {
        "api_version": "v1",
        "status": "operational",
        "service": "data_processing",
        "environment": ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/pipelines/status")
async def pipelines_status():
    """Data pipelines status endpoint."""
    return {
        "pipelines": {
            "etl_pipeline": "running",
            "stream_processor": "running",
            "data_aggregator": "running",
            "data_validator": "ready"
        },
        "connectors": ["azure_sql", "cosmos_db", "azure_storage"],
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        log_level=LOG_LEVEL.lower()
    )
